
##Global token management
import os
import csv
import time
import math
import shutil
import zipfile
import requests
from datetime import datetime
from io import BytesIO

import numpy as np
import rasterio
from rasterio.transform import xy as transform_xy
from scipy.ndimage import label
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from pyproj import Transformer
import folium
from folium.plugins import MarkerCluster, HeatMap
import geopandas as gpd
import concurrent.futures
from collections import defaultdict

# Ensure proxies are disabled
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# =============================================================================
# 1. USER CONFIGURATION & CONSTANTS
# =============================================================================

USERNAME = "sethstock_94@hotmail.com"
PASSWORD = "ComradeHobo94!"

# API endpoints and token URL
odata_search_base = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="
odata_download_base = "https://download.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# Date ranges (ISO 8601)
start_date     = "2022-03-25T00:00:00.000Z"    # Monitoring period start
end_date       = "2022-03-30T23:59:59.999Z"      # Monitoring period end
baseline_start = "2019-03-25T00:00:00.000Z"      # Baseline start
baseline_end   = "2019-03-30T23:59:59.999Z"      # Baseline end

max_cloud = 20.0  # Maximum allowed cloud cover

# Site locations in GeoJSON
SITE_FILES = [
    "C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/geojson/mre_production_sites.geojson",
    "C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/geojson/pingtan_staging.geojson",
    "C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/geojson/pla_73rd_group_bases.geojson",
    "C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/geojson/storage_depots.geojson"
]

# Output files and parameters for maps & CSV logging
DETECTIONS_CSV = "2019_2022_detection_events.csv"
DETECTION_MAP_PREFIX = "TruckDetections_Map"
HEATMAP_MAP_PREFIX   = "TruckHeatmap_Map"
MAX_MAP_SIZE = 100 * 1024 * 1024  # 100 MB
SITE_DISTANCE_KM = 1.0  # km threshold for a detection to be considered "near" a site

# Directory to download and extract Sentinel SAFE products
download_dir = "sentinel_downloads"
os.makedirs(download_dir, exist_ok=True)




# =============================================================================
# 2. GLOBAL TOKEN MANAGEMENT
# =============================================================================
# The TokenManager obtains a token and creates a requests.Session using it.
# If a request returns 401, you can refresh the token.
class TokenManager:
    def __init__(self, username, password, token_url, max_retries=3):
        self.username = username
        self.password = password
        self.token_url = token_url.strip()
        self.max_retries = max_retries
        self.token = None
        self.session = None

    def get_new_token(self):
        data = {
            "grant_type": "password",
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.token_url, data=data, headers=headers, timeout=5)
                resp.raise_for_status()
                token = resp.json().get("access_token")
                if token:
                    print("Successfully fetched new token.")
                    return token
                else:
                    raise ValueError("No access_token found in response.")
            except Exception as e:
                wait = 3 * (attempt + 1)
                print(f"⚠️ Token retry {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        print("❌ Final token fetch failed.")
        return None

    def create_session(self):
        self.token = self.get_new_token()
        if not self.token:
            return None
        sess = requests.Session()
        sess.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        })
        self.session = sess
        return self.session

    def get_session(self):
        # If session is already created, return it;
        # otherwise, create a new one.
        if self.session is None:
            return self.create_session()
        return self.session

    def refresh_token(self):
        print("Refreshing token...")
        new_token = self.get_new_token()
        if new_token:
            self.token = new_token
            if self.session is None:
                self.session = requests.Session()
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
        else:
            print("Failed to refresh token.")
        return self.token

# Instantiate a global TokenManager
token_manager = TokenManager(USERNAME, PASSWORD, TOKEN_URL)

# Helper function to wrap requests so that we refresh token on 401
def request_with_token(method, url, **kwargs):
    sess = token_manager.get_session()
    response = sess.request(method, url, **kwargs)
    if response.status_code == 401:
        print("401 Unauthorized detected. Refreshing token and retrying...")
        token_manager.refresh_token()
        sess = token_manager.get_session()  # update session after refresh
        response = sess.request(method, url, **kwargs)
    return response

# =============================================================================
# 3. LOAD SITE LOCATIONS FROM GEOJSON FILES
# =============================================================================

print("Loading site location data...")
sites = []
for sf in SITE_FILES:
    try:
        gdf = gpd.read_file(sf)
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom and geom.geom_type != 'Point':
                geom = geom.centroid
            if not geom:
                continue
            lat, lon = geom.y, geom.x
            name = next((row[k] for k in ['name', 'Name', 'site', 'Site'] if k in row and row[k]),
                        f"Site_{len(sites)+1}")
            stype = next((row[k] for k in ['type', 'Type', 'category', 'Category'] if k in row and row[k]),
                         "Site")
            sites.append({"name": str(name), "type": str(stype), "lat": lat, "lon": lon})
    except Exception as e:
        print(f"❌ Couldn't read {sf}: {e}")
site_type_colors = {"Production": "red", "Logistics": "blue", "Storage": "green", "Deployment": "orange", "Military": "black"}
print(f"Loaded {len(sites)} site(s).")

def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    return R * 2 * math.asin(math.sqrt(math.sin((lat2 - lat1)/2)**2 +
                                        math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1)/2)**2))

def find_nearest_site(lat, lon):
    nearest_site = None
    nearest_dist = float('inf')
    for site in sites:
        dist = haversine(lon, lat, site['lon'], site['lat'])
        if dist < nearest_dist:
            nearest_site = site
            nearest_dist = dist
    if nearest_dist <= SITE_DISTANCE_KM:
        return nearest_site, nearest_dist
    else:
        return None, None

# =============================================================================
# 4. TRUCK DETECTION MODEL FUNCTION
# =============================================================================
def detect_trucks(B02, B03, B04, B08, B11):
    scale = 1.0
    if B02.max() > 1.0:
        scale = 10000.0

    mask = np.ones(B02.shape, dtype=bool)
    mask &= (B02 > int(0.04 * scale))
    mask &= (B03 > int(0.04 * scale))
    mask &= (B04 > int(0.04 * scale))
    mask &= (B04 < int(0.15 * scale))
    mask &= (B03 < int(0.15 * scale))
    mask &= (B02 < int(0.4 * scale))
    mask &= (3 * B08.astype(np.int32) < 17 * B04.astype(np.int32))
    mask &= (999 * B03.astype(np.int32) < 1001 * B08.astype(np.int32))
    mask &= (9999 * B03.astype(np.int32) < 10001 * B11.astype(np.int32))
    mask &= (B11 > int(0.05 * scale))
    mask &= (B11 < int(0.55 * scale))
    mask &= ((B02.astype(np.int32) - B03.astype(np.int32)) > int(0.05 * scale))
    mask &= ((B02.astype(np.int32) - B04.astype(np.int32)) > int(0.1 * scale))
    detections = []
    if not mask.any():
        return detections

    coords = np.argwhere(mask)
    coords_set = {(int(r), int(c)) for r, c in coords}
    B02_int = B02.astype(np.int32)
    B03_int = B03.astype(np.int32)
    B04_int = B04.astype(np.int32)

    while coords_set:
        r, c = coords_set.pop()
        cluster_pixels = [(r, c)]
        stack = [(r, c)]
        conf_max = 0
        while stack:
            pr, pc = stack.pop()
            diff_b02_b03 = B02_int[pr, pc] - B03_int[pr, pc]
            diff_b02_b04 = B02_int[pr, pc] - B04_int[pr, pc]
            pixel_score = diff_b02_b03 + diff_b02_b04
            conf_max = max(conf_max, pixel_score)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = pr + dr, pc + dc
                    if (nr, nc) in coords_set:
                        coords_set.remove((nr, nc))
                        stack.append((nr, nc))
                        cluster_pixels.append((nr, nc))
        if not cluster_pixels:
            continue
        rs = [p[0] for p in cluster_pixels]
        cs = [p[1] for p in cluster_pixels]
        row_center = sum(rs) / len(rs)
        col_center = sum(cs) / len(cs)
        conf_value = conf_max / (0.15 * scale) if scale != 1.0 else conf_max / 0.15
        conf_value = min(conf_value, 1.0)
        detections.append((row_center, col_center, conf_value))
    return detections

# =============================================================================
# 5. QUERY CONSTRUCTION FOR THE CDSE API
# =============================================================================
# Mainland China polygon
# =============================================================================
# 5. QUERY CONSTRUCTION FOR THE CDSE API
#    (polygon‐aware tiling + Xinjiang/Tibet exclusions)
# =============================================================================
from shapely.geometry import Polygon

# Mainland China (excludes Taiwan)
mainland_china_polygon = Polygon([
    (73.0, 18.0),
    (73.0, 54.0),
    (123.0, 54.0),
    (123.0, 24.5),
    (119.5, 24.5),
    (119.5, 18.0),
    (73.0, 18.0)
])

# Exclusion zones as boxes
xinjiang_polygon = Polygon([
    (73.0, 35.0),
    (96.0, 35.0),
    (96.0, 49.5),
    (73.0, 49.5),
    (73.0, 35.0)
])
tibet_polygon = Polygon([
    (78.0, 26.0),
    (99.0, 26.0),
    (99.0, 37.0),
    (78.0, 37.0),
    (78.0, 26.0)
])

min_lon, min_lat, max_lon, max_lat = mainland_china_polygon.bounds
TILE_SIZE = 5.0

tiles = []
lat_val = min_lat
while lat_val < max_lat:
    lon_val = min_lon
    lat_end = min(lat_val + TILE_SIZE, max_lat)
    while lon_val < max_lon:
        lon_end = min(lon_val + TILE_SIZE, max_lon)
        tile_poly = Polygon([
            (lon_val,     lat_val),
            (lon_end,     lat_val),
            (lon_end,     lat_end),
            (lon_val,     lat_end),
            (lon_val,     lat_val)
        ])
        # include only if in China AND not in Xinjiang or Tibet
        if (tile_poly.intersects(mainland_china_polygon)
            and not tile_poly.intersects(xinjiang_polygon)
            and not tile_poly.intersects(tibet_polygon)):
            tiles.append((lon_val, lat_val, lon_end, lat_end))
        lon_val += TILE_SIZE
    lat_val += TILE_SIZE

print(f"Prepared {len(tiles)} tile(s) for querying.")


def query_images_for_range(start_date, end_date):
    products_dict = {}
    for (lon1, lat1, lon2, lat2) in tiles:
        poly_wkt = f"{lon1} {lat1},{lon2} {lat1},{lon2} {lat2},{lon1} {lat2},{lon1} {lat1}"
        filter_parts = [
            "Collection/Name eq 'SENTINEL-2'",
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')",
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {max_cloud})",
            "not contains(Name, '_N9999_')",
            f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({poly_wkt}))')",
            f"ContentDate/Start ge {start_date} and ContentDate/Start lt {end_date}"
        ]
        query = odata_search_base + "%20and%20".join(filter_parts)
        query = query.replace(" ", "%20").replace("'", "%27")
        # Use our global session via token_manager for querying
        response = request_with_token("GET", query + "&$top=100", timeout=30)
        try:
            response.raise_for_status()
            prods = response.json().get("value", [])
            for prod in prods:
                # Attach tile information so we know which chunk it came from.
                prod["tile"] = (lon1, lat1, lon2, lat2)
                prod_id = prod["Id"]
                products_dict[prod_id] = prod
        except Exception as e:
            print(f"❌ CDSE query failed for tile {lat1}-{lon1}: {e}")
    image_list = []
    for prod in products_dict.values():
        ts = prod.get("ContentDate", {}).get("Start")
        if ts is None:
            continue
        image_list.append({
            "Id": prod["Id"],
            "Name": prod["Name"],
            "ContentDate": prod.get("ContentDate"),
            "DownloadUrl": f"{odata_download_base}({prod['Id']})/$value",
            "tile": prod["tile"]
        })
    image_list.sort(key=lambda x: x["ContentDate"].get("Start"))
    return image_list


print(f"Querying baseline images from {baseline_start} to {baseline_end}...")
baseline_images = query_images_for_range(baseline_start, baseline_end)
print(f"Found {len(baseline_images)} baseline image(s).")
print(f"Querying monitoring images from {start_date} to {end_date}...")
monitor_images = query_images_for_range(start_date, end_date)
print(f"Found {len(monitor_images)} monitoring image(s).")

# =============================================================================
# 6. PROCESSING A SINGLE PRODUCT
# =============================================================================
def process_product(prod):
    prod_id = prod["Id"]
    prod_name = prod["Name"]
    zip_url = prod["DownloadUrl"]
    zip_path = os.path.join(download_dir, f"{prod_id}.zip")
    extract_dir = os.path.join(download_dir, prod_id)
    try:
        # Use our global session instead of creating a new one
        sess = token_manager.get_session()
        if sess is None:
            print(f"Skipping {prod_name} due to failed authentication.")
            return []
        print(f"Downloading product {prod_name} (ID: {prod_id})...")
        r = request_with_token("GET", zip_url, stream = True, timeout = 30)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"Extracting product {prod_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
        if len(subdirs) == 1:
            safe_folder = os.path.join(extract_dir, subdirs[0])
        else:
            safe_folder = extract_dir# The extracted SAFE folder
        granule_dir = os.path.join(safe_folder, "GRANULE")
        if not os.path.isdir(granule_dir):
            raise FileNotFoundError("GRANULE directory not found in SAFE product.")
        gran_subdirs = os.listdir(granule_dir)
        if not gran_subdirs:
            raise FileNotFoundError("No granule subdirectory found in the GRANULE folder.")
        granule_subdir = os.path.join(granule_dir, gran_subdirs[0])
        img_data_dir = os.path.join(granule_subdir, "IMG_DATA")
        if not os.path.isdir(img_data_dir):
            raise FileNotFoundError("IMG_DATA directory not found in granule.")
        r10m_dir = os.path.join(img_data_dir, "R10m")
        r20m_dir = os.path.join(img_data_dir, "R20m")
        if not os.path.isdir(r10m_dir):
            raise FileNotFoundError("R10m folder not found in IMG_DATA.")
        if not os.path.isdir(r20m_dir):
            raise FileNotFoundError("R20m folder not found in IMG_DATA.")
        B02_path = B03_path = B04_path = B08_path = B11_path = None
        for fname in os.listdir(r10m_dir):
            if "_B02_10m" in fname:
                B02_path = os.path.join(r10m_dir, fname)
            elif "_B03_10m" in fname:
                B03_path = os.path.join(r10m_dir, fname)
            elif "_B04_10m" in fname:
                B04_path = os.path.join(r10m_dir, fname)
            elif "_B08_10m" in fname:
                B08_path = os.path.join(r10m_dir, fname)
        for fname in os.listdir(r20m_dir):
            if "_B11_20m" in fname:
                B11_path = os.path.join(r20m_dir, fname)
        if not (B02_path and B03_path and B04_path and B08_path and B11_path):
            raise FileNotFoundError("Missing necessary band file(s) in SAFE product.")
        print(f"Found bands for {prod_name}:")
        print("  B02:", B02_path)
        print("  B03:", B03_path)
        print("  B04:", B04_path)
        print("  B08:", B08_path)
        print("  B11:", B11_path)
        with rasterio.open(B02_path) as src:
            B02 = src.read(1)
            transform = src.transform
            src_crs = src.crs
        with rasterio.open(B03_path) as src:
            B03 = src.read(1)
        with rasterio.open(B04_path) as src:
            B04 = src.read(1)
        with rasterio.open(B08_path) as src:
            B08 = src.read(1)
        with rasterio.open(B11_path) as src:
            B11_20m = src.read(1)
        B11 = np.repeat(np.repeat(B11_20m, 2, axis=0), 2, axis=1)
        if B11.shape != B02.shape:
            B11 = B11[:B02.shape[0], :B02.shape[1]]
        detections = detect_trucks(B02, B03, B04, B08, B11)
        timestamp = prod.get("ContentDate", {}).get("Start")
        results = []
        if str(src_crs) != "EPSG:4326":
            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        else:
            transformer = None
        for (row_center, col_center, conf) in detections:
            x, y = transform * (col_center + 0.5, row_center + 0.5)
            if transformer:
                lon_det, lat_det = transformer.transform(x, y)
            else:
                lon_det, lat_det = x, y
            results.append({
                "ProductName": prod_name,
                "Timestamp": timestamp,
                "Latitude": lat_det,
                "Longitude": lon_det,
                "Confidence": round(conf, 3)
            })
        print(f"Detected {len(results)} truck(s) in product {prod_name}.")
        return results
    except Exception as e:
        print(f"❌ Failed processing {prod_name}: {e}")
        return []
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)

def process_chunk(products, max_workers=4):
    """Process a list of products concurrently."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prod = {executor.submit(process_product, prod): prod for prod in products}
        for future in concurrent.futures.as_completed(future_to_prod):
            try:
                res = future.result()
                results.extend(res)
            except Exception as e:
                # Log or handle exceptions for individual products.
                prod = future_to_prod[future]
                print(f"❌ Exception processing product {prod['Name']}: {e}")
    return results

def group_by_tile(products):
    """Group products by their tile information."""
    tile_groups = defaultdict(list)
    for prod in products:
        tile_key = prod.get("tile")
        tile_groups[tile_key].append(prod)
    return tile_groups

def process_tile_group_sequentially(products, delay_between=1):
    """
    Process all products in a tile group sequentially.
    
    Optionally wait for a specified delay (in seconds) between downloads
    to avoid rate limiting (default 1 second).
    """
    results = []
    for prod in products:
        res = process_product(prod)
        results.extend(res)
        # Wait a little before processing the next image to reduce API stress.
        time.sleep(delay_between)
    return results

import concurrent.futures
from collections import defaultdict

def download_product(prod, delay_between=1):
    """
    Download a single product sequentially.
    Returns a dictionary with product metadata and the paths for each required band,
    or None if the download/extraction fails.
    """
    prod_id = prod["Id"]
    prod_name = prod["Name"]
    zip_url = prod["DownloadUrl"]
    zip_path = os.path.join(download_dir, f"{prod_id}.zip")
    extract_dir = os.path.join(download_dir, prod_id)
    try:
        sess = token_manager.get_session()
        if sess is None:
            print(f"Skipping {prod_name} due to failed authentication.")
            return None
        print(f"Downloading product {prod_name} (ID: {prod_id})...")
        r = request_with_token("GET", zip_url, stream=True, timeout=30)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"Extracting product {prod_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        # Determine the actual SAFE folder (if there is a single subdirectory, use it)
        subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
        safe_folder = os.path.join(extract_dir, subdirs[0]) if len(subdirs) == 1 else extract_dir
        granule_dir = os.path.join(safe_folder, "GRANULE")
        if not os.path.isdir(granule_dir):
            raise FileNotFoundError("GRANULE directory not found in SAFE product.")
        gran_subdirs = os.listdir(granule_dir)
        if not gran_subdirs:
            raise FileNotFoundError("No granule subdirectory found in the GRANULE folder.")
        granule_subdir = os.path.join(granule_dir, gran_subdirs[0])
        img_data_dir = os.path.join(granule_subdir, "IMG_DATA")
        if not os.path.isdir(img_data_dir):
            raise FileNotFoundError("IMG_DATA directory not found in granule.")
        r10m_dir = os.path.join(img_data_dir, "R10m")
        r20m_dir = os.path.join(img_data_dir, "R20m")
        if not os.path.isdir(r10m_dir):
            raise FileNotFoundError("R10m folder not found in IMG_DATA.")
        if not os.path.isdir(r20m_dir):
            raise FileNotFoundError("R20m folder not found in IMG_DATA.")
        B02_path = B03_path = B04_path = B08_path = B11_path = None
        for fname in os.listdir(r10m_dir):
            if "_B02_10m" in fname:
                B02_path = os.path.join(r10m_dir, fname)
            elif "_B03_10m" in fname:
                B03_path = os.path.join(r10m_dir, fname)
            elif "_B04_10m" in fname:
                B04_path = os.path.join(r10m_dir, fname)
            elif "_B08_10m" in fname:
                B08_path = os.path.join(r10m_dir, fname)
        for fname in os.listdir(r20m_dir):
            if "_B11_20m" in fname:
                B11_path = os.path.join(r20m_dir, fname)
        if not (B02_path and B03_path and B04_path and B08_path and B11_path):
            raise FileNotFoundError("Missing necessary band file(s) in SAFE product.")
        print(f"Downloaded product {prod_name} successfully.")
        return {
            "ProductName": prod_name,
            "Timestamp": prod.get("ContentDate", {}).get("Start"),
            "B02_path": B02_path,
            "B03_path": B03_path,
            "B04_path": B04_path,
            "B08_path": B08_path,
            "B11_path": B11_path,
            # Return the extraction directory so we can clean it later
            "extract_dir": extract_dir
        }
    except Exception as e:
        print(f"❌ Failed downloading {prod_name}: {e}")
        return None
    finally:
        # Remove zip file after extraction (optional cleanup)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        # DO NOT remove extract_dir here since we need the files for analysis.


def analyze_downloaded_product(info):
    """
    Given the downloaded product info, open each band, run truck detection,
    and return a list of detection results.
    """
    try:
        with rasterio.open(info["B02_path"]) as src:
            B02 = src.read(1)
            transform = src.transform
            src_crs = src.crs
        with rasterio.open(info["B03_path"]) as src:
            B03 = src.read(1)
        with rasterio.open(info["B04_path"]) as src:
            B04 = src.read(1)
        with rasterio.open(info["B08_path"]) as src:
            B08 = src.read(1)
        with rasterio.open(info["B11_path"]) as src:
            B11_20m = src.read(1)
        B11 = np.repeat(np.repeat(B11_20m, 2, axis=0), 2, axis=1)
        if B11.shape != B02.shape:
            B11 = B11[:B02.shape[0], :B02.shape[1]]
        detections = detect_trucks(B02, B03, B04, B08, B11)
        results = []
        if str(src_crs) != "EPSG:4326":
            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        else:
            transformer = None
        for (row_center, col_center, conf) in detections:
            x, y = transform * (col_center + 0.5, row_center + 0.5)
            if transformer:
                lon_det, lat_det = transformer.transform(x, y)
            else:
                lon_det, lat_det = x, y
            results.append({
                "ProductName": info["ProductName"],
                "Timestamp": info["Timestamp"],
                "Latitude": lat_det,
                "Longitude": lon_det,
                "Confidence": round(conf, 3)
            })
        print(f"Analyzed product {info['ProductName']}: found {len(results)} truck(s).")
        return results
    except Exception as e:
        print(f"❌ Failed processing downloaded product {info['ProductName']}: {e}")
        return []

def process_tile_group(products, max_workers=4, download_delay=1):
    """
    For a given tile group (list of products):
      1. Download the products sequentially (with a delay between each).
      2. Then process (analyze) the successfully downloaded products concurrently.
      3. Finally, clean up (delete) the downloaded/extracted files.
    Returns a list of detection results.
    """
    downloaded = []
    for prod in products:
        info = download_product(prod, delay_between=download_delay)
        if info is not None:
            downloaded.append(info)
    results = []
    if downloaded:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(analyze_downloaded_product, info) for info in downloaded]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    results.extend(res)
                except Exception as e:
                    print(f"❌ Exception during analysis: {e}")
    # Cleanup: Delete the extraction directories for every downloaded product.
    for info in downloaded:
        extract_dir = info.get("extract_dir")
        if extract_dir and os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
    return results


# =============================================================================
# 7. NEW PROCESSING PIPELINE: BASELINE & MONITORING IMAGES
# =============================================================================
detection_events = []    # Log of individual detections
site_counts_by_day = {}  # For daily statistics (heatmaps)

# --- Process Baseline Images ---
print(f"Querying baseline images from {baseline_start} to {baseline_end}...")
baseline_images = query_images_for_range(baseline_start, baseline_end)
print(f"Found {len(baseline_images)} baseline image(s).")

# Group baseline images by tile
baseline_groups = group_by_tile(baseline_images)
print("Processing baseline images (tile by tile)...")
for tile, products in baseline_groups.items():
    print(f"Processing {len(products)} baseline images for tile {tile} sequentially for download, then analyzing concurrently...")
    chunk_results = process_tile_group(products, max_workers=4, download_delay=1)
    for rec in chunk_results:
        site, dist = find_nearest_site(rec["Latitude"], rec["Longitude"])
        if site:
            rec["NearestSite"] = site["name"]
            rec["SiteType"] = site["type"]
        else:
            rec["NearestSite"] = ""
            rec["SiteType"] = ""
        detection_events.append(rec)
        day = rec["Timestamp"][:10]
        site_counts_by_day.setdefault(day, {})
        if rec["NearestSite"]:
            site_counts_by_day[day][rec["NearestSite"]] = site_counts_by_day[day].get(rec["NearestSite"], 0) + 1
print("Baseline processing complete.")

# --- Process Monitoring Images ---
print(f"Querying monitoring images from {start_date} to {end_date}...")
monitor_images = query_images_for_range(start_date, end_date)
print(f"Found {len(monitor_images)} monitoring image(s).")

# Group monitoring images by tile
monitor_groups = group_by_tile(monitor_images)
for tile, products in monitor_groups.items():
    print(f"Processing {len(products)} monitoring images for tile {tile} sequentially for download, then analyzing concurrently...")
    chunk_results = process_tile_group(products, max_workers=4, download_delay=1)
    for rec in chunk_results:
        site, dist = find_nearest_site(rec["Latitude"], rec["Longitude"])
        if site:
            rec["NearestSite"] = site["name"]
            rec["SiteType"] = site["type"]
        else:
            rec["NearestSite"] = ""
            rec["SiteType"] = ""
        detection_events.append(rec)
        day = rec["Timestamp"][:10]
        site_counts_by_day.setdefault(day, {})
        if rec["NearestSite"]:
            site_counts_by_day[day][rec["NearestSite"]] = site_counts_by_day[day].get(rec["NearestSite"], 0) + 1
    print(f"Processed monitoring tile {tile} with {len(products)} image(s).")


# =============================================================================
# 8. OUTPUT CSV LOG OF DETECTIONS
# =============================================================================
print(f"Writing detection log to {DETECTIONS_CSV}...")
with open(DETECTIONS_CSV, 'w', newline='') as csvfile:
    fieldnames = ["ProductName", "Timestamp", "Latitude", "Longitude", "Confidence", "NearestSite", "SiteType"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for rec in detection_events:
        writer.writerow(rec)

# =============================================================================
# 9. GENERATE INTERACTIVE MAPS
# =============================================================================
def save_map_with_size_check(folium_map, prefix, part):
    html_str = folium_map.get_root().render()
    html_bytes = html_str.encode('utf-8')
    filename = f"{prefix}_{part}.html"
    with open(filename, 'wb') as f:
        f.write(html_bytes)
    size_mb = len(html_bytes) / (1024 * 1024)
    print(f"Saved {filename} (size ~{size_mb:.2f} MB).")

# (a) Truck Detection Map with markers
print("Generating truck detection maps...")
detection_map_part = 1
detection_map = folium.Map(location=[35.0, 105.0], zoom_start=5)
sites_layer = folium.FeatureGroup(name="Sites", show=True)
for site in sites:
    color = site_type_colors.get(site["type"].lower(), "gray")
    folium.Marker(location=[site["lat"], site["lon"]],
                  popup=site["name"],
                  icon=folium.Icon(color=color)).add_to(sites_layer)
sites_layer.add_to(detection_map)

for day in sorted({rec["Timestamp"][:10] for rec in detection_events}):
    day_layer = folium.FeatureGroup(name=day, show=False)
    marker_cluster = MarkerCluster().add_to(day_layer)
    for rec in [r for r in detection_events if r["Timestamp"][:10] == day]:
        popup_text = (f"<b>Product:</b> {rec['ProductName']}<br>"
                      f"<b>Time:</b> {rec['Timestamp']}<br>"
                      f"<b>Confidence:</b> {rec['Confidence']}<br>"
                      f"<b>Site:</b> {rec['NearestSite']} ({rec['SiteType']})")
        folium.Marker(location=[rec["Latitude"], rec["Longitude"]],
                      popup=popup_text,
                      icon=folium.Icon(color='orange', icon='truck', prefix='fa')).add_to(marker_cluster)
    day_layer.add_to(detection_map)
    folium.LayerControl().add_to(detection_map)
    html_bytes = detection_map.get_root().render().encode('utf-8')
    if len(html_bytes) > MAX_MAP_SIZE:
        detection_map._children.pop(day_layer.get_name(), None)
        save_map_with_size_check(detection_map, DETECTION_MAP_PREFIX, detection_map_part)
        detection_map_part += 1
        detection_map = folium.Map(location=[35.0, 105.0], zoom_start=5)
        sites_layer.add_to(detection_map)
save_map_with_size_check(detection_map, DETECTION_MAP_PREFIX, detection_map_part)

# (b) Heatmap Map: Daily counts vs. baseline
print("Generating truck heatmap maps...")
heatmap_map_part = 1
heatmap_map = folium.Map(location=[35.0, 105.0], zoom_start=5)
for site in sites:
    folium.CircleMarker(location=[site["lat"], site["lon"]],
                        radius=5,
                        color=site_type_colors.get(site["type"].lower(), "gray"),
                        fill=True,
                        fill_opacity=1,
                        popup=site["name"]).add_to(heatmap_map)
for day in sorted(site_counts_by_day.keys()):
    heat_data = []
    for site_name, count in site_counts_by_day[day].items():
        for site in sites:
            if site["name"] == site_name:
                heat_data.append([site["lat"], site["lon"], count])
    if not heat_data:
        continue
    heat_layer = folium.FeatureGroup(name=day, show=False)
    HeatMap(heat_data, radius=25).add_to(heat_layer)
    heat_layer.add_to(heatmap_map)
    folium.LayerControl().add_to(heatmap_map)
    html_bytes = heatmap_map.get_root().render().encode('utf-8')
    if len(html_bytes) > MAX_MAP_SIZE:
        heatmap_map._children.pop(heat_layer.get_name(), None)
        save_map_with_size_check(heatmap_map, HEATMAP_MAP_PREFIX, heatmap_map_part)
        heatmap_map_part += 1
        heatmap_map = folium.Map(location=[35.0, 105.0], zoom_start=5)
        for site in sites:
            folium.CircleMarker(location=[site["lat"], site["lon"]],
                                radius=5,
                                color=site_type_colors.get(site["type"].lower(), "gray"),
                                fill=True,
                                fill_opacity=1,
                                popup=site["name"]).add_to(heatmap_map)
save_map_with_size_check(heatmap_map, HEATMAP_MAP_PREFIX, heatmap_map_part)

print("Processing complete!")
