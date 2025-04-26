'''
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap
from shapely.geometry import Point
from datetime import datetime
import math

# === Load detection data ===
df = pd.read_csv("detection_events.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Year'] = df['Timestamp'].dt.year

# === Load site data ===
site_files = {
    "Production": "C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/geojson/mre_production_sites.geojson",
    "Deployment": "C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/geojson/pingtan_staging.geojson",
    "Military": "C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/geojson/pla_73rd_group_bases.geojson",
    "Storage": "C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/geojson/storage_depots.geojson"
}
site_type_colors = {"Production": "red", "Deployment": "orange", "Storage": "green", "Military": "black"}
sites = []

for site_type, path in site_files.items():
    gdf = gpd.read_file(path)
    for _, row in gdf.iterrows():
        geom = row.geometry.centroid if row.geometry.type != 'Point' else row.geometry
        sites.append({
            "name": row.get("name", f"{site_type}_Site"),
            "type": site_type,
            "lat": geom.y,
            "lon": geom.x
        })

# === Haversine to find nearest site ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def assign_nearest_sites(df):
    site_names, site_types = [], []
    for _, row in df.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        min_dist = float("inf")
        nearest = None
        for site in sites:
            dist = haversine(lat, lon, site['lat'], site['lon'])
            if dist < min_dist:
                nearest = site
                min_dist = dist
        site_names.append(nearest['name'])
        site_types.append(nearest['type'])
    df['NearestSite'] = site_names
    df['SiteType'] = site_types
    return df

df = assign_nearest_sites(df)
df_2024 = df[df['Year'] == 2024]
df_2025 = df[df['Year'] == 2025]

# === Shared legend HTML ===
def build_legend_html():
    legend = "\n".join([
        f"<i style='background:{site_type_colors[s['type']]};"
        f"width:10px;height:10px;display:inline-block;margin-right:6px;'></i>{s['name']} ({s['type']})<br>"
        for s in sites
    ])
    return f"""
    <div style='position: fixed; bottom: 30px; left: 30px; width: 260px; 
         background-color: white; z-index:9999; font-size:13px; 
         border:2px solid grey; border-radius:6px; padding: 10px;'>
         <b>Site Color Key</b><br>{legend}</div>
    """

# === Shared layer logic ===
def add_map_layers(map_obj, df_subset, include_cluster=True):
    counts = df_subset['NearestSite'].value_counts().to_dict()
    for site in sites:
        count = counts.get(site['name'], 0)
        folium.Circle(
            location=[site["lat"], site["lon"]],
            radius=2500 + count * 8,
            color=site_type_colors[site["type"]],
            fill=True,
            fill_opacity=0.6,
            popup=f"<b>{site['name']}</b><br>{site['type']}<br>Detections: {count}"
        ).add_to(map_obj)

    if include_cluster:
        cluster = MarkerCluster().add_to(map_obj)
        for _, row in df_subset.iterrows():
            popup = (f"<b>Time:</b> {row['Timestamp']}<br>"
                     f"<b>Confidence:</b> {row['Confidence']}<br>"
                     f"<b>Nearest Site:</b> {row['NearestSite']} ({row['SiteType']})")
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=popup,
                icon=folium.Icon(color="orange", icon="truck", prefix="fa")
            ).add_to(cluster)

    map_obj.get_root().html.add_child(folium.Element(build_legend_html()))

# === Generate maps ===
def generate_maps(df_subset, year):
    base = folium.Map(location=[35, 105], zoom_start=5)
    add_map_layers(base, df_subset, include_cluster=True)
    base.save(f"MAP_{year}_TruckDetections.html")
    print(f"✅ Detection map saved for {year}")

    heatmap = folium.Map(location=[35, 105], zoom_start=5)
    heat_data = [[r['Latitude'], r['Longitude'], r['Confidence']] for _, r in df_subset.iterrows()]
    HeatMap(heat_data, radius=20, blur=30, min_opacity=0.25).add_to(heatmap)
    add_map_layers(heatmap, df_subset, include_cluster=True)
    heatmap.save(f"MAP_{year}_TruckDensity.html")
    print(f"✅ Heatmap saved for {year}")

# === Run generation ===
generate_maps(df_2024, 2024)
generate_maps(df_2025, 2025)
'''



import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap
from shapely.geometry import Point, Polygon
import math

# === Load detection data ===
df = pd.read_csv("C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/2019_2022_detection_events.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Year'] = df['Timestamp'].dt.year

# === Mainland China polygon (excludes Taiwan)
mainland_china_polygon = Polygon([
    (73.0, 18.0),
    (73.0, 54.0),
    (123.0, 54.0),
    (123.0, 24.5),
    (119.5, 24.5),
    (119.5, 18.0),
    (73.0, 18.0)
])
def is_in_mainland_china(lat, lon):
    return mainland_china_polygon.contains(Point(lon, lat))

df = df[df.apply(lambda row: is_in_mainland_china(row['Latitude'], row['Longitude']), axis=1)]

# === Load site data ===
site_files = {
    "Production": "C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/geojson/mre_production_sites.geojson",
    "Deployment": "C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/geojson/pingtan_staging.geojson",
    "Military": "C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/geojson/pla_73rd_group_bases.geojson",
    "Storage": "C:/Utah State University/Spring Semester 2025/CAI 5990/satelliteanalysis/geojson/storage_depots.geojson"
}
site_type_colors = {"Production": "red", "Deployment": "orange", "Storage": "green", "Military": "black"}
sites = []

for site_type, path in site_files.items():
    gdf = gpd.read_file(path)
    for _, row in gdf.iterrows():
        geom = row.geometry.centroid if row.geometry.type != 'Point' else row.geometry
        sites.append({
            "name": row.get("name", f"{site_type}_Site"),
            "type": site_type,
            "lat": geom.y,
            "lon": geom.x
        })

# === Haversine + attach nearest site
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def attach_nearest_site(df_subset):
    site_names, site_types = [], []
    for _, row in df_subset.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        nearest_site, min_dist = None, float('inf')
        for site in sites:
            dist = haversine(lat, lon, site['lat'], site['lon'])
            if dist < min_dist:
                nearest_site = site
                min_dist = dist
        site_names.append(nearest_site['name'])
        site_types.append(nearest_site['type'])
    df_subset['NearestSite'] = site_names
    df_subset['SiteType'] = site_types
    return df_subset

df = attach_nearest_site(df)

# === Assign site numbers
for i, site in enumerate(sites):
    site['id'] = i + 1  # human-readable

# === Legend HTML
def build_legend_html(counts):
    legend = "\n".join([
        f"<b>{site['id']}</b>. <i style='background:{site_type_colors[site['type']]};"
        f"width:14px;height:14px;display:inline-block;margin-right:6px;'></i>"
        f"{site['name']} ({site['type']}) — {counts.get(site['name'], 0)} detections<br>"
        for site in sites
    ])
    return f"""
    <div style='position: fixed; bottom: 30px; left: 30px; width: 360px; 
         background-color: white; z-index:9999; font-size:13px; 
         border:2px solid grey; border-radius:6px; padding: 10px;'>
         <b>Site Legend</b><br>{legend}</div>
    """

# === Draw density layer with labeled circles
def add_density_layers(map_obj, df_subset):
    counts = df_subset['NearestSite'].value_counts().to_dict()
    for site in sites:
        count = counts.get(site['name'], 0)
        folium.Circle(
            location=[site["lat"], site["lon"]],
            radius=2500 + count * 8,
            color=site_type_colors[site["type"]],
            fill=True,
            fill_opacity=0.9,
        ).add_to(map_obj)

        folium.map.Marker(
            location=[site["lat"], site["lon"]],
            icon=folium.DivIcon(html=f"""<div style="font-size:14px;font-weight:bold;color:black">{site['id']}</div>""")
        ).add_to(map_obj)

    map_obj.get_root().html.add_child(folium.Element(build_legend_html(counts)))

# === Draw truck detections using MarkerCluster
def add_detection_clusters(map_obj, df_subset):
    counts = df_subset['NearestSite'].value_counts().to_dict()

    for site in sites:
        folium.CircleMarker(
            location=[site["lat"], site["lon"]],
            radius=10,
            color=site_type_colors[site["type"]],
            fill=True,
            fill_opacity=0.9,
            popup=f"<b>{site['name']}</b><br>{site['type']}<br>Detections: {counts.get(site['name'], 0)}"
        ).add_to(map_obj)

        folium.map.Marker(
            location=[site["lat"], site["lon"]],
            icon=folium.DivIcon(html=f"""<div style="font-size:14px;font-weight:bold;color:black">{site['id']}</div>""")
        ).add_to(map_obj)

    cluster = MarkerCluster().add_to(map_obj)
    for _, row in df_subset.iterrows():
        popup = (f"<b>Time:</b> {row['Timestamp']}<br>"
                 f"<b>Confidence:</b> {row['Confidence']}<br>"
                 f"<b>Nearest Site:</b> {row['NearestSite']} ({row['SiteType']})")
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup,
            icon=folium.Icon(color="orange", icon="truck", prefix="fa")
        ).add_to(cluster)

    map_obj.get_root().html.add_child(folium.Element(build_legend_html(counts)))

# === Generate maps
def generate_maps(year_df, year_label):
    print(f"Generating {year_label} maps...")

    # Truck Detection Map
    m_det = folium.Map(location=[35, 105], zoom_start=5)
    add_detection_clusters(m_det, year_df)
    m_det.save(f"MAP_{year_label}_TruckDetections.html")

    # Density Map
    m_density = folium.Map(location=[35, 105], zoom_start=5)
    HeatMap([[r['Latitude'], r['Longitude'], r['Confidence']] for _, r in year_df.iterrows()],
            radius=20, blur=30, min_opacity=0.25).add_to(m_density)
    add_density_layers(m_density, year_df)
    m_density.save(f"MAP_{year_label}_TruckDensity.html")

    print(f"✅ Saved: MAP_{year_label}_TruckDetections.html and MAP_{year_label}_TruckDensity.html")

# === Execute
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as year_executor:
    year_executor.submit(generate_maps, df[df['Year'] == 2019], 2019)
    year_executor.submit(generate_maps, df[df['Year'] == 2022], 2022)
