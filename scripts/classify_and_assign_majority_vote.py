import os
import re
import torch
import geopandas as gpd
from torchvision import transforms, models
from shapely.geometry import box
from PIL import Image
from tqdm import tqdm
import pandas as pd

TILES_DIR = "../training_images"
ROADS_PATH = "../data/roads_with_lanes_info.geojson"
MODEL_PATH = "best_model_small_model.pth"
OUT_GPKG = "../data/pred_lanes_small_model_majority_vote.gpkg"
CRS_EPSG = 25832

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

roads = gpd.read_file(ROADS_PATH)

from rasterio.crs import CRS

ETRS89_UTM32_WKT = """
PROJCS["ETRS89 / UTM zone 32N",
    GEOGCS["ETRS89",
        DATUM["European_Terrestrial_Reference_System_1989",
            SPHEROID["GRS 1980",6378137,298.257222101]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",9],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",0],
    UNIT["metre",1],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH]]
"""

dop_crs = CRS.from_wkt(ETRS89_UTM32_WKT)
print("OSM CRS:", roads.crs)
print("Reprojecting OSM → ETRS89 / UTM 32N")
roads = roads.to_crs(dop_crs)

OSM_ID_FIELD = "@id"

print(f"Loaded {len(roads)} road geometries")

checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]
num_classes = len(class_names)

model = models.convnext_small(weights=None)
model.classifier[2] = torch.nn.Linear(
    model.classifier[2].in_features,
    num_classes
)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print("Model classes:", class_names)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def parse_filename(fname):
    """
    tile format: tile_0_way-268877845_(minx, miny, maxx, maxy).png
    """
    osm_match = re.search(r"(way-\d+)", fname)
    bbox_match = re.search(r"\(([^)]+)\)", fname)

    if osm_match is None or bbox_match is None:
        return None

    osm_id = osm_match.group(1).replace("-", "/")
    minx, miny, maxx, maxy = map(float, bbox_match.group(1).split(","))

    return osm_id, box(minx, miny, maxx, maxy)

image_paths = []
for root, _, files in os.walk(TILES_DIR):
    for f in files:
        if f.lower().endswith(".png"):
            image_paths.append(os.path.join(root, f))

print(f"Found {len(image_paths)} tiles")

records = []

with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Assigning predictions to roads"):
        fname = os.path.basename(img_path)

        parsed = parse_filename(fname)
        if parsed is None:
            continue

        osm_id, tile_geom = parsed

        road = roads[roads[OSM_ID_FIELD] == osm_id]
        if road.empty:
            continue

        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)

        output = model(img_t)

        probabilities = torch.nn.functional.softmax(output, dim=1)

        conf, pred_idx_tensor = torch.max(probabilities, dim=1)

        confidence = conf.item()
        pred_idx = pred_idx_tensor.item()
        pred_class = class_names[pred_idx]

        real_lanes = int(road["lanes"].item()) if not pd.isna(road["lanes"].item()) else None

        pred_idx+=1
        records.append({
            "osm_id": osm_id,
            "pred_class": pred_class,
            "pred_lanes_num": pred_idx,
            "confidence": confidence,
            "image": fname,
            "real_lanes": real_lanes,
            "highway_type": road["highway"].item(),
            "surface": road["surface"].item(),
            "geometry": road.geometry.iloc[0],
            "parking:both": road["parking:both"].item(),
            "parking:left": road["parking:left"].item(),
            "parking:right": road["parking:right"].item()
        })

gdf_raw = gpd.GeoDataFrame(
    records,
    geometry="geometry",
    crs=f"EPSG:{CRS_EPSG}"
)

mode_per_osm = (
    gdf_raw
    .groupby("osm_id")["pred_lanes_num"]
    .agg(lambda x: x.mode().iloc[0])
)
avg_conf_per_osm = gdf_raw.groupby("osm_id")["confidence"].mean()

gdf_filtered = (gdf_raw[
    gdf_raw["pred_lanes_num"] == gdf_raw["osm_id"].map(mode_per_osm)]
    .groupby("osm_id", as_index=False)
    .first())

gdf_filtered.crs = f"EPSG:{CRS_EPSG}"

gdf_filtered["avg_confidence"] = gdf_filtered["osm_id"].map(avg_conf_per_osm)
gdf_filtered = gdf_filtered.drop(columns=["confidence"])

gdf_filtered.to_file(
    OUT_GPKG,
    driver="GPKG"
)

print(f"Saved → {OUT_GPKG}")
