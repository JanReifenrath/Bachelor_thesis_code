import os
import math
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import rasterio.features
from shapely.geometry import box
from PIL import Image
from tqdm import tqdm

def apply_buffer_mask(img_chw, buffer_geom, transform):
    """
    img_chw: numpy array (C, H, W)
    buffer_geom: shapely geometry in raster CRS
    transform: affine transform of the cropped raster
    """

    mask_buf = rasterio.features.rasterize(
        [(buffer_geom, 1)],
        out_shape=(img_chw.shape[1], img_chw.shape[2]),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True
    )

    img_chw = img_chw * mask_buf[np.newaxis, :, :]

    return img_chw

OSM_PATH = "../data/roads_for_full_dataset.geojson"
DOP_PATH = "../DOP_data"
OUT_DIR = "../images_for_filled_dataset"

BUFFER_M = 7.5
PIXEL_SIZE_M = 0.20
PATCH_SIZE_PX = 224
PATCH_SIZE_M = PATCH_SIZE_PX * PIXEL_SIZE_M
MIN_ROAD_COVERAGE = 0.25

print("Loading OSM roads...")
roads = gpd.read_file(OSM_PATH)

print("Opening DOP...")
dop = rasterio.open(DOP_PATH)


print("OSM CRS:", roads.crs)
print("DOP CRS:", dop.crs)

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

print("OSM CRS:", roads.crs)
print("DOP CRS:", dop.crs)

roads = roads[roads.geometry.type == "LineString"]
os.makedirs((OUT_DIR), exist_ok=True)
img_id = 0

for _, row in tqdm(roads.iterrows(), total=len(roads)):

    geom = row.geometry
    if geom is None or geom.is_empty:
        continue

    buffer_geom = geom.buffer(BUFFER_M)

    minx, miny, maxx, maxy = buffer_geom.bounds

    x_tiles = math.ceil((maxx - minx) / PATCH_SIZE_M)
    y_tiles = math.ceil((maxy - miny) / PATCH_SIZE_M)

    for ix in range(x_tiles):
        for iy in range(y_tiles):
            tile = box(
                minx + ix * PATCH_SIZE_M,
                miny + iy * PATCH_SIZE_M,
                minx + (ix + 1) * PATCH_SIZE_M,
                miny + (iy + 1) * PATCH_SIZE_M,
            )

            intersection = tile.intersection(buffer_geom)
            if intersection.is_empty:
                continue

            if intersection.area / tile.area < MIN_ROAD_COVERAGE:
                continue

            try:
                img, transform = mask(
                    dop,
                    [tile],
                    crop=True,
                    all_touched=True
                )
                img = apply_buffer_mask(img, buffer_geom, transform)
            except:
                continue

            if img.shape[1] < 20 or img.shape[2] < 20:
                continue

            # Convert to HWC, RGB only
            img = np.transpose(img[:3], (1, 2, 0))

            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            pil = Image.fromarray(img)
            pil = pil.resize((PATCH_SIZE_PX, PATCH_SIZE_PX), Image.BILINEAR)

            osm_id = row.get("@id")
            osm_id = osm_id.replace("/", "-")
            out_path = os.path.join(
                OUT_DIR,
                f"tile_{img_id}_{osm_id}_{minx, miny, maxx, maxy}.png"
            )
            pil.save(out_path)
            img_id += 1

dop.close()

print(f"Finished. Saved {img_id} tiles.")
