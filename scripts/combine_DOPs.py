import zipfile
from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.crs import CRS
import shutil
import tempfile
from tqdm import tqdm

base_dir = Path(r"../DOP_data")
output_tif = base_dir / "DOP_gesamt_EPSG25832.tif"

# CRS without EPSG lookup (avoids PROJ issues)
target_crs = CRS.from_string(
    "PROJCS[\"ETRS89 / UTM zone 32N\","
    "GEOGCS[\"ETRS89\","
    "DATUM[\"European_Terrestrial_Reference_System_1989\","
    "SPHEROID[\"GRS 1980\",6378137,298.257222101]],"
    "PRIMEM[\"Greenwich\",0],"
    "UNIT[\"degree\",0.0174532925199433]],"
    "PROJECTION[\"Transverse_Mercator\"],"
    "PARAMETER[\"latitude_of_origin\",0],"
    "PARAMETER[\"central_meridian\",9],"
    "PARAMETER[\"scale_factor\",0.9996],"
    "PARAMETER[\"false_easting\",500000],"
    "PARAMETER[\"false_northing\",0],"
    "UNIT[\"metre\",1]]"
)

temp_dir = Path(tempfile.mkdtemp())
assigned_dir = temp_dir / "assigned"
assigned_dir.mkdir(parents=True, exist_ok=True)

zip_dirs = []
for zip_path in base_dir.glob("*.zip"):
    print(f"Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    zip_dirs.append(temp_dir / zip_path.stem)

tif_files = []

for folder in zip_dirs:
    if not folder.exists():
        continue

    for tif in folder.glob("*.tif"):
        tif_files.append(tif)

if not tif_files:
    raise RuntimeError("No DOP GeoTIFFs found")

print(f"Found {len(tif_files)} DOP tiles")

assigned_tifs = []

train_bar = tqdm(tif_files, desc="combining", leave=False)

for tif in train_bar:
    out_name = f"{tif.parent.name}_{tif.name}"
    out_tif = assigned_dir / out_name

    with rasterio.open(tif) as src:
        profile = src.profile.copy()
        profile.update(crs=target_crs)

        data = src.read()

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(data)

    assigned_tifs.append(out_tif)

print("CRS assigned to all tiles")

src_files = [rasterio.open(tif) for tif in assigned_tifs]

mosaic, transform = merge(src_files)

out_profile = src_files[0].profile.copy()
out_profile.update(
    height=mosaic.shape[1],
    width=mosaic.shape[2],
    transform=transform,
    crs=target_crs,
    compress="lzw",
    BIGTIFF="YES"
)

with rasterio.open(output_tif, "w", **out_profile) as dst:
    dst.write(mosaic)

print(f"Final mosaic written to:\n{output_tif}")

for src in src_files:
    src.close()

shutil.rmtree(temp_dir)
print("Temporary files removed")
