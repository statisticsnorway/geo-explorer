# %%
import sys
from pathlib import Path
import os

import sgis as sg

src = str(Path(__file__).parent).replace("tests", "") + "src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

sys.path.insert(0, src)


from geo_explorer import GeoExplorer
from geo_explorer import LocalFileSystem


def test_geo_explorer():
    explorer = GeoExplorer(
        start_dir="",
        data=[
            "tests/data/test_path_p2025_v2.parquet",
            "tests/data/test_path_p2025-01_v1.parquet",
        ],
        zoom=15,
        center=(59.91740845, 10.71394444),
        file_system=LocalFileSystem(),
        port=None,
    )
    explorer.run(debug=True)


def not_test_geo_explorer_locally():

    explorer = GeoExplorer(
        start_dir="C:/users/ort/OneDrive - Statistisk sentralbyrå/data",
        data=[
            "C:/users/ort/OneDrive - Statistisk sentralbyrå/data/N5000_fylke_flate_2023.parquet",
            "C:/users/ort/OneDrive - Statistisk sentralbyrå/data/N5000_fylke_flate_2024.parquet",
            {
                "df1": sg.to_gdf((10.8, 59.9), 4326).assign(num_col=100),
                "df2": sg.to_gdf((10.8, 59.9), 4326)
                .to_crs(3035)
                .pipe(sg.buff, 1000)
                .assign(num_col=1000),
                "df3": sg.to_gdf((10.8, 59.9), 4326)
                .to_crs(3035)
                .pipe(sg.buff, 1000)
                .pipe(sg.to_lines)
                .assign(num_col=10000),
            },
        ],
        selected_features=["0_0", "0_2", "1_0"],
        column="FYLKE",
        zoom=13,
        center=(59.91740845, 10.71394444),
        file_system=LocalFileSystem(),
        port=None,
    )
    explorer.run(debug=True)


def not_test_geo_explorer_dapla():

    explorer = GeoExplorer(
        start_dir="/buckets/delt-kart/analyse_data/klargjorte-data/2025",
        data=[
            # "/buckets/delt-kart/analyse_data/klargjorte-data/2025/FKB_arealressurs_flate_p2025_v1.parquet",
        ],
        # column="arealtype",
        zoom=13,
        center=(59.91740845, 10.71394444),
        file_system=LocalFileSystem(),
        port=8055,
    )
    explorer.run(debug=True)

    explorer = GeoExplorer(
        start_dir="/buckets",
        data=[
            "/buckets/delt-kart/analyse_data/klargjorte-data/2025/FKB_arealbruk_flate_p2025_v1.parquet",
            "/buckets/delt-kart/analyse_data/klargjorte-data/2025/FKB_anlegg_flate_p2025_v1.parquet",
        ],
        column="objtype",
        zoom=13,
        center=(59.91740845, 10.71394444),
        file_system=LocalFileSystem(),
        port=8055,
    )
    explorer.run(debug=True)


if __name__ == "__main__":
    if any("dapla" in key.lower() for key in os.environ):
        not_test_geo_explorer_dapla()
    else:
        not_test_geo_explorer_locally()
