# %%
import os
import sys
from pathlib import Path

src = str(Path(__file__).parent).replace("tests", "") + "src"

sys.path.insert(0, src)


import sgis as sg

from dash import Dash
from geo_explorer import GeoExplorer
from geo_explorer import LocalFileSystem
from geo_explorer.geo_explorer import DEBUG


def test_debugging_is_off():
    assert not DEBUG, DEBUG


def test_geo_explorer():
    explorer = GeoExplorer(
        start_dir=src,
        data=[
            "tests/data/test_path_p2025_v2.parquet",
            "tests/data/test_path_p2025-01_v1.parquet",
            {
                "df1": sg.to_gdf((10.8, 59.9), 4326).assign(num_col=100),
                "df2": sg.to_gdf([(10.8, 59.9), (10.8001, 59.9001)], 4326)
                .to_crs(3035)
                .pipe(sg.buff, 1000)
                .assign(num_col=1000),
            },
            {
                "df3": sg.to_gdf((10.8, 59.9), 4326)
                .to_crs(3035)
                .pipe(sg.buff, 1000)
                .pipe(sg.to_lines)
                .assign(num_col=10000),
            },
        ],
        column="num_col",
        zoom=15,
        center=(59.91740845, 10.71394444),
        file_system=LocalFileSystem(),
        port=3000,
    )
    assert isinstance(explorer.app, Dash)


def not_test_geo_explorer_locally():
    explorer = GeoExplorer(
        start_dir="C:/users/ort/OneDrive - Statistisk sentralbyrå/data",
        favorites=[
            "C:/users/ort/OneDrive - Statistisk sentralbyrå/data",
            "C:/users/ort",
        ],
        data=[
            "C:/users/ort/OneDrive - Statistisk sentralbyrå/data/N5000_fylke_flate_2023.parquet",
            "C:/users/ort/OneDrive - Statistisk sentralbyrå/data/N5000_fylke_flate_2024.parquet",
            {
                "df1": sg.to_gdf((10.8, 59.9), 4326).assign(num_col=100),
                "df2": sg.to_gdf([(10.8, 59.9), (10.8001, 59.9001)], 4326)
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
        wms={
            "norge_i_bilder": sg.NorgeIBilderWms(
                years=range(2020, 2024),
                not_contains="Sentinel|CIR",
            ),
        },
        selected_features=[1, 1.05],
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
        favorites=[
            "/buckets/delt-kart/analyse_data/klargjorte-data/2025",
            "/buckets/produkt",
        ],
        data=[
            "/buckets/delt-kart/analyse_data/klargjorte-data/2025/FKB_arealbruk_flate_p2025_v1.parquet",
            "/buckets/delt-kart/analyse_data/klargjorte-data/2025/FKB_anlegg_flate_p2025_v1.parquet",
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
        selected_features=[1, 1.05],
        column="arealtype",
        zoom=13,
        center=(59.91740845, 10.71394444),
        file_system=LocalFileSystem(),
        port=3000,
    )
    explorer.run(debug=True)


if __name__ == "__main__":

    if any("dapla" in key.lower() for key in os.environ):
        not_test_geo_explorer_dapla()
    else:
        not_test_geo_explorer_locally()
