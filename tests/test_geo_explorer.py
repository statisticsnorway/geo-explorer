# %%

import os
import sys
from pathlib import Path

src = str(Path(__file__).parent).replace("tests", "") + "src"

sys.path.insert(0, src)


import polars as pl
import sgis as sg

from geo_explorer import GeoExplorer
from geo_explorer import LocalFileSystem
from geo_explorer.geo_explorer import DEBUG


def test_debugging_is_off():
    assert not DEBUG


query = """
SELECT
    N5000_fylke_flate_2024.FYLKESNAVN, 
    FYLKE,
    KOMMUNENR as komm_nr, 
    area,
    area / 1000000 as area_km2,
    SUM(df.area) AS area_sum,
FROM
    df
INNER JOIN
    N5000_fylke_flate_2024 USING (FYLKE)
WHERE
    area < 2099301532
ORDER BY
    area  DESC
"""


def _get_explorer():

    return GeoExplorer(
        start_dir="C:/users/ort/OneDrive - Statistisk sentralbyrå/data",
        favorites=[
            "C:/users/ort/OneDrive - Statistisk sentralbyrå/data",
            "C:/users/ort",
        ],
        data={
            "df1": sg.to_gdf((10.8, 59.9), 4326).assign(num_col=10),
            "C:/users/ort/OneDrive - Statistisk sentralbyrå/data/N5000_fylke_flate_2024.parquet": '(pl.col("FYLKE").str.starts_with("5"), pl.col("FYLKE") != "56")',
            "df2": sg.to_gdf([(10.8, 59.9), (10.8001, 59.9001)], 4326)
            .to_crs(3035)
            .pipe(sg.buff, 1000)
            .assign(num_col=[100, 111]),
            "df3": sg.to_gdf((10.8, 59.9), 4326)
            .to_crs(3035)
            .pipe(sg.buff, 1000)
            .pipe(sg.to_lines)
            .assign(num_col=10000),
            "C:/users/ort/OneDrive - Statistisk sentralbyrå/data/N5000_fylke_flate_2023.parquet": 'lambda x: x["FYLKE"].isin(["03", "30"])',
            "C:/users/ort/OneDrive - Statistisk sentralbyrå/data/ABAS_kommune_flate_p2023_v1.parquet": query,
            "df_out_of_bounds": sg.to_gdf((10.8, 61.0), 4326).assign(num_col=-1),
        },
        wms={
            "norge_i_bilder": sg.NorgeIBilderWms(
                years=range(2020, 2024),
                not_contains="Sentinel|CIR",
            ),
        },
        wms_layers_checked={"norge_i_bilder": ["Oslo kommune 2020"]},
        selected_features=["1.151", "5.13", "0.0"],
        column="FYLKE",
        zoom=13,
        center=(59.91740845, 10.71394454),
        file_system=LocalFileSystem(),
        port=None,
        zoomDelta=0.5,
        maxZoom=16,
        minZoom=6,
    )


def not_test_geo_explorer_locally(run=False):
    explorer = _get_explorer()
    print(explorer)
    assert not explorer._deleted_categories
    assert len(explorer._loaded_data) == 7
    assert all(explorer.selected_files.values())
    assert all(isinstance(x, pl.LazyFrame) for x in explorer._loaded_data.values()), [
        type(x) for x in explorer._loaded_data.values()
    ]
    for i, (k, v) in enumerate(explorer._loaded_data.items()):
        assert (
            v.select("_unique_id").collect()["_unique_id"].str.slice(0, 1) == str(i)
        ).all(), (
            i,
            v.select("_unique_id").collect()["_unique_id"].str.slice(0, 1),
        )
    assert (
        features := {
            key: value[col].split()[0] if isinstance(value[col], str) else value[col]
            for col, (key, value) in zip(
                ["FYLKESNAVN", "FYLKESNAVN", "num_col"],
                explorer.selected_features.items(),
                strict=True,
            )
        }
    ) == ({"1.151": "Trøndelag", "5.13": "Troms", "0.0": -1}), features
    assert explorer._queries == (
        {
            "C:/users/ort/OneDrive - Statistisk sentralbyrå/data/N5000_fylke_flate_2024.parquet": '(pl.col("FYLKE").str.starts_with("5"), pl.col("FYLKE") != "56")',
            "C:/users/ort/OneDrive - Statistisk sentralbyrå/data/N5000_fylke_flate_2023.parquet": 'lambda x: x["FYLKE"].isin(["03", "30"])',
            "C:/users/ort/OneDrive - Statistisk sentralbyrå/data/ABAS_kommune_flate_p2023_v1.parquet": query,
        }
    )

    if run:
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
                "df_out_of_bounds": sg.to_gdf((10.8, 61.0), 4326).assign(num_col=-1),
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
        not_test_geo_explorer_locally(run=True)
