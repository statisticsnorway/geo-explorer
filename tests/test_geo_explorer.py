from pathlib import Path

src = str(Path(__file__).parent).replace("tests", "") + "src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

import sys

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
            # "/buckets/delt-kart/analyse_data/klargjorte-data/2025/FKB_arealbruk_flate_p2025_v1.parquet",
            # "/buckets/delt-kart/analyse_data/klargjorte-data/2025/FKB_anlegg_flate_p2025_v1.parquet",
        ],
        # column="objtype",
        zoom=15,
        center=(59.91740845, 10.71394444),
        file_system=LocalFileSystem(),
        # port=PORT,
        port=None,
    )
    explorer.run(debug=True)


if __name__ == "__main__":
    not_test_geo_explorer_locally()
