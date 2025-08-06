# %%
import os
import sys
from pathlib import Path

src = str(Path(__file__).parent).replace("tests", "") + "src"

sys.path.insert(0, src)


import dash_bootstrap_components as dbc
from dash import Dash

from geo_explorer import FileBrowser
from geo_explorer import LocalFileSystem


def not_test_file_browser_locally():

    browser = FileBrowser(
        start_dir="C:/users/ort/OneDrive - Statistisk sentralbyrå/data",
        file_system=LocalFileSystem(),
    )
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.SOLAR],
        serve_locally=True,
        assets_folder="assets",
    )
    app.layout = dbc.Container(browser.get_file_browser_components())

    app.run(debug=True)


if __name__ == "__main__":
    if any("dapla" in key.lower() for key in os.environ):
        not_test_geo_explorer_dapla()
    else:
        not_test_file_browser_locally()
