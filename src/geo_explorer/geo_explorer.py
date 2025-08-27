import datetime
import inspect
import itertools
import json
import logging
import math
import os
import signal
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from numbers import Number
from pathlib import Path
from pathlib import PurePath
from time import perf_counter
from typing import Any
from typing import ClassVar

import matplotlib.colors as mcolors
import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import joblib
import matplotlib
import numpy as np
import pandas as pd
import polars as pl
import pyarrow
import pyarrow.parquet as pq
import sgis as sg
from sgis.maps.wms import WmsLoader
import shapely
from dash import Dash
from dash import Input
from dash import Output
from dash import State
from dash import callback
from dash import dash_table
from dash import dcc
from dash import html
from dash.development.base_component import Component
from dash_extensions.javascript import Namespace
from fsspec.spec import AbstractFileSystem
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from jenkspy import jenks_breaks
from pandas.api.types import is_datetime64_any_dtype
from sgis.io.dapla_functions import _get_geo_metadata
from shapely.errors import GEOSException
from shapely.geometry import Point
from shapely.geometry import Polygon

from .file_browser import FileBrowser
from .fs import LocalFileSystem
from .utils import _clicked_button_style
from .utils import _standardize_path
from .utils import _unclicked_button_style
from .utils import get_button_with_tooltip

OFFWHITE: str = "#ebebeb"
FILE_CHECKED_COLOR: str = "#3e82ff"
DEFAULT_ZOOM: int = 12
DEFAULT_CENTER: tuple[float, float] = (59.91740845, 10.71394444)
CURRENT_YEAR: int = datetime.datetime.now().year

DEBUG: bool = False

if DEBUG:

    def debug_print(*args):
        print(*args)

else:

    def debug_print(*args):
        pass


class GeoExplorer:
    """Class for exploring geodata interactively.

    It's best to run the app in the terminal, but it works in jupyter as well.

    The main arguments in the class initializer are 'start_dir', 'favorites', 'file_system' and 'port'.

    The rest of the arguments can be fetched with the "Export as code" button in the app.
    Copy the code into a new python program and run it to get an app that opens with
    the same bounds, data, filtering and coloring.

    Args:
        start_dir: Directory to list files at init.
        favorites: List of directories to quick-jump to.
        port: default to 8050. Set to None if running locally.
        file_system: File system used to list and read files.
            Must act like fsspec's AbstractFileSystem and
            implement the methods *ls* and *glob*. The methods
            should take the argument 'detail', which, if set to True,
            will return a dict for each listed path with the keys
            "updated" (timestamp), "size" (bytes), "name" (full path)
            and "type" ("directory" or "file").
        data: Optional data to load at start. Either a list of file paths,
            a dict of file paths as keys and filter strings as values
            or a dict of GeoDataFrames.
        column: Column to color the data.
        color_dict: Optionally set the coloring when when 'column' is
            a non-numeric column. Dict keys must be valid column values
            and dict values can be hex color code or a named color
            (https://matplotlib.org/stable/gallery/color/named_colors.html).
        center: y and x coordinates to use as map center as init.
        zoom: zoom level to use at init.
        wms: dict of wms loaders, e.g. a sg.NorgeIBilderWms instance.
            Can also be added after app is started.
        wms_layers_checked: dict where keys correspond to the keys in the "wms"
            dict and the values are a list of wms layer names to be shown at init.
        max_rows: Max number of rows to sample per dataset if number of feature in bounds excedes.
            Note that rendering more than the default (10,000) might crash the server, especially for
            polygon features.
        selected_features: list of indices of features (rows) to show in attribute table
            at init. Fetch this list with the "Export as code" button.
        hard_click: If True, clicking on a geometry triggers all overlapping geometries to be marked.
        splitted: If True, all rows will have a separate label and color.
        alpha: Opacity/transparency of the geometries.
        nan_color: Color for missing values. Defaults to a shade of gray.
        nan_label: Defaults to "Missing".
        **kwargs: Additional keyword arguments passed to dash_leaflet.Map: https://www.dash-leaflet.com/components/map_container.

    A "clean" GeoExplorer can be initialized like this:

    >>> from geo_explorer import GeoExplorer
    >>> from geo_explorer import LocalFileSystem
    >>> GeoExplorer(
    ...     start_dir="dir1",
    ...     favorites=["dir1", "dir2"],
    ...     file_system=LocalFileSystem(),
    ...     port=3000
    ... ).run()

    Starting a custom GeoExplorer with data loaded, filtered and colored:

    >>> from geo_explorer import GeoExplorer
    >>> from gcsfs import GCSFileSystem
    >>> import sgis as sg
    >>> DELT_KART = "ssb-areal-data-delt-kart-prod"
    >>> YEAR = 2025
    >>> entur_path = f"{DELT_KART}/analyse_data/klargjorte-data/{YEAR}/ENTUR_Holdeplasser_punkt_p{YEAR}_v1.parquet"
    >>> GeoExplorer(
    ...     start_dir=f"{DELT_KART}/analyse_data/klargjorte-data/{YEAR}",
    ...     favorites=[
    ...         f"{DELT_KART}/analyse_data/klargjorte-data/{YEAR}",
    ...         f"{DELT_KART}/visualisering_data/klargjorte-data/{YEAR}/parquet",
    ...     ],
    ...     data={
    ...         "jernbanetorget": sg.to_gdf([10.7535581, 59.9110967], crs=4326), # GeoDataFrame
    ...         entur_path: "kjoeretoey != 'fly'", # file path and filter function
    ...     },
    ...     column="kjoeretoey",
    ...     color_dict={
    ...         "jernbane": "darkgreen",
    ...         "buss": "red",
    ...         "trikk": "deepskyblue",
    ...         "tbane": "yellow",
    ...         "baat": "navy",
    ...     },
    ...     center=(59.91740845, 10.71394444),
    ...     zoom=13,
    ...     file_system=GCSFileSystem(),
    ...     port=3000,
    ... ).run()
    """

    _wms_constructors: ClassVar[dict[str, Callable]] = {
        "Norge i bilder": sg.NorgeIBilderWms,
    }
    _base_layers: ClassVar[list[dl.BaseLayer]] = [
        dl.BaseLayer(
            dl.TileLayer("OpenStreetMap"),
            name="OpenStreetMap",
            checked=True,
        ),
        dl.BaseLayer(
            dl.TileLayer(
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                attribution='&copy; <a href="https://carto.com/">CARTO</a>',
            ),
            name="CartoDB Dark Matter",
            checked=False,
        ),
        dl.BaseLayer(
            dl.TileLayer(
                url="https://opencache.statkart.no/gatekeeper/gk/gk.open_nib_web_mercator_wmts_v2?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=Nibcache_web_mercator_v2&STYLE=default&FORMAT=image/jpgpng&tileMatrixSet=default028mm&tileMatrix={z}&tileRow={y}&tileCol={x}",
                attribution="© Geovekst",
            ),
            name="Norge i bilder",
            checked=False,
        ),
    ]
    _map_children: ClassVar[list[Component]] = [
        dl.ScaleControl(position="bottomleft"),
        dl.MeasureControl(
            position="bottomright",
            primaryLengthUnit="meters",
        ),
    ]

    def __init__(
        self,
        start_dir: str,
        *,
        favorites: list[str] | None = None,
        port: int = 8050,
        file_system: AbstractFileSystem | None = None,
        data: dict[str, str | GeoDataFrame] | list[str | dict] | None = None,
        column: str | None = None,
        color_dict: dict | None = None,
        center: tuple[float, float] | None = None,
        zoom: int | None = None,
        wms: dict[str, WmsLoader] | None = None,
        wms_layers_checked: dict[str, list[str]] | None = None,
        max_rows: int = 10_000,
        selected_features: list[str] | None = None,
        hard_click: bool = False,
        splitted: bool = False,
        alpha: float = 0.6,
        nan_color: str = "#969696",
        nan_label: str = "Missing",
        **kwargs,
    ) -> None:
        """Initialiser."""
        self.start_dir = start_dir
        self.port = port
        self._kwargs = kwargs  # store kwargs for the "export" button
        self.maxZoom = kwargs.get("maxZoom", 40)
        self.minZoom = kwargs.get("minZoom", 4)
        self.bounds = None
        self.column = column
        self.color_dict = color_dict or {}
        self.color_dict = {
            key: (color if color.startswith("#") else _named_color_to_hex(color))
            for key, color in self.color_dict.items()
        }

        self.wms = dict(wms or {})
        for wms_name, constructor in self._wms_constructors.items():
            if constructor in {type(x) for x in self.wms.values()}:
                continue
            self.wms[wms_name] = constructor()
        wms_layers_checked = wms_layers_checked or {}
        self.wms_layers_checked = {
            wms_name: wms_layers_checked.get(wms_name, []) for wms_name in self.wms
        }
        self.file_system = _get_file_system(self.start_dir, file_system)
        self.nan_color = nan_color
        self.nan_label = nan_label
        self.splitted = splitted
        self.hard_click = hard_click
        self.max_rows = max_rows
        self.alpha = alpha
        self.bounds_series = GeoSeries()
        self.selected_files: dict[str, int] = {}
        self.loaded_data: dict[str, pl.DataFrame] = {}
        self._max_unique_id_int: int = -1
        self._loaded_data_sizes: dict[str, int] = {}
        self.concatted_data: pl.DataFrame | None = None
        self._deleted_categories = set()
        self.selected_features = {}
        self._file_browser = FileBrowser(
            start_dir, file_system=file_system, favorites=favorites
        )

        if is_jupyter():
            service_prefix = os.environ["JUPYTERHUB_SERVICE_PREFIX"].strip("/")
            requests_pathname_prefix = (
                f"/{service_prefix}/proxy/{port}/" if port else None
            )
        else:
            requests_pathname_prefix = f"/proxy/{self.port}/" if self.port else None

        self.app = Dash(
            __name__,
            suppress_callback_exceptions=DEBUG is False,
            external_stylesheets=[dbc.themes.SOLAR],
            requests_pathname_prefix=requests_pathname_prefix,
            serve_locally=True,
            assets_folder="assets",
        )

        if is_jupyter():
            self.app.logger.setLevel(logging.ERROR)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))

        def get_layout():
            debug_print("\n\n\n\n\nget_layout", self.bounds, self.zoom, self.column)
            return dbc.Container(
                [
                    dcc.Location(id="url"),
                    dbc.Row(html.Div(id="alert")),
                    dbc.Row(html.Div(id="alert2")),
                    dbc.Row(html.Div(id="alert3")),
                    dbc.Row(html.Div(id="alert4")),
                    dbc.Row(html.Div(id="new-file-added")),
                    html.Div(id="file-deleted"),
                    dbc.Row(
                        [
                            dbc.Col(html.Div(id="loading", style={"height": "3vh"})),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                self._map_constructor(html.Div(id="lc"), **kwargs),
                                width=9,
                            ),
                            dbc.Col(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                get_button_with_tooltip(
                                                    "Split rows",
                                                    id="splitter",
                                                    n_clicks=1 if self.splitted else 0,
                                                    tooltip_text="Split all geometries into separate colors",
                                                ),
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        *get_button_with_tooltip(
                                                            "Export as code",
                                                            id="export",
                                                            style={"color": "#285cd4"},
                                                            tooltip_text="Get code to reproduce current view",
                                                        ),
                                                        dcc.Dropdown(
                                                            value=self.alpha,
                                                            options=[
                                                                {
                                                                    "label": f"opacity={round(x, 1)}",
                                                                    "value": round(
                                                                        x, 1
                                                                    ),
                                                                }
                                                                for x in np.arange(
                                                                    0.1, 1.1, 0.1
                                                                )
                                                            ],
                                                            id="alpha",
                                                            clearable=False,
                                                        ),
                                                        dbc.Modal(
                                                            [
                                                                dbc.ModalHeader(
                                                                    dbc.ModalTitle(
                                                                        "Copy the code below to reproduce current view"
                                                                    ),
                                                                    close_button=False,
                                                                ),
                                                                dbc.ModalBody(
                                                                    id="export-text"
                                                                ),
                                                            ],
                                                            id="export-view",
                                                            is_open=False,
                                                        ),
                                                    ]
                                                )
                                            ),
                                            dbc.Col(id="max_rows"),
                                        ],
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        dcc.Dropdown(
                                                            id="column-dropdown",
                                                            value=self.column,
                                                            placeholder="Select column to color by",
                                                            style={
                                                                "font-size": 22,
                                                                "overflow": "visible",
                                                            },
                                                            maxHeight=600,
                                                            clearable=True,
                                                        ),
                                                    ],
                                                ),
                                                width=9,
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    id="force-categorical",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        style={
                                            "margin-top": "7px",
                                        },
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="k",
                                                    options=[
                                                        {"label": f"k={i}", "value": i}
                                                        for i in [3, 4, 5, 6, 7, 8, 9]
                                                    ],
                                                    value=5,
                                                    style={
                                                        "font-size": 22,
                                                        "overflow": "visible",
                                                    },
                                                    maxHeight=300,
                                                    clearable=False,
                                                ),
                                                width=4,
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    dcc.Dropdown(
                                                        id="cmap-placeholder",
                                                        options=[
                                                            {
                                                                "label": f"cmap={name}",
                                                                "value": name,
                                                            }
                                                            for name in [
                                                                "viridis",
                                                                "plasma",
                                                                "inferno",
                                                                "magma",
                                                                "Greens",
                                                            ]
                                                            + [
                                                                name
                                                                for name, cmap in matplotlib.colormaps.items()
                                                                if "linear"
                                                                in str(cmap).lower()
                                                            ]
                                                        ],
                                                        value="viridis",
                                                        maxHeight=200,
                                                        clearable=False,
                                                    ),
                                                ),
                                                width=5,
                                            ),
                                        ],
                                        id="numeric-options",
                                        style={"display": "none"},
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Button(
                                                            "Hide/show wms options",
                                                            id="wms-hide-button",
                                                            n_clicks=(
                                                                1
                                                                if (
                                                                    not wms
                                                                    and not any(
                                                                        self.wms_layers_checked.values()
                                                                    )
                                                                )
                                                                else 0
                                                            ),
                                                        ),
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            dcc.Checklist(
                                                                options=list(self.wms),
                                                                value=list(
                                                                    set(wms or [])
                                                                    | {
                                                                        name
                                                                        for name, checked_layers in self.wms_layers_checked.items()
                                                                        if checked_layers
                                                                    }
                                                                ),
                                                                id="wms-checklist",
                                                            ),
                                                        ),
                                                        html.Div(id="wms-panel"),
                                                    ],
                                                    style={
                                                        "border": "1px solid #ccc",
                                                        "border-radius": "5px",
                                                        "padding": "20px",
                                                        "background-color": OFFWHITE,
                                                    },
                                                ),
                                                id="wms-hide-div",
                                                style={"display": "none"},
                                            ),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            html.Div(html.B("Layers")),
                                            dbc.Col(id="file-control-panel"),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "border": "1px solid #ccc",
                                            "border-radius": "3px",
                                            "padding": "0px",
                                            "background-color": OFFWHITE,
                                            "margin-bottom": "7px",
                                            "margin-top": "7px",
                                            "margin-left": "0px",
                                            "margin-right": "0px",
                                        },
                                    ),
                                    dbc.Col(
                                        get_button_with_tooltip(
                                            "Reload categories",
                                            id="reload-categories",
                                            n_clicks=0,
                                            tooltip_text="Get back categories that have been X-ed out",
                                        )
                                    ),
                                    dbc.Row(
                                        id="colorpicker-container",
                                    ),
                                ],
                                style={
                                    "height": "90vh",
                                    "width": "2vh",
                                    "overflow": "scroll",
                                },
                                className="scroll-container",
                            ),
                        ],
                    ),
                    *get_button_with_tooltip(
                        "Hard click",
                        id="hard-click",
                        n_clicks=int(self.hard_click),
                        tooltip_text="'Hard' click means that clicking on a geometry triggers all overlapping geometries to be marked",
                    ),
                    get_data_table(
                        title_id="clicked-features-title",
                        table_id="feature-table-rows-clicked",
                        div_id="feature-table-container-clicked",
                        clear_id="clear-table-clicked",
                    ),
                    get_data_table(
                        title_id="all-features-title",
                        table_id="feature-table-rows",
                        div_id="feature-table-container",
                        clear_id="clear-table",
                    ),
                    *self._file_browser.get_file_browser_components(),
                    dcc.Store(id="is_splitted", data=False),
                    dcc.Store(id="column-dropdown2", data=None),
                    dcc.Input(
                        id="debounced_bounds",
                        value=None,
                        style={"display": "none"},
                        debounce=0.25,
                    ),
                    dcc.Store(id="viewport-container", data=None),
                    html.Div(id="color-container", style={"display": "none"}),
                    html.Div(id="missing", style={"display": "none"}),
                    dcc.Store(id="colors-are-updated"),
                    dcc.Store(id="dummy3"),
                    dcc.Store(id="wms-added"),
                    html.Div(id="data-was-concatted", style={"display": "none"}),
                    html.Div(id="data-was-changed", style={"display": "none"}),
                    dcc.Store(id="order-was-changed", data=None),
                    html.Div(id="new-data-read", style={"display": "none"}),
                    html.Div(id="max_rows_was_changed", style={"display": "none"}),
                    dbc.Input(id="max_rows_value", style={"display": "none"}),
                    dcc.Store(id="dummy-output", data=None),
                    dcc.Store(id="dummy-output2", data=None),
                    html.Div(id="bins", style={"display": "none"}),
                    html.Div(False, id="is-numeric", style={"display": "none"}),
                    dcc.Store(id="map-bounds", data=None),
                    dcc.Store(id="map-zoom", data=None),
                    dcc.Store(id="map-center", data=None),
                    dcc.Store(id="clicked-features", data=[]),
                    dcc.Store(id="all-features", data=[]),
                    dcc.Store(id="clicked-ids", data=None),
                    dcc.Interval(
                        id="interval-component",
                        interval=2000,
                        n_intervals=0,
                        disabled=True,
                    ),
                ],
                fluid=True,
            )

        error_mess = "'data' must be a list of file paths or a dict of GeoDataFrames."
        bounds_series_dict = {}
        if isinstance(data, dict):
            data = [data]

        self._filters = {}
        for x in data or []:
            if isinstance(x, (str | os.PathLike | PurePath)):
                self.selected_files[_standardize_path(x)] = True
                continue
            elif not isinstance(x, dict):
                raise ValueError(error_mess)
            for key, value in x.items():
                key = _standardize_path(key)
                if value is not None and not isinstance(value, (GeoDataFrame | str)):
                    raise ValueError(error_mess)
                elif not isinstance(value, GeoDataFrame):
                    self.selected_files[key] = True
                    self._filters[key] = value
                    continue
                value = _fix_df(value)
                bounds_series_dict[key] = shapely.box(*value.total_bounds)
                self.loaded_data[key] = pl.from_pandas(
                    value.assign(geometry=value.geometry.to_wkb())
                )
                self.selected_files[key] = True

        self.bounds_series = GeoSeries(bounds_series_dict)

        # storing bounds here before file paths are loaded. To avoid setting center as the entire map bounds if large data
        if len(self.bounds_series):
            minx, miny, maxx, maxy = self.bounds_series.total_bounds
        else:
            minx, miny, maxx, maxy = None, None, None, None

        if not self.selected_files:
            self.center = DEFAULT_CENTER
            self.zoom = DEFAULT_ZOOM
            self.app.layout = get_layout
            self._register_callbacks()
            return

        child_paths = _get_child_paths(
            [x for x in self.selected_files if x not in self.loaded_data],
            self.file_system,
        )
        if child_paths:
            self.bounds_series = pd.concat(
                [
                    self.bounds_series,
                    sg.get_bounds_series(
                        child_paths, file_system=self.file_system
                    ).to_crs(4326),
                ]
            )

        temp_center = center if center is not None else DEFAULT_CENTER
        _read_files(
            self,
            [x for x in self.selected_files if x not in self.loaded_data],
            mask=Point(temp_center),
        )

        # dataframe dicts as input data are currently sorted first because they were added to loaded_data first.
        # now to get back original order
        loaded_data_sorted = {}
        self._max_unique_id_int: int = -1
        for x in data or []:
            if isinstance(x, dict):
                for key in x:
                    key = _standardize_path(key)
                    df = self.loaded_data[key]
                    self._max_unique_id_int += 1
                    loaded_data_sorted[key] = df.with_columns(
                        _unique_id=_get_unique_id(
                            df, self._max_unique_id_int
                        )  # len(loaded_data_sorted))
                    )
            else:
                x = _standardize_path(x)
                df = self.loaded_data[x]
                self._max_unique_id_int += 1
                loaded_data_sorted[x] = df.with_columns(
                    _unique_id=_get_unique_id(
                        df, self._max_unique_id_int
                    )  # len(loaded_data_sorted))
                )

        self.loaded_data = loaded_data_sorted

        if center is not None:
            self.center = center
        elif self.loaded_data and all((minx, miny, maxx, maxy)):
            self.center = ((maxy + miny) / 2, (maxx + minx) / 2)
        else:
            self.center = DEFAULT_CENTER

        if zoom is not None:
            self.zoom = zoom
        elif self.loaded_data and all((minx, miny, maxx, maxy)):
            self.zoom = get_zoom_from_bounds(minx, miny, maxx, maxy, 800, 600)
        else:
            self.zoom = DEFAULT_ZOOM

        self.app.layout = get_layout

        for idx in selected_features if selected_features is not None else []:
            i = int(idx)
            df = list(self.loaded_data.values())[i]
            row = df.filter(pl.col("_unique_id") == idx)
            columns = [col for col in row.columns if col != "geometry"]
            features = GeoDataFrame(
                row.drop("geometry"),
                geometry=shapely.from_wkb(row["geometry"]),
                crs=4326,
            ).__geo_interface__["features"]
            feature = next(iter(features))
            self.selected_features[idx] = {
                col: value
                for col, value in zip(
                    columns, feature["properties"].values(), strict=True
                )
            }

        self._register_callbacks()

    def run(
        self, debug: bool = False, jupyter_mode: str = "external", **kwargs
    ) -> None:
        """Run the app."""
        if is_jupyter():
            kwargs["jupyter_server_url"] = str(
                Path(os.environ["JUPYTERHUB_HTTP_REFERER"])
                / os.environ["JUPYTERHUB_SERVICE_PREFIX"].strip("/")
            )
            display_url = f"{kwargs['jupyter_server_url']}/proxy/{self.port}/"
            # make sure there's two slashes to make link clickable in terminal
            # (env variable might only have one slash, which redirects to two-slash-url)
            display_url = display_url.replace("https:/", "https://").replace(
                "https:///", "https://"
            )
            self.logger.info(f"\nDash is running on {display_url}\n\n")

        try:
            self.app.run(
                debug=debug,
                port=self.port,
                jupyter_mode=jupyter_mode,
                threaded=False,
                **kwargs,
            )
        except KeyboardInterrupt:
            os.kill(os.getpid(), signal.SIGTERM)

    def _register_callbacks(self) -> None:

        @callback(
            Output("export-text", "children"),
            Output("export-view", "is_open"),
            Input("export", "n_clicks"),
            Input("file-deleted", "children"),
            prevent_initial_call=True,
        )
        def export(
            export_clicks,
            file_deleted,
        ):
            triggered = dash.callback_context.triggered_id
            if triggered in ["file-deleted", "close-export"] or not export_clicks:
                return None, False
            data = self._get_self_as_dict()
            defaults = inspect.getfullargspec(self.__class__).kwonlydefaults
            data = {
                key: value for key, value in data.items() if value != defaults.get(key)
            } | self._kwargs
            txt = self._get_self_as_string(data)
            return html.Div(f"{txt}.run()"), True

        @callback(
            Output("debounced_bounds", "value"),
            Input("map", "bounds"),
            Input("map", "zoom"),
            State("map-bounds", "data"),
            prevent_initial_call=True,
        )
        def update_bounds(bounds, zoom, bounds2):
            if bounds is None:
                return dash.no_update
            self.bounds = bounds
            centroid = shapely.box(*self._nested_bounds_to_bounds(bounds)).centroid
            self.center = (centroid.y, centroid.x)
            self.zoom = zoom
            return json.dumps(bounds)

        @callback(
            Output("map", "bounds"),
            Output("map", "zoom"),
            Output("map", "center"),
            Input("map-bounds", "data"),
            Input("map-zoom", "data"),
            State("map-center", "data"),
            prevent_initial_call=True,
        )
        def intermediate_update_bounds(bounds, zoom, center):
            """Update map bounds after short sleep because otherwise it's buggy."""
            time.sleep(0.1)
            if not zoom and not bounds and not center:
                return dash.no_update, dash.no_update, dash.no_update
            return bounds, zoom, center

        @callback(
            Output("new-file-added", "children"),
            Input({"type": "load-parquet", "index": dash.ALL}, "n_clicks"),
            Input({"type": "load-parquet", "index": dash.ALL}, "id"),
            State({"type": "file-path", "index": dash.ALL}, "id"),
        )
        def append_path(load_parquet, load_parquet_ids, ids):
            triggered = dash.callback_context.triggered_id
            debug_print("append_path", triggered)
            if not any(load_parquet) or not triggered:
                return dash.no_update
            try:
                selected_path = triggered["index"]
            except Exception as e:
                raise type(e)(f"{e}: {triggered}") from e
            n_clicks = get_index(load_parquet, load_parquet_ids, selected_path)
            if selected_path in self.selected_files or not n_clicks:
                return dash.no_update
            try:
                more_bounds = _get_bounds_series(
                    selected_path, file_system=self.file_system
                )
            except Exception as e:
                return dbc.Alert(
                    f"Couldn't read {selected_path}. {type(e)}: {e}",
                    color="warning",
                    dismissable=True,
                )
            self.selected_files[selected_path] = True
            self.bounds_series = pd.concat(
                [
                    self.bounds_series,
                    more_bounds,
                ]
            )
            return None

        @callback(
            Output("new-data-read", "children"),
            Output("missing", "children"),
            Output("interval-component", "disabled"),
            Input("debounced_bounds", "value"),
            Input("new-file-added", "children"),
            Input("interval-component", "n_intervals"),
            Input("missing", "children"),
            Input({"type": "checked-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "checked-btn", "index": dash.ALL}, "id"),
        )
        def get_files_in_bounds(
            bounds,
            file_added,
            n_intervals,
            missing,
            checked_clicks,
            checked_ids,
        ):
            t = perf_counter()

            triggered = dash.callback_context.triggered_id
            debug_print(
                "get_files_in_bounds",
                triggered,
                f"{len(missing or [])=}, {len(self.loaded_data)=}",
            )
            if isinstance(triggered, dict) and triggered["type"] == "checked-btn":
                path = get_index_if_clicks(checked_clicks, checked_ids)
                if path is None:
                    return dash.no_update, dash.no_update, dash.no_update

            if triggered != "missing":
                box = shapely.box(*self._nested_bounds_to_bounds(bounds))
                files_in_bounds = sg.sfilter(self.bounds_series, box)

                def is_checked(path) -> bool:
                    return next(
                        iter(
                            is_checked
                            for sel_path, is_checked in self.selected_files.items()
                            if sel_path in path
                        )
                    )

                missing = list(
                    {
                        path
                        for path in files_in_bounds.index
                        if path not in self.loaded_data and is_checked(path)
                    }
                )
                if not all(path in self._loaded_data_sizes for path in missing):
                    missing_size = [
                        path for path in missing if path not in self._loaded_data_sizes
                    ]
                    with ThreadPoolExecutor() as executor:
                        more_sizes = {
                            path: x["size"]
                            for path, x in zip(
                                missing_size,
                                executor.map(self.file_system.info, missing_size),
                                strict=True,
                            )
                        }
                    self._loaded_data_sizes |= more_sizes

            if triggered != "interval-component":
                new_missing = []
                for path in missing:
                    size = self._loaded_data_sizes[path]
                    if size < 1e9:
                        new_missing.append(path)
                        continue
                    with self.file_system.open(path, "rb") as file:
                        nrow = pq.read_metadata(file).num_rows
                    n = 30
                    rows_to_read = nrow // n
                    for i in range(n):
                        new_path = path + f"-_-{rows_to_read}-{i}"
                        if new_path in missing:
                            continue
                        new_missing.append(new_path)
                        self._loaded_data_sizes[new_path] = size / n
                        self.bounds_series = pd.concat(
                            [
                                self.bounds_series,
                                GeoSeries({new_path: self.bounds_series.loc[path]}),
                            ]
                        )
                missing = new_missing

            if missing:
                if len(missing) > 10:
                    to_read = 0
                    cumsum = 0
                    for i, path in enumerate(missing):
                        size = self._loaded_data_sizes[path]
                        cumsum += size
                        to_read += 1
                        if cumsum > 500_000_000 or to_read > cpu_count() * 2:
                            break
                else:
                    to_read = min(10, len(missing))
                debug_print(f"{to_read=}, {len(missing)=}")
                if len(missing) > to_read:
                    _read_files(self, missing[:to_read])
                    missing = missing[to_read:]
                    disabled = False if len(missing) else True
                    new_data_read = dash.no_update if len(missing) else False
                else:
                    _read_files(self, missing)
                    missing = []
                    disabled = True
                    new_data_read = True
            else:
                new_data_read = True
                disabled = True

            debug_print(
                "get_files_in_bounds ferdig etter",
                perf_counter() - t,
                "-",
                f"{len(missing)=}, {len(self.loaded_data)=}, {new_data_read=}, {disabled=}",
            )

            return new_data_read, missing, disabled

        @callback(
            Output("is_splitted", "data"),
            Output("column-dropdown", "value"),
            Input("splitter", "n_clicks"),
            Input("file-deleted", "children"),
        )
        def set_column_to_split_index(splitter_clicks, file_deleted):
            if not self.selected_files:
                return False, None
            triggered = dash.callback_context.triggered_id
            if triggered == "file-deleted":
                # TODO: why is this needed?
                return dash.no_update, dash.no_update
            if triggered is not None:
                self.splitted = not self.splitted
            if self.splitted:
                return self.splitted, "split_index"
            return self.splitted, self.column

        @callback(
            Output("data-was-concatted", "children"),
            Output("data-was-changed", "children"),
            Output("alert2", "children"),
            Input("new-data-read", "children"),
            Input("file-deleted", "children"),
            Input("is_splitted", "data"),
            Input({"type": "filter", "index": dash.ALL}, "value"),
            Input({"type": "filter", "index": dash.ALL}, "id"),
            State("debounced_bounds", "value"),
            prevent_initial_call=True,
        )
        def concat_data(
            new_data_read,
            file_deleted,
            is_splitted,
            filter_functions: list[str],
            filter_ids: list[str],
            bounds,
        ):
            triggered = dash.callback_context.triggered_id
            debug_print("concat_data", triggered, new_data_read, self.splitted)

            t = perf_counter()
            if not new_data_read:
                return dash.no_update, 1, dash.no_update

            bounds = self._nested_bounds_to_bounds(bounds)

            for path in self.selected_files:
                try:
                    filter_function = get_index(filter_functions, filter_ids, path)
                    self._filters[path] = filter_function
                except ValueError:
                    pass

            df, alerts = self._concat_data(bounds)
            self.concatted_data = df
            debug_print(
                "concat_data finished after",
                perf_counter() - t,
                len(self.concatted_data) if self.concatted_data is not None else None,
            )

            return 1, 1, alerts

        @callback(
            Output("file-control-panel", "children"),
            Output("order-was-changed", "data"),
            Input("data-was-changed", "children"),
            # Input("map", "bounds"),
            Input("file-deleted", "children"),
            Input({"type": "order-button-up", "index": dash.ALL}, "n_clicks"),
            Input({"type": "order-button-down", "index": dash.ALL}, "n_clicks"),
            State({"type": "order-button-up", "index": dash.ALL}, "id"),
            State({"type": "order-button-down", "index": dash.ALL}, "id"),
            State("file-control-panel", "children"),
            prevent_initial_call=True,
        )
        def render_items(
            _,
            file_deleted,
            n_clicks_up,
            n_clicks_down,
            ids_up,
            ids_down,
            buttons,
        ):
            triggered = dash.callback_context.triggered_id
            if isinstance(triggered, dict) and triggered["type"] == "order-button-up":
                return _change_order(self, n_clicks_up, ids_up, buttons, "up")
            if isinstance(triggered, dict) and triggered["type"] == "order-button-down":
                return _change_order(self, n_clicks_down, ids_down, buttons, "down")

            return [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    get_button_with_tooltip(
                                        "🡑",
                                        id={
                                            "type": "order-button-up",
                                            "index": path,
                                        },
                                        n_clicks=0,
                                        style={"width": "1vh"},
                                        tooltip_text="Move forwards",
                                    ),
                                ),
                                dbc.Row(
                                    get_button_with_tooltip(
                                        "🡓",
                                        id={
                                            "type": "order-button-down",
                                            "index": path,
                                        },
                                        n_clicks=0,
                                        style={"width": "1vh"},
                                        tooltip_text="Move backwards",
                                    ),
                                ),
                            ],
                            style={
                                "marginRight": "10px",
                            },
                        ),
                        dbc.Col(
                            get_button_with_tooltip(
                                "x",
                                id={
                                    "type": "checked-btn",
                                    "index": path,
                                },
                                style=(
                                    {
                                        "color": FILE_CHECKED_COLOR,
                                        "background-color": FILE_CHECKED_COLOR,
                                    }
                                    if checked
                                    else {
                                        "color": OFFWHITE,
                                        "background-color": OFFWHITE,
                                    }
                                ),
                                tooltip_text="Show/hide data",
                            )
                        ),
                        dbc.Col(
                            get_button_with_tooltip(
                                "Show table",
                                id={
                                    "type": "table-btn",
                                    "index": path,
                                },
                                n_clicks=0,
                                style={
                                    "color": "#285cd4",
                                    "border": "none",
                                    "background": "none",
                                    "cursor": "pointer",
                                    "marginLeft": "auto",
                                },
                                tooltip_text="Show all rows",
                            )
                        ),
                        dbc.Col(
                            get_button_with_tooltip(
                                "❌",
                                id={
                                    "type": "delete-btn",
                                    "index": path,
                                },
                                n_clicks=0,
                                style={
                                    "color": "red",
                                    "border": "none",
                                    "background": "none",
                                    "cursor": "pointer",
                                    "marginLeft": "auto",
                                },
                                tooltip_text="Remove data",
                            )
                        ),
                        dbc.Col(html.Span(path)),
                        dbc.Row(
                            [
                                dcc.Input(
                                    self._filters.get(path, None),
                                    placeholder="Filter (with polars or pandas). E.g. komm_nr == '0301'",
                                    id={
                                        "type": "filter",
                                        "index": path,
                                    },
                                    debounce=3,
                                ),
                                dbc.Tooltip(
                                    "E.g. komm_nr == '0301' or pl.col('komm_nr') == '0301'",
                                    target={
                                        "type": "filter",
                                        "index": path,
                                    },
                                    delay={"show": 500, "hide": 100},
                                ),
                            ],
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "marginBottom": "5px",
                    },
                )
                for path, checked in (reversed(self.selected_files.items()))
            ], dash.no_update

        @callback(
            Output({"type": "checked-btn", "index": dash.MATCH}, "style"),
            Output({"type": "checked-btn", "index": dash.MATCH}, "n_clicks"),
            Input({"type": "checked-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "checked-btn", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def check_or_uncheck(n_clicks_list, ids):
            path = get_index_if_clicks(n_clicks_list, ids)
            if not path:
                return dash.no_update, dash.no_update
            is_checked = self.selected_files[path]
            if not is_checked:
                self.selected_files[path] = True
                return {
                    "color": FILE_CHECKED_COLOR,
                    "background-color": FILE_CHECKED_COLOR,
                }, 0
            else:
                self.selected_files[path] = False
                return {
                    "color": OFFWHITE,
                    "background-color": OFFWHITE,
                }, 1

        @callback(
            Output(
                {"type": "checked-btn-wms", "wms_name": dash.MATCH, "tile": dash.MATCH},
                "style",
            ),
            Input(
                {"type": "checked-btn-wms", "wms_name": dash.ALL, "tile": dash.ALL},
                "n_clicks",
            ),
            State(
                {"type": "checked-btn-wms", "wms_name": dash.ALL, "tile": dash.ALL},
                "id",
            ),
            prevent_initial_call=True,
        )
        def check_or_uncheck(n_clicks_list, ids):
            triggered = dash.callback_context.triggered_id
            if triggered is None:
                return dash.no_update, dash.no_update
            wms_name = triggered["wms_name"]
            tile = triggered["tile"]
            is_checked: bool = tile in self.wms_layers_checked[wms_name]
            if not is_checked:
                self.wms_layers_checked[wms_name].append(tile)
                return {
                    "color": FILE_CHECKED_COLOR,
                    "background-color": FILE_CHECKED_COLOR,
                }
            else:
                i = self.wms_layers_checked[wms_name].index(tile)
                self.wms_layers_checked[wms_name].pop(i)
                return {
                    "color": OFFWHITE,
                    "background-color": OFFWHITE,
                }

        @callback(
            Output("file-deleted", "children", allow_duplicate=True),
            Output("alert3", "children", allow_duplicate=True),
            Input({"type": "delete-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "delete-btn", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def delete_file(n_clicks_list, delete_ids):
            return self._delete_file(n_clicks_list, delete_ids)

        @callback(
            Output("file-deleted", "children", allow_duplicate=True),
            Output("alert3", "children", allow_duplicate=True),
            Output("color-container", "children", allow_duplicate=True),
            Input({"type": "delete-cat-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "delete-cat-btn", "index": dash.ALL}, "id"),
            State("color-container", "children"),
            prevent_initial_call=True,
        )
        def delete_category(n_clicks_list, delete_ids, color_container):
            path_to_delete = get_index_if_clicks(n_clicks_list, delete_ids)
            if path_to_delete is None:
                return dash.no_update, dash.no_update, dash.no_update
            debug_print(f"path to delete: {path_to_delete}")
            if not self.column:
                return (*self._delete_file(n_clicks_list, delete_ids), dash.no_update)
            else:
                self._deleted_categories.add(path_to_delete)
                return None, None, dash.no_update

        @callback(
            Output("file-deleted", "children", allow_duplicate=True),
            Input("reload-categories", "n_clicks"),
            prevent_initial_call=True,
        )
        def reload_categories(n_clicks):
            if not n_clicks:
                return dash.no_update
            self._deleted_categories = set()
            return None

        @callback(
            Output("splitter", "style"),
            Input("is_splitted", "data"),
            Input("column-dropdown", "value"),
            prevent_initial_call=True,
        )
        def update_splitter_style(_, column):
            if column is None:
                self.column = None
                self.splitted = None
            if self.splitted and column == "split_index":
                return _clicked_button_style()
            else:
                return _unclicked_button_style()

        @callback(
            Output("hard-click", "style"),
            Input("hard-click", "n_clicks"),
        )
        def update_hard_click_button_style(n_clicks):
            if (n_clicks or 0) % 2 == 1:
                self.hard_click = True
                return _clicked_button_style()
            else:
                self.hard_click = False
                return _unclicked_button_style()

        @callback(
            Output("loading", "children", allow_duplicate=True),
            Input({"type": "load-parquet", "index": dash.ALL}, "n_clicks"),
            Input("map", "bounds"),
            Input("map", "zoom"),
            prevent_initial_call=True,
        )
        def add_loading_text(load_parquet, bounds, zoom):
            return "Loading data..."

        @callback(
            Output("loading", "children", allow_duplicate=True),
            Input("missing", "children"),
            prevent_initial_call=True,
        )
        def update_loading_text(missing):
            if missing:
                return dash.no_update
            else:
                return "Processing data..."

        @callback(
            Output("column-dropdown", "options"),
            Input("colors-are-updated", "data"),
            prevent_initial_call=True,
        )
        def update_column_dropdown_options(_):
            if self.concatted_data is None or not len(self.concatted_data):
                return dash.no_update
            return self._get_column_dropdown_options()

        @callback(
            Output("numeric-options", "style"),
            Input("is-numeric", "children"),
        )
        def hide_or_show_numeric_options(is_numeric):
            if is_numeric:
                return {"margin-bottom": "7px"}
            else:
                return {"display": "none"}

        @callback(
            Output("force-categorical", "n_clicks"),
            Input("column-dropdown", "value"),
            prevent_initial_call=True,
        )
        def reset_force_categorical(_):
            return 0

        @callback(
            Output("dummy3", "data"),
            Input({"type": "colorpicker", "column_value": dash.ALL}, "value"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def set_colorpicker_value(colors, ids):
            triggered = dash.callback_context.triggered_id
            if triggered is None:
                return dash.no_update
            column_value = triggered["column_value"]
            i = next(
                iter(i for i, x in enumerate(ids) if x["column_value"] == column_value)
            )
            color = colors[i]
            self.color_dict[column_value] = color

        @callback(
            Output("colorpicker-container", "children"),
            Output("bins", "children"),
            Output("is-numeric", "children"),
            Output("force-categorical", "children"),
            Output("colors-are-updated", "data"),
            Input("cmap-placeholder", "value"),
            Input("k", "value"),
            Input("force-categorical", "n_clicks"),
            Input("data-was-concatted", "children"),
            State("column-dropdown", "value"),
            State("debounced_bounds", "value"),
            State("bins", "children"),
        )
        def get_column_value_color_dict(
            cmap: str,
            k: int,
            force_categorical_clicks: int,
            data_was_concatted,
            column,
            bounds,
            bins,
        ):
            triggered = dash.callback_context.triggered_id
            debug_print("\nget_column_value_color_dict, column=", column, triggered)

            if not self.selected_files:
                self.column = None
                self.color_dict = {}
                return html.Div(), None, False, None, 1
            elif column and column != self.column:
                self.color_dict = {}
                self.column = column
            elif not column and triggered is None:
                column = self.column
            elif self.concatted_data is None:
                return [], None, dash.no_update, dash.no_update, dash.no_update
            else:
                self.column = column

            default_colors = list(sg.maps.map._CATEGORICAL_CMAP.values())

            debug_print(self.column, column)
            debug_print(self.color_dict)

            if not column or (
                self.concatted_data is not None and column not in self.concatted_data
            ):
                new_values = [_get_name(value) for value in self.selected_files]
                new_colors = (
                    default_colors
                    + [
                        _random_color()
                        for _ in range(len(new_values) - len(default_colors))
                    ]
                )[: len(new_values)]

                try:
                    color_dict = dict(zip(new_values, new_colors, strict=True))
                except ValueError as e:
                    raise ValueError(f"{e}: {new_values} - {new_colors}") from e

                color_dict |= self.color_dict
                self.color_dict = color_dict

                color_dict = {
                    key: color
                    for key, color in color_dict.items()
                    if any(str(key) == Path(x).stem for x in self.selected_files)
                }

                return (
                    _get_colorpicker_container(color_dict),
                    None,
                    False,
                    None,
                    1,
                )

            bounds = self._nested_bounds_to_bounds(bounds)

            values = filter_by_bounds(
                self.concatted_data[[column, "minx", "miny", "maxx", "maxy"]],
                bounds,
            )[column]
            values_no_nans = values.drop_nans().drop_nulls()
            values_no_nans_unique = set(values_no_nans.unique())

            force_categorical_button = _get_force_categorical_button(
                values_no_nans, force_categorical_clicks
            )
            is_numeric: bool = (
                force_categorical_clicks or 0
            ) % 2 == 0 and values_no_nans.dtype.is_numeric()

            if is_numeric and len(values_no_nans):
                if len(values_no_nans_unique) <= k:
                    bins = list(values_no_nans_unique)
                else:
                    bins = jenks_breaks(values_no_nans.to_numpy(), n_classes=k)

                cmap_ = matplotlib.colormaps.get_cmap(cmap)
                colors_ = [
                    matplotlib.colors.to_hex(cmap_(int(i)))
                    for i in np.linspace(0, 255, num=k)
                ]
                rounded_bins = [round(x, 1) for x in bins]
                color_dict = {
                    f"{round(min(values_no_nans), 1)} - {rounded_bins[0]}": colors_[0],
                    **{
                        f"{start} - {stop}": colors_[i + 1]
                        for i, (start, stop) in enumerate(
                            itertools.pairwise(rounded_bins[1:-1])
                        )
                    },
                    f"{rounded_bins[-1]} - {round(max(values_no_nans), 1)}": colors_[
                        -1
                    ],
                }
            else:
                new_values = [
                    value
                    for value in values_no_nans_unique
                    if value not in self.color_dict
                ]
                existing_values = [
                    value for value in values_no_nans_unique if value in self.color_dict
                ]
                default_colors = [
                    x for x in default_colors if x not in set(self.color_dict.values())
                ]
                colors = default_colors[
                    len(existing_values) : min(
                        len(values_no_nans_unique), len(default_colors)
                    )
                ]
                colors = colors + [
                    _random_color() for _ in range(len(new_values) - len(colors))
                ]
                color_dict = dict(zip(new_values, colors, strict=True))
                bins = None

                color_dict |= self.color_dict

            if color_dict.get(self.nan_label, self.nan_color) != self.nan_color:
                self.nan_color = color_dict[self.nan_label]

            elif self.nan_label not in color_dict and polars_isna(values).any():
                color_dict[self.nan_label] = self.nan_color

            debug_print("\n\ncolor_dict nederst")
            debug_print(color_dict)
            if not is_numeric:
                self.color_dict = color_dict
                any_isnull = values.is_null().any()
                color_dict = {
                    key: color
                    for key, color in color_dict.items()
                    if key in values_no_nans_unique
                    or (key == self.nan_label and any_isnull)
                }
            debug_print(color_dict)

            return (
                _get_colorpicker_container(color_dict),
                bins,
                is_numeric,
                force_categorical_button,
                1,
            )

        @callback(
            Output("max_rows_was_changed", "children"),
            Input("max_rows_value", "value"),
        )
        def update_max_rows(value):
            if value is not None:
                self.max_rows = value
                return 1
            return dash.no_update

        @callback(
            Output("loading", "children", allow_duplicate=True),
            Input("alert", "children"),
            prevent_initial_call=True,
        )
        def update_loading(_):
            if self.concatted_data is None or not len(self.concatted_data):
                return None
            return "Finished loading"

        @callback(
            Output("lc", "children"),
            Output("alert", "children"),
            Output("max_rows", "children"),
            Output({"type": "wms-list", "index": dash.ALL}, "children"),
            Input("colors-are-updated", "data"),
            Input({"type": "colorpicker", "column_value": dash.ALL}, "value"),
            Input("is-numeric", "children"),
            Input("wms-checklist", "value"),
            Input("wms-added", "data"),
            Input("max_rows_was_changed", "children"),
            Input("data-was-changed", "children"),
            Input("order-was-changed", "data"),
            Input("alpha", "value"),
            Input({"type": "checked-btn", "index": dash.ALL}, "n_clicks"),
            Input(
                {"type": "checked-btn-wms", "wms_name": dash.ALL, "tile": dash.ALL},
                "style",
            ),
            State("debounced_bounds", "value"),
            State("column-dropdown", "value"),
            State("bins", "children"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "id"),
            State("max_rows", "children"),
        )
        def add_data(
            currently_in_bounds,
            colorpicker_values_list,
            is_numeric,
            # wms_items,
            wms_checklist,
            wms_added,
            max_rows_was_changed,
            data_was_changed,
            order_was_changed,
            alpha,
            checked_clicks,
            checked_wms_clicks,
            bounds,
            column,
            bins,
            colorpicker_ids,
            max_rows_component,
        ):
            debug_print(
                "\n\nadd_data",
                dash.callback_context.triggered_id,
                len(self.loaded_data),
                f"{column=}" f"{self.column=}",
            )
            t = perf_counter()

            bounds = self._nested_bounds_to_bounds(bounds)

            column_values = [x["column_value"] for x in colorpicker_ids]
            color_dict = dict(zip(column_values, colorpicker_values_list, strict=True))

            wms_layers, all_tiles_lists = self._add_wms(wms_checklist, bounds)

            if is_numeric:
                color_dict = {i: color for i, color in enumerate(color_dict.values())}

            add_data_func = partial(
                _add_data_one_path,
                max_rows=self.max_rows,
                concatted_data=self.concatted_data,
                nan_color=self.nan_color,
                bounds=bounds,
                column=column,
                is_numeric=is_numeric,
                color_dict=color_dict,
                bins=bins,
                alpha=alpha,
            )

            # not parallelizing here because polars does and doesn't like double parallelization
            results = [
                add_data_func(path)
                for path, checked in self.selected_files.items()
                if checked
            ]

            data = list(itertools.chain.from_iterable([x[0] for x in results if x[0]]))
            out_alert = [x[1] for x in results if x[1]]
            rows_are_not_hidden = not any(x[2] for x in results)

            debug_print(
                "add_data ferdig etter",
                perf_counter() - t,
                "loaded_data:",
                len(self.loaded_data),
                "concatted_data:",
                len(self.concatted_data if self.concatted_data is not None else []),
            )
            if rows_are_not_hidden:
                max_rows_component = None
            else:
                max_rows_component = _get_max_rows_displayed_component(self.max_rows)

            return (
                dl.LayersControl(self._base_layers + wms_layers + data),
                (out_alert if out_alert else None),
                max_rows_component,
                all_tiles_lists,
            )

        @callback(
            Input("alpha", "value"),
            prevent_initial_call=True,
        )
        def update_alpha(alpha):
            self.alpha = alpha

        @callback(
            Output("clicked-features-title", "children"),
            Input("clicked-features", "data"),
        )
        def update_clicked_features_title(features):
            if not features:
                return dash.no_update
            return (f"Clicked features (n={len(features)})",)

        @callback(
            Output("all-features-title", "children"),
            Input("all-features", "data"),
        )
        def update_all_features_title(features):
            if not features:
                return dash.no_update
            return (
                f"All features (n={len(features)})"
                " (note that for partitioned files, only partitions in bounds are loaded)",
            )

        @callback(
            Output("clicked-features", "data"),
            Output("clicked-ids", "data"),
            Output("alert4", "children"),
            Input("clear-table-clicked", "n_clicks"),
            Input({"type": "geojson", "filename": dash.ALL}, "n_clicks"),
            State({"type": "geojson", "filename": dash.ALL}, "clickData"),
            State({"type": "geojson", "filename": dash.ALL}, "id"),
            State("clicked-features", "data"),
            State("clicked-ids", "data"),
        )
        def display_clicked_feature_attributes(
            clear_table,
            geojson_n_clicks,
            features,
            feature_ids,
            clicked_features,
            clicked_ids,
        ):
            triggered = dash.callback_context.triggered_id
            debug_print("display_clicked_feature_attributes", triggered)
            if triggered == "clear-table-clicked":
                self.selected_features = {}
                return [], [], None
            if triggered is None or (
                self.selected_features and not features or not any(features)
            ):
                clicked_ids = list(self.selected_features)
                clicked_features = list(self.selected_features.values())
                return clicked_features, clicked_ids, None

            if not features or not any(features):
                return dash.no_update, dash.no_update, None

            filename_id = triggered["filename"]
            index = next(
                iter(
                    i
                    for i, id_ in enumerate(feature_ids)
                    if id_["filename"] == filename_id
                )
            )

            _used_file_paths = set()
            for path in self.loaded_data:
                selected_path = next(iter(x for x in self.selected_files if x in path))
                if selected_path in filename_id:
                    _used_file_paths.add(path)
                    break

            feature = features[index]
            assert feature, (feature, index, filename_id)
            unique_id = next(
                iter(
                    value
                    for key, value in feature["properties"].items()
                    if key == "_unique_id"
                )
            )
            if self.hard_click:
                geom = shapely.geometry.shape(feature["geometry"])

                def geoms_relate(wkb: bytes) -> bool:
                    this_geom = shapely.from_wkb(wkb)
                    try:
                        intersection = this_geom.intersection(geom)
                        return (intersection.area / this_geom.area) > 0.01
                    except (ZeroDivisionError, GEOSException):
                        return (
                            this_geom.overlaps(geom)
                            or this_geom.within(geom)
                            or geom.covers(this_geom)
                        )

                intersecting = (
                    filter_by_bounds(self.concatted_data, geom.bounds)
                    .filter(pl.col("_unique_id") != unique_id)
                    .filter(
                        pl.col("geometry").map_elements(
                            geoms_relate, return_dtype=pl.Boolean
                        )
                    )
                )
                _used_file_paths |= set(intersecting["__file_path"].unique())

            columns = set()
            for path in _used_file_paths:
                columns |= set(self.loaded_data[path].columns).difference({"geometry"})

            props_list = [
                {
                    key: value
                    for key, value in feature["properties"].items()
                    if key in columns
                }
            ]

            if self.hard_click:
                props_list += intersecting.select(*columns).to_dicts()

            for props in props_list:
                if props["_unique_id"] not in clicked_ids:
                    clicked_features.append(props)
            clicked_ids = [x["_unique_id"] for x in clicked_features]
            self.selected_features = dict(
                zip(clicked_ids, clicked_features, strict=True)
            )
            return clicked_features, clicked_ids, None

        @callback(
            Output("feature-table-container-clicked", "style"),
            Input("clicked-features", "data"),
        )
        def hide_show_table_clicked(features):
            if not features or not len(features):
                return {"display": "none"}
            return None

        @callback(
            Output("feature-table-container", "style"),
            Input("all-features", "data"),
        )
        def hide_show_table_all(features):
            if not features or not len(features):
                return {"display": "none"}
            return None

        @callback(
            Output("all-features", "data"),
            Input("clear-table", "n_clicks"),
            Input({"type": "table-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "table-btn", "index": dash.ALL}, "id"),
        )
        def display_all_feature_attributes(
            clear_table, table_btn_n_clicks, table_btn_ids
        ):
            triggered = dash.callback_context.triggered_id
            debug_print("display_all_feature_attributes", triggered)
            if triggered == "clear-table":
                return []
            if triggered is None:
                return dash.no_update

            clicked_path = get_index_if_clicks(table_btn_n_clicks, table_btn_ids)
            if clicked_path is None:
                return dash.no_update

            df, _ = self._concat_data(bounds=None, paths=[clicked_path])
            if df is None or not len(df):
                # read data out of bounds to get table
                _read_files(
                    self,
                    [
                        x
                        for x in self.bounds_series[
                            lambda x: x.index.str.contains(clicked_path)
                        ].index
                        if x not in self.loaded_data
                    ],
                )
                df, _ = self._concat_data(bounds=None, paths=[clicked_path])
                if df is None or not len(df):
                    return None
            df = df.drop("geometry")
            if not len(df.columns):
                return None
            clicked_features = df.to_dicts()
            return clicked_features

        @callback(
            Output("feature-table-rows", "columns"),
            Output("feature-table-rows", "data"),
            Output("feature-table-rows", "style_table"),
            Output("feature-table-rows", "hidden_columns"),
            Input("all-features", "data"),
            State("feature-table-rows", "style_table"),
        )
        def update_table(data, style_table):
            return self._update_table(
                data, column_dropdown=_UseColumns(), style_table=style_table
            )

        @callback(
            Output("feature-table-rows-clicked", "columns"),
            Output("feature-table-rows-clicked", "data"),
            Output("feature-table-rows-clicked", "style_table"),
            Output("feature-table-rows-clicked", "hidden_columns"),
            Input("clicked-features", "data"),
            State("column-dropdown", "options"),
            State("feature-table-rows-clicked", "style_table"),
        )
        def update_table_clicked(data, column_dropdown, style_table):
            return self._update_table(data, column_dropdown, style_table)

        @callback(
            Output("map-bounds", "data"),
            Output("map-zoom", "data"),
            Output("map-center", "data"),
            Input("feature-table-rows", "active_cell"),
            Input("feature-table-rows-clicked", "active_cell"),
            State("viewport-container", "data"),
            prevent_initial_call=True,
        )
        def zoom_to_feature(active: dict, active_clicked: dict, viewport):
            triggered = dash.callback_context.triggered_id
            if triggered == "feature-table-rows":
                if active is None:
                    return dash.no_update, dash.no_update, dash.no_update
                unique_id = active["row_id"]
            else:
                if active is None:
                    return dash.no_update, dash.no_update, dash.no_update
                unique_id = active_clicked["row_id"]

            i = int(unique_id)
            file = list(self.loaded_data)[i]
            df, _ = self._concat_data(bounds=None, paths=[file])

            matches = (
                df.lazy()
                .filter(pl.col("_unique_id") == unique_id)
                .select("minx", "miny", "maxx", "maxy")
                .collect()
            )
            if not len(matches):
                return dash.no_update, dash.no_update, dash.no_update
            minx, miny, maxx, maxy = matches.row(0)
            center = ((miny + maxy) / 2, (minx + maxx) / 2)
            bounds = [[miny, minx], [maxy, maxx]]

            width = int(viewport["width"] * 0.7)
            height = int(viewport["height"] * 0.7)
            zoom_level = get_zoom_from_bounds(minx, miny, maxx, maxy, width, height)
            zoom_level = min(zoom_level, self.maxZoom)
            zoom_level = max(zoom_level, self.minZoom)
            return bounds, int(zoom_level), center

        @callback(
            Output("wms-hide-div", "style"),
            Input("wms-hide-button", "n_clicks"),
        )
        def hide_wms(n_clicks):
            if not n_clicks or n_clicks % 2 == 0:
                return None
            return {"display": "none"}

        @callback(
            Output("wms-panel", "children"),
            Input("wms-checklist", "value"),
            State("wms-panel", "children"),
        )
        def add_wms_panel(
            wms_checklist,
            items,
        ):
            items = []
            for wms_name in self.wms:
                if wms_name in wms_checklist:
                    self._construct_wms_obj(wms_name)
                    try:
                        from_year = int(self.wms[wms_name].years[0])
                    except IndexError:
                        from_year = dash.no_update
                    try:
                        to_year = int(self.wms[wms_name].years[-1])
                    except IndexError:
                        to_year = dash.no_update

                    def as_none_if_falsy(x):
                        if not x or (hasattr(x, "__iter__") and not any(x)):
                            return None
                        else:
                            return str(x)

                    not_contains = as_none_if_falsy(self.wms[wms_name].not_contains)
                    contains = as_none_if_falsy(self.wms[wms_name].contains)
                    style = None
                else:
                    from_year = None
                    to_year = None
                    not_contains = None
                    contains = None
                    style = {"display": "none"}
                    # TODO close wms also on reload

                items.append(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Row(html.Div(html.B(wms_name))),
                                    dbc.Row(
                                        [
                                            dcc.Input(
                                                value=from_year,
                                                id={
                                                    "type": "from-year",
                                                    "index": wms_name,
                                                },
                                                type="number",
                                                placeholder="From year",
                                                debounce=0.25,
                                            ),
                                            dbc.Tooltip(
                                                "From year",
                                                target={
                                                    "type": "from-year",
                                                    "index": wms_name,
                                                },
                                            ),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dcc.Input(
                                                value=to_year,
                                                id={
                                                    "type": "to-year",
                                                    "index": wms_name,
                                                },
                                                type="number",
                                                placeholder="To year",
                                                debounce=0.25,
                                            ),
                                            dbc.Tooltip(
                                                "To year",
                                                target={
                                                    "type": "to-year",
                                                    "index": wms_name,
                                                },
                                            ),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dcc.Input(
                                                value=not_contains,
                                                id={
                                                    "type": "wms-not-contains",
                                                    "index": wms_name,
                                                },
                                                type="text",
                                                placeholder="Substrings to be excluded (use | for OR)",
                                                debounce=0.25,
                                            ),
                                            dbc.Tooltip(
                                                "Substrings to be excluded (use | for OR)",
                                                target={
                                                    "type": "wms-not-contains",
                                                    "index": wms_name,
                                                },
                                            ),
                                        ],
                                    ),
                                    dbc.Row(
                                        [
                                            dcc.Input(
                                                value=contains,
                                                id={
                                                    "type": "wms-contains",
                                                    "index": wms_name,
                                                },
                                                type="text",
                                                placeholder="Substrings to be included (use | for OR)",
                                                debounce=0.25,
                                            ),
                                            dbc.Tooltip(
                                                "Substrings to be included (use | for OR)",
                                                target={
                                                    "type": "wms-contains",
                                                    "index": wms_name,
                                                },
                                            ),
                                        ],
                                    ),
                                    dbc.Row(
                                        id={
                                            "type": "wms-list",
                                            "index": wms_name,
                                        },
                                    ),
                                ],
                                style={
                                    "border": "1px solid #ccc",
                                    "border-radius": "3px",
                                },
                            ),
                        ],
                        style=style,
                    )
                )

            return items

        @callback(
            Output("wms-added", "data", allow_duplicate=True),
            Output({"type": "from-year", "index": dash.ALL}, "value"),
            Input({"type": "from-year", "index": dash.ALL}, "value"),
            State({"type": "to-year", "index": dash.ALL}, "value"),
            State({"type": "from-year", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def update_wms_from_year(from_year_values, to_year_values, ids):
            triggered = dash.callback_context.triggered_id
            return self._update_wms_year(
                triggered,
                what="from",
                to_year_values=to_year_values,
                from_year_values=from_year_values,
                ids=ids,
            )

        @callback(
            Output("wms-added", "data", allow_duplicate=True),
            Output({"type": "to-year", "index": dash.ALL}, "value"),
            Input({"type": "to-year", "index": dash.ALL}, "value"),
            State({"type": "from-year", "index": dash.ALL}, "value"),
            State({"type": "to-year", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def update_wms_to_year(to_year_values, from_year_values, ids):
            triggered = dash.callback_context.triggered_id
            return self._update_wms_year(
                triggered,
                what="to",
                to_year_values=to_year_values,
                from_year_values=from_year_values,
                ids=ids,
            )

        @callback(
            Output("wms-added", "data", allow_duplicate=True),
            Input({"type": "wms-not-contains", "index": dash.ALL}, "value"),
            State({"type": "wms-not-contains", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def update_wms_not_contains(values, ids):
            triggered = dash.callback_context.triggered_id
            if triggered is None:
                return dash.no_update
            wms_name = triggered["index"]
            try:
                # convert list string etc. to python list
                not_contains = eval(get_index(values, ids, wms_name))
            except Exception:
                not_contains = str(get_index(values, ids, wms_name))
            if not not_contains or not_contains == "None":
                not_contains = None
            self._construct_wms_obj(wms_name, not_contains=not_contains)
            return True

        @callback(
            Output("wms-added", "data", allow_duplicate=True),
            Input({"type": "wms-contains", "index": dash.ALL}, "value"),
            State({"type": "wms-contains", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def update_wms_contains(values, ids):
            triggered = dash.callback_context.triggered_id
            if triggered is None:
                return dash.no_update
            wms_name = triggered["index"]
            try:
                # convert list string etc. to python list
                contains = eval(get_index(values, ids, wms_name))
            except Exception:
                contains = str(get_index(values, ids, wms_name))
            if not contains or contains == "None":
                contains = None
            self._construct_wms_obj(wms_name, contains=contains)
            return True

        self.app.clientside_callback(
            """
            function(href) {
                var h = window.innerHeight;
                var v = window.innerWidth;
                return {'height': h, 'width': v};
            }
            """,
            Output("viewport-container", "data"),
            Input("url", "href"),
        )

    def _get_column_dropdown_options(self):
        if self.concatted_data is None:
            return []
        columns = set(self.concatted_data.columns).difference(
            {
                "__file_path",
                "_unique_id",
                "minx",
                "miny",
                "maxx",
                "maxy",
                "geometry",
            }
        )
        return [{"label": col, "value": col} for col in sorted(columns)]

    def _update_table(self, data, column_dropdown, style_table):
        if not data:
            return None, None, style_table | {"height": "1vh"}, None
        if isinstance(column_dropdown, _UseColumns):
            column_dropdown = [
                {"label": col}
                for col in next(iter(data))
                if col not in ["minx", "miny", "maxx", "maxy", "__file_path"]
            ]
        elif column_dropdown is None:
            column_dropdown = self._get_column_dropdown_options()
        all_columns = {x["label"] for x in column_dropdown}
        if not self.splitted:
            all_columns = all_columns.difference({"split_index"})
        height = min(40, len(data) * 5 + 5)
        for x in data:
            x["id"] = x.pop("_unique_id")
        columns_union = set()
        for x in data:
            columns_union |= set(x)
        columns = [{"name": k, "id": k} for k in columns_union if k in all_columns]
        return (
            columns,
            data,
            style_table | {"height": f"{height}vh"},
            ["id"],
        )

    def _delete_file(self, n_clicks_list, delete_ids):
        path_to_delete = get_index_if_clicks(n_clicks_list, delete_ids)
        if path_to_delete is None:
            return dash.no_update, dash.no_update
        debug_print(f"path to delete: {path_to_delete}")
        for path in dict(self.selected_files):
            if path_to_delete in [path, Path(path).stem]:
                self.selected_files.pop(path)

        for path in list(self.loaded_data):
            if path_to_delete in path:
                del self.loaded_data[path]

        self.bounds_series = self.bounds_series[
            lambda x: ~x.index.str.contains(path_to_delete)
        ]
        return None, None

    def _nested_bounds_to_bounds(
        self,
        bounds: list[list[float]],
    ) -> tuple[float, float, float, float]:
        if bounds is None and self.bounds is None:
            return (
                sg.to_gdf(reversed(self.center), 4326)
                .to_crs(3035)
                .buffer(165_000 / (self.zoom**1.25))
                .to_crs(4326)
                .total_bounds
            )
        elif self.bounds is not None:
            bounds = self.bounds
        if isinstance(bounds, str):
            bounds = json.loads(bounds)
        mins, maxs = bounds
        miny, minx = mins
        maxy, maxx = maxs
        return minx, miny, maxx, maxy

    def _concat_data(
        self, bounds, paths: list[str] | None = None
    ) -> tuple[pl.DataFrame | None, list[dbc.Alert] | None]:
        dfs = []
        alerts = set()
        for path in self.selected_files:
            for key in self.loaded_data:
                if paths and (path not in paths and key not in paths):
                    continue
                if path not in key:
                    continue
                df = self.loaded_data[key].lazy().with_columns(__file_path=pl.lit(key))
                if self.splitted:
                    df = get_split_index(df)
                if bounds is not None:
                    df = filter_by_bounds(df, bounds)
                if self._deleted_categories and self.column in df:
                    expression = (
                        pl.col(self.column).is_in(list(self._deleted_categories))
                        == False
                    )
                    if self.nan_label in self._deleted_categories:
                        expression &= pl.col(self.column).is_not_null()
                    df = df.filter(expression)
                elif (
                    self.nan_label in self._deleted_categories and self.column not in df
                ):
                    continue
                df = df.collect()
                if not len(df):
                    continue
                # filtering by function after collect because LazyFrame doesnt implement to_pandas.
                if self._filters.get(path, None) is not None:
                    df, alert = _filter_data(df, self._filters[path])
                    alerts.add(alert)
                if not len(df):
                    continue
                dfs.append(df)

        if dfs:
            df = pl.concat(dfs, how="diagonal_relaxed")
        else:
            df = None

        if not alerts:
            alerts = None
        else:
            alerts = [
                dbc.Alert(txt, color="warning", dismissable=True)
                for txt in alerts
                if txt
            ]

        return df, alerts

    def _map_constructor(
        self, data: dl.LayersControl, preferCanvas=True, zoomAnimation=False, **kwargs
    ) -> dl.Map:
        return dl.Map(
            center=self.center,
            bounds=self.bounds,
            zoom=self.zoom,
            children=self._map_children + [data],
            preferCanvas=preferCanvas,
            zoomAnimation=zoomAnimation,
            id="map",
            style={
                "height": "90vh",
            },
            **kwargs,
        )

    def _add_wms(self, wms_checklist, bounds):
        wms_layers = []
        all_tiles_lists = []
        for wms_name, wms_obj in self.wms.items():
            if wms_name not in wms_checklist:
                all_tiles_lists.append(None)
                continue
            tiles = wms_obj.filter_tiles(shapely.box(*bounds))
            for tile in tiles:
                is_checked: bool = tile in self.wms_layers_checked[wms_name]
                if not is_checked:
                    continue
                wms_layers.append(
                    dl.Overlay(
                        dl.WMSTileLayer(
                            url=wms_obj.url,
                            layers=tile,
                            format="image/png",
                            transparent=True,
                        ),
                        name=tile,
                        checked=is_checked,
                    )
                )

            all_tiles_lists.append(
                self._get_wms_list(tiles, wms_name, self.wms_layers_checked[wms_name])
            )
        return wms_layers, all_tiles_lists

    def _get_wms_list(
        self, tiles: list[str], wms_name: str, wms_layers_checked: list[str]
    ) -> html.Ul:
        height = int(min(20, 10 + len(tiles)))
        assert isinstance(wms_layers_checked, list), wms_layers_checked

        return html.Ul(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            get_button_with_tooltip(
                                "x",
                                id={
                                    "type": "checked-btn-wms",
                                    "wms_name": wms_name,
                                    "tile": tile,
                                },
                                style=(
                                    {
                                        "color": FILE_CHECKED_COLOR,
                                        "background-color": FILE_CHECKED_COLOR,
                                    }
                                    if tile in wms_layers_checked
                                    else {
                                        "color": OFFWHITE,
                                        "background-color": OFFWHITE,
                                    }
                                ),
                                tooltip_text="Show/hide data",
                            )
                        ),
                        dbc.Col(
                            dbc.Label([tile]),
                            width="auto",
                        ),
                    ]
                )
                for tile in tiles
            ],
            style={
                "height": f"{height}vh",
                "overflow-y": "auto",
                "overflow-x": "visible",
            },
        )

    def _update_wms_year(
        self, triggered, what: str, from_year_values, to_year_values, ids
    ):
        if triggered is None:
            return dash.no_update, [dash.no_update for _ in ids]
        wms_name = triggered["index"]
        i = [x["index"] for x in ids].index(wms_name)
        if from_year_values[i] is None or to_year_values[i] is None:
            return dash.no_update, [dash.no_update for _ in ids]
        from_year = max(from_year_values[i], self.wms[wms_name]._min_year)
        from_year_values[i] = from_year
        to_year = min(to_year_values[i], CURRENT_YEAR)
        to_year_values[i] = to_year
        years = list(range(from_year, to_year + 1))
        self._construct_wms_obj(wms_name, years=years)
        values = from_year_values if what == "from" else to_year_values
        return True, values

    def _construct_wms_obj(self, wms_name: str, **kwargs):
        constructor = self.wms[wms_name].__class__
        defaults = dict(
            zip(
                inspect.getfullargspec(constructor).args[1:],
                inspect.getfullargspec(constructor).defaults,
                strict=True,
            )
        )

        if wms_name in self.wms:
            current_kwargs = {
                arg: value
                for arg, value in self.wms[wms_name].__dict__.items()
                if arg in defaults
            }
        else:
            current_kwargs = defaults

        current_kwargs["show"] = True

        self.wms[wms_name] = constructor(**(current_kwargs | kwargs))

    def _get_self_as_dict(self) -> dict[str, Any]:
        data = {
            key: value
            for key, value in self.__dict__.items()
            if key
            not in [
                "app",
                "bounds_series",
                "loaded_data",
                "bounds",
                "concatted_data",
                "logger",
            ]
            and not key.startswith("_")
            and not (isinstance(value, (dict, list, tuple)) and not value)
        }

        if self.selected_files:
            data = {
                "data": {
                    key: self._filters.get(key)
                    for key in data.pop("selected_files", [])
                },
                **data,
            }
        else:
            data.pop("selected_files", [])

        if self._file_browser.favorites:
            data["favorites"] = self._file_browser.favorites

        data["file_system"] = data["file_system"].__class__.__name__ + "()"

        if "selected_features" in data:
            data["selected_features"] = [
                x["_unique_id"] for x in data["selected_features"].values()
            ]
        return data

    def _get_self_as_string(self, data: dict[str, Any]) -> str:
        def maybe_to_string(key: str, value: Any):
            if isinstance(value, str) and key not in ["file_system"]:
                return f"'{value}'"
            return value

        txt = ", ".join(f"{k}={maybe_to_string(k, v)}" for k, v in data.items())
        return f"{self.__class__.__name__}({txt})"

    def __str__(self) -> str:
        """String representation."""
        data = self._get_self_as_dict()
        return self._get_self_as_string(data)


def _get_max_rows_displayed_component(max_rows: int):
    return [
        dbc.Row(html.Div("Max rows displayed")),
        dbc.Row(
            dbc.Input(
                id="max_rows_value",
                value=max_rows,
                type="number",
                debounce=1,
            ),
            style={"width": "13vh"},
        ),
        dbc.Tooltip(
            "Set number of rows rendered. Currently showing random sample of data to avoid crashing.",
            target="max_rows_value",
            delay={"show": 500, "hide": 100},
        ),
    ]


def _change_order(explorer, n_clicks_list, ids, buttons, what: str):
    if what not in ["up", "down"]:
        raise ValueError(what)
    path = get_index_if_clicks(n_clicks_list, ids)
    if path is None or not buttons:
        return dash.no_update, dash.no_update
    i = list(reversed(explorer.selected_files)).index(path)
    if (what == "up" and i == 0) or (what == "down" and i == len(buttons) - 1):
        return dash.no_update, dash.no_update
    if what == "up":
        i2 = i - 1
    else:
        i2 = i + 1
    buttons[i], buttons[i2] = buttons[i2], buttons[i]
    keys = list(reversed(explorer.selected_files))
    values = list(reversed(explorer.selected_files.values()))
    keys[i], keys[i2] = keys[i2], keys[i]
    values[i], values[i2] = values[i2], values[i]
    explorer.selected_files = dict(reversed(list(zip(keys, values, strict=False))))
    return buttons, True


def _named_color_to_hex(color: str) -> str:
    return mcolors.to_hex(color)


def _get_colorpicker_container(color_dict: dict[str, str]) -> html.Div:
    def to_python_type(x):
        if isinstance(x, Number):
            return float(x)
        return x

    color_dict = {
        to_python_type(column_value): color
        for column_value, color in color_dict.items()
    }

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            type="color",
                            id={
                                "type": "colorpicker",
                                "column_value": column_value,
                            },
                            value=color,
                            style={"width": 50, "height": 50},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Label([column_value]),
                        width="auto",
                    ),
                    dbc.Col(
                        get_button_with_tooltip(
                            "❌",
                            id={
                                "type": "delete-cat-btn",
                                "index": column_value,
                            },
                            n_clicks=0,
                            style={
                                "color": "red",
                                "border": "none",
                                "background": "none",
                                "cursor": "pointer",
                                "marginLeft": "auto",
                            },
                            tooltip_text="Remove all data in this category",
                        ),
                        width="auto",
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "flex-start",
                    "alignItems": "center",
                    "marginBottom": "5px",
                },
            )
            for column_value, color in color_dict.items()
        ],
        id="color-container",
    )


def _get_df(path, loaded_data, paths_concatted, override: bool = False):
    t = perf_counter()
    debug_print("_get_df", path, path in loaded_data, path in paths_concatted)
    if path in loaded_data and not override and path in paths_concatted:
        debug_print("_get_df", "00000")
        return []
    if path in loaded_data:
        debug_print("_get_df", "11111", path)
        df = loaded_data[path].with_columns(__file_path=pl.lit(path))
        return [df]

    if paths_concatted:  # is not None and len(concatted_data):
        debug_print("_get_df", "222")
        matches = [
            key
            for key in loaded_data
            if path in key and (override or key not in paths_concatted)
        ]
        if matches:
            matches = [
                loaded_data[key].with_columns(__file_path=pl.lit(key))
                for key in matches
            ]

    else:
        debug_print("_get_df", "333")
        matches = [
            df.with_columns(__file_path=pl.lit(key))
            for key, df in loaded_data.items()
            if path in key
        ]
    debug_print(len(matches), "matches for", path)
    return matches


def _add_data_one_path(
    path,
    bounds,
    column,
    is_numeric,
    color_dict,
    bins,
    max_rows,
    concatted_data,
    nan_color,
    alpha,
):
    ns = Namespace("onEachFeatureToggleHighlight", "default")
    data = []
    if concatted_data is None:
        return (
            [
                dl.Overlay(
                    dl.GeoJSON(id={"type": "geojson", "filename": path}),
                    name=_get_name(path),
                    id={"type": "geojson-overlay", "filename": path},
                    checked=True,
                )
            ],
            None,
            False,
        )
    try:
        df = concatted_data.filter(pl.col("__file_path").str.contains(path))
    except Exception as e:
        raise type(e)(f"{e}: {path} - {concatted_data['__file_path']}")

    if not len(df):
        return (
            [
                dl.Overlay(
                    dl.GeoJSON(id={"type": "geojson", "filename": path}),
                    name=_get_name(path),
                    id={"type": "geojson-overlay", "filename": path},
                    checked=True,
                )
            ],
            None,
            False,
        )

    df = filter_by_bounds(df, bounds)

    rows_are_hidden = len(df) > max_rows

    if len(df) > max_rows:
        df = df.sample(max_rows)
    df = df.to_pandas()
    df["geometry"] = shapely.from_wkb(df["geometry"])
    df = GeoDataFrame(df, crs=4326)

    out_alert = None

    if column and column in df and not is_numeric:
        df["_color"] = df[column].map(color_dict)
    elif column and bins is None:
        df["_color"] = nan_color
    elif column and column in df:
        notnas = df[df[column].notna()]
        if bins is not None and len(bins) == 1:
            notnas["_color"] = next(
                iter(color for color in color_dict.values() if color != nan_color)
            )
        else:
            conditions = [
                (notnas[column] < bins[1]) & (notnas[column].notna()),
                *[
                    (notnas[column] >= bins[i])
                    & (notnas[column] < bins[i + 1])
                    & (notnas[column].notna())
                    for i in np.arange(2, len(bins) - 1)
                ],
                (notnas[column] >= bins[-1]) & (notnas[column].notna()),
            ]
            choices = np.arange(len(conditions)) if bins is not None else None
            try:
                notnas["_color"] = [
                    color_dict[x] for x in np.select(conditions, choices)
                ]
            except KeyError as e:
                raise KeyError(e, color_dict, conditions, choices, bins) from e
        df = pd.concat([notnas, df[df[column].isna()]])
    if column and column not in df:
        data.append(
            dl.Overlay(
                dl.GeoJSON(
                    data=df.__geo_interface__,
                    style={
                        "color": nan_color,
                        "fillColor": nan_color,
                        "weight": 2,
                        "fillOpacity": alpha,
                    },
                    onEachFeature=ns("yellowIfHighlighted"),
                    pointToLayer=ns("pointToLayerCircle"),
                    hideout=dict(
                        circleOptions=dict(fillOpacity=1, stroke=False, radius=5),
                    ),
                    id={"type": "geojson", "filename": path},
                ),
                name=_get_name(path),
                checked=True,
                id={"type": "geojson-overlay", "filename": path},
            )
        )
    elif column:
        data.append(
            dl.Overlay(
                dl.LayerGroup(
                    [
                        dl.GeoJSON(
                            data=(df[df["_color"] == color_]).__geo_interface__,
                            style={
                                "color": color_,
                                "fillColor": color_,
                                "weight": 2,
                                "fillOpacity": alpha,
                            },
                            onEachFeature=ns("yellowIfHighlighted"),
                            pointToLayer=ns("pointToLayerCircle"),
                            id={
                                "type": "geojson",
                                "filename": path + color_,
                            },
                            hideout=dict(
                                circleOptions=dict(
                                    fillOpacity=1, stroke=False, radius=5
                                ),
                            ),
                        )
                        for color_ in df["_color"].unique()
                        if pd.notna(color_)
                    ]
                    + [
                        dl.GeoJSON(
                            data=df[df[column].isna()].__geo_interface__,
                            style={
                                "color": nan_color,
                                "fillColor": nan_color,
                                "weight": 2,
                                "fillOpacity": alpha,
                            },
                            id={
                                "type": "geojson",
                                "filename": path + "_nan",
                            },
                            onEachFeature=ns("yellowIfHighlighted"),
                            pointToLayer=ns("pointToLayerCircle"),
                            hideout=dict(
                                circleOptions=dict(
                                    fillOpacity=1, stroke=False, radius=5
                                ),
                            ),
                        )
                    ]
                ),
                name=_get_name(path),
                checked=True,
                id={"type": "geojson-overlay", "filename": path},
            )
        )
    else:
        # no column
        color = color_dict[_get_name(path)]
        data.append(
            dl.Overlay(
                dl.GeoJSON(
                    data=df.__geo_interface__,
                    style={
                        "color": color,
                        "fillColor": color,
                        "weight": 2,
                        "fillOpacity": alpha,
                    },
                    onEachFeature=ns("yellowIfHighlighted"),
                    pointToLayer=ns("pointToLayerCircle"),
                    hideout=dict(
                        circleOptions=dict(fillOpacity=1, stroke=False, radius=5),
                    ),
                    id={"type": "geojson", "filename": path},
                ),
                name=_get_name(path),
                checked=True,
                id={"type": "geojson-overlay", "filename": path},
            )
        )
    return data, out_alert, rows_are_hidden


def polars_isna(df):
    try:
        return (df.is_nan()) | (df.is_null())
    except pl.exceptions.InvalidOperationError:
        return df.is_null()


def filter_by_bounds(df: pl.DataFrame, bounds: tuple[float]) -> pl.DataFrame:
    minx, miny, maxx, maxy = bounds

    df = df.filter(
        (pl.col("minx") <= float(maxx))
        & (pl.col("maxx") >= float(minx))
        & (pl.col("miny") <= float(maxy))
        & (pl.col("maxy") >= float(miny))
    )
    return df


def read_nrows(file, nrow: int, nth_batch: int, file_system) -> pyarrow.Table:
    """Read first n rows of a parquet file."""
    for _, batch in zip(
        range(nth_batch + 1),
        pq.ParquetFile(file, filesystem=file_system).iter_batches(nrow),
        strict=False,
    ):
        pass
    return pyarrow.Table.from_batches([batch]).to_pandas()


def _read_and_to_4326(path: str, file_system, **kwargs) -> GeoDataFrame:
    if "-_-" in path:
        rows = path.split("-_-")[-1]
        nrow, nth_batch = rows.split("-")
        nrow = int(nrow)
        nth_batch = int(nth_batch)
        path = path.split("-_-")[0]
        try:
            df = read_nrows(path, nrow, nth_batch, file_system=file_system)
        except Exception:
            df = read_nrows(path, nrow, nth_batch, file_system=None)
        metadata = _get_geo_metadata(path, file_system)
        primary_column = metadata["primary_column"]
        geo_metadata = metadata["columns"][primary_column]
        crs = geo_metadata["crs"]
        df["geometry"] = GeoSeries.from_wkb(df[primary_column])
        if primary_column != "geometry":
            df = df.drop(primary_column, axis=1)
        df = GeoDataFrame(df, crs=crs)
        return _fix_df(df)
    df = sg.read_geopandas(path, file_system=file_system, **kwargs)
    return _fix_df(df)


def _fix_df(df: GeoDataFrame) -> GeoDataFrame:
    df["area"] = df.area
    if df["area"].median() > 10:
        df["area"] = df["area"].astype(int)
    if df.crs is not None:
        df = df.to_crs(4326)
    bounds = df.geometry.bounds.astype("float32[pyarrow]")
    df[["minx", "miny", "maxx", "maxy"]] = bounds[["minx", "miny", "maxx", "maxy"]]
    return df


def _get_unique_id(df, i):
    """Float column of 0.0, 0.01, ..., 3.1211 etc."""
    divider = 10 ** len(str(len(df)))
    return (np.array(range(len(df))) / divider) + i


def _read_files(explorer, paths: list[str], **kwargs) -> None:
    if not paths:
        return
    paths = list(paths)
    backend = "threading" if len(paths) <= 3 else "loky"
    with joblib.Parallel(len(paths), backend=backend) as parallel:
        more_data = parallel(
            joblib.delayed(_read_and_to_4326)(
                path, file_system=explorer.file_system, **kwargs
            )
            for path in paths
        )
    for path, df in zip(paths, more_data, strict=True):
        for col in df.columns:
            if is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = [str(x) for x in df[col].dt.round("d")]
                except Exception:
                    df = df.drop(col, axis=1)
        df["__file_path"] = path
        explorer._max_unique_id_int += 1
        df["_unique_id"] = _get_unique_id(df, explorer._max_unique_id_int)
        if isinstance(df, GeoDataFrame):
            df = df.assign(geometry=df.geometry.to_wkb())
        df = pl.from_pandas(df)
        explorer.loaded_data[path] = df


def _random_color(min_diff: int = 50) -> str:
    """Get a random hex color code that is not too gray.

    Args:
        min_diff: minimum total distance between red, green and blue.
            Maximum possible value will be 510, if one value is 0 and another is 255 (the third one will not matter).
    """
    while True:
        r, g, b = np.random.choice(range(256), size=3)
        if abs(r - g) + abs(r - b) > min_diff:
            return f"#{r:02x}{g:02x}{b:02x}"


def _get_name(path):
    return Path(path).stem


def _get_child_paths(paths, file_system):
    child_paths = []
    for path in paths:
        suffix = Path(path).suffix
        if suffix:
            these_child_paths = list(
                file_system.glob(str(Path(path) / f"**/*{suffix}"))
            )
            if not these_child_paths:
                child_paths.append(path)
            else:
                child_paths += these_child_paths
        else:
            child_paths.append(path)
    return child_paths


def _get_bounds_series(path, file_system):
    paths = [
        _standardize_path(path)
        for path in (
            set(file_system.glob(str(Path(path) / "*.parquet")))
            | set(file_system.glob(str(Path(path) / "**/*.parquet")))
        )
    ]
    if not paths:
        paths = [path]
    bounds_series = sg.get_bounds_series(paths, file_system=file_system)
    if len(bounds_series):
        bounds_series = bounds_series.to_crs(4326)
    return bounds_series


def get_index(values: list[Any], ids: list[Any], index: Any):
    i = [x["index"] for x in ids].index(index)
    return values[i]


def get_zoom_from_bounds(
    lon_min, lat_min, lon_max, lat_max, map_width_px, map_height_px
):
    """Estimate Leaflet zoom level for a bounding box and viewport size.

    Parameters:
        lat_min, lon_min: coordinates of bottom-left corner
        lat_max, lon_max: coordinates of top-right corner
        map_width_px: width of map in pixels
        map_height_px: height of map in pixels
    Returns:
        Approximate zoom level (can be float)
    """

    # Earth's circumference in meters (WGS 84)
    C = 40075016.686

    # Center latitude for cosine correction
    lat = (lat_min + lat_max) / 2
    lat_rad = math.radians(lat)

    # Width and height in degrees
    lon_delta = abs(lon_max - lon_min)
    lat_delta = abs(lat_max - lat_min)

    # Adjusted width in meters at that latitude
    width_m = lon_delta * (C / 360.0) * math.cos(lat_rad)
    height_m = lat_delta * (C / 360.0)

    # Meters per pixel required
    meters_per_pixel_w = width_m / map_width_px
    meters_per_pixel_h = height_m / map_height_px
    meters_per_pixel = max(meters_per_pixel_w, meters_per_pixel_h)

    # Invert the meters/pixel formula:
    zoom = math.log2(C * math.cos(lat_rad) / (meters_per_pixel * 256))
    debug_print("get_zoom_from_bounds", map_width_px, map_height_px, zoom)
    return round(zoom, 2)


def get_index_if_clicks(n_clicks_list, ids) -> str | None:
    if not any(n_clicks_list):
        return None
    triggered = dash.callback_context.triggered_id
    if not isinstance(triggered, dict):
        return None
    triggered_index = triggered["index"]
    for index in (i for i, id_ in enumerate(ids) if id_["index"] == triggered_index):
        n_clicks = n_clicks_list[index]
        if n_clicks:
            return ids[index]["index"]
    return None


def get_data_table(*data, title_id: str, table_id: str, div_id: str, clear_id: str):
    return dbc.Col(
        [
            dbc.Row(
                [
                    dbc.Col(html.B(id=title_id)),
                    *[dbc.Col(x) for x in data],
                    dbc.Col(
                        html.Button(
                            "❌ Clear table",
                            id=clear_id,
                            style={
                                "color": "red",
                                "border": "none",
                                "background": "none",
                                "cursor": "pointer",
                            },
                        ),
                        width=2,
                    ),
                ]
            ),
            dbc.Row(
                dash_table.DataTable(
                    id=table_id,
                    style_header={
                        "background-color": "#2f2f2f",
                        "color": "white",
                        "fontWeight": "bold",
                    },
                    style_data={
                        "background-color": OFFWHITE,
                        "color": "black",
                    },
                    style_table={
                        "overflow-x": "visible",
                        "overflow-y": "scroll",
                        "height": "1vh",
                    },
                    sort_action="native",
                    row_deletable=True,
                ),
            ),
        ],
        style={"display": "none"},
        id=div_id,
    )


def is_jupyter():
    return (
        "JUPYTERHUB_SERVICE_PREFIX" in os.environ
        and "JUPYTERHUB_HTTP_REFERER" in os.environ
    )


def get_split_index(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (
            pl.col("__file_path").map_elements(_get_name, return_dtype=pl.Utf8)
            + " "
            + pl.col("__file_path").cum_count().over("__file_path").cast(pl.Utf8)
        ).alias("split_index")
    )


def _filter_data(df: pl.DataFrame, filter_function: str | None) -> pl.DataFrame:
    filter_function = filter_function.strip()
    try:
        filter_function = eval(filter_function)
    except Exception:
        pass

    if filter_function is None or (
        isinstance(filter_function, str) and filter_function == ""
    ):
        return df, None

    alert = None

    # try to filter with polars, then pandas.loc, then pandas.query
    # no need for pretty code and specific exception handling here, as this a convenience feature
    try:
        # polars needs functions called, pandas does not
        if callable(filter_function):
            filter_function = filter_function(df)
        df = df.filter(filter_function)
    except Exception as e:
        try:
            df = pl.DataFrame(df.to_pandas().loc[filter_function])
        except Exception as e2:
            try:
                df = pl.DataFrame(df.to_pandas().query(filter_function))
            except Exception as e3:
                e_name = type(e).__name__
                e2_name = type(e2).__name__
                e3_name = type(e3).__name__
                e = str(e)
                e2 = str(e2)
                e3 = str(e3)
                if len(e) > 1000:
                    e = e[:997] + "... "
                if len(e2) > 1000:
                    e2 = e2[:997] + "... "
                alert = (
                    f"Filter function failed with polars ({e_name}: {e}) "
                    f"-- and pandas loc: ({e2_name}: {e2}) "
                    f"-- and pandas query: ({e3_name}: {e3}) "
                )

    return df, alert


def _get_force_categorical_button(
    values_no_nans: pl.Series, force_categorical_clicks: int | None
):
    if not values_no_nans.dtype.is_numeric():
        return None
    elif (force_categorical_clicks or 0) % 2 == 0:
        return get_button_with_tooltip(
            "Force categorical",
            id="force-categorical-button",
            n_clicks=force_categorical_clicks,
            tooltip_text="Get all numeric values as a single color group",
            style={
                "background": "white",
                "color": "black",
            },
        )
    else:
        return get_button_with_tooltip(
            "Force categorical",
            id="force-categorical-button",
            tooltip_text="Back to numeric values",
            n_clicks=force_categorical_clicks,
            style={
                "background": "black",
                "color": "white",
            },
        )


class _UseColumns:
    pass


def _get_file_system(path, file_system) -> AbstractFileSystem:
    if file_system is not None:
        return file_system
    if str(path).startswith("gs://"):
        from gcsfs import GCSFileSystem

        return GCSFileSystem()
    return LocalFileSystem()
