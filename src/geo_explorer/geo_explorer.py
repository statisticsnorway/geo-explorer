import datetime
import inspect
import itertools
import json
import logging
import math
import os
import re
import signal
import sys
import time
from collections.abc import Callable
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from functools import wraps
from multiprocessing import cpu_count
from numbers import Number
from pathlib import Path
from pathlib import PurePath
from time import perf_counter
from typing import Any
from typing import ClassVar
import random

import pickle
import geopandas as gpd
import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import joblib
import matplotlib
import matplotlib.colors as mcolors
import msgspec
import numpy as np
import pandas as pd
import polars as pl
import pyarrow
import pyarrow.parquet as pq
import sgis as sg
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
from geopandas.array import GeometryArray
from jenkspy import jenks_breaks
from sgis.io.dapla_functions import _get_geo_metadata
from sgis.io.dapla_functions import _read_pyarrow
from sgis.io.dapla_functions import _get_bounds_parquet
from sgis.io.dapla_functions import _get_bounds_parquet_from_open_file
from sgis.maps.wms import WmsLoader
from shapely import Geometry
from shapely.errors import GEOSException
from shapely.geometry import Point
from sgis import get_common_crs

from .file_browser import FileBrowser
from .fs import LocalFileSystem
from .utils import _clicked_button_style
from .utils import _standardize_path
from .utils import _unclicked_button_style
from .utils import get_button_with_tooltip
from .utils import time_method_call
from .utils import time_function_call

OFFWHITE: str = "#ebebeb"
FILE_CHECKED_COLOR: str = "#3e82ff"
DEFAULT_ZOOM: int = 12
DEFAULT_CENTER: tuple[float, float] = (59.91740845, 10.71394444)
CURRENT_YEAR: int = datetime.datetime.now().year
FILE_SPLITTER_TXT: str = "-_-"
ADDED_COLUMNS = {
    "minx",
    "miny",
    "maxx",
    "maxy",
    "_unique_id",
    "__file_path",
    "geometry",
}
ns = Namespace("onEachFeatureToggleHighlight", "default")

DEBUG: bool = False

_PROFILE_DICT = {}

if DEBUG:

    def debug_print(*args):
        print(
            *(
                f"{type(arg).__name__}: {arg}" if isinstance(arg, Exception) else arg
                for arg in args
            )
        )

else:

    def debug_print(*args):
        pass

    def time_method_call(_) -> Callable:
        def decorator(method):
            @wraps(method)
            def wrapper(self, *args, **kwargs):
                return method(self, *args, **kwargs)

            return wrapper

        return decorator

    def time_function_call(_):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator


def _get_default_sql_query(df: pl.LazyFrame | pl.DataFrame, columns: list[str]) -> str:
    if isinstance(df, pl.LazyFrame):
        mean_area = df.select("area").mean().collect().item()
        min_area = df.select("area").min().collect().item()
    else:
        mean_area = df["area"].mean()
        min_area = df["area"].min()

    columns = [col for col in columns if col != "area"]
    cols = ", ".join(columns[: (min(3, len(columns)))])

    if mean_area is not None and mean_area > min_area:
        where_and_order_by_clauses = f"WHERE area > {mean_area} ORDER BY area DESC"
        cols = "area, " + cols
    else:
        where_and_order_by_clauses = ""
    query = f"SELECT {cols} FROM df {where_and_order_by_clauses} LIMIT 10000"
    return query.replace("  ", " ")


def _get_sql_query_with_col(
    df: pl.LazyFrame, col: str, columns: list[str], all_cols: bool
) -> str:
    if len(df.select(col).unique().collect()) <= 1:
        raise ValueError(f"Not enough unique values in {col} (0 or 1)")

    def maybe_to_string(value: Any):
        if isinstance(value, str):
            return f"'{value}'"
        return value

    cols = [
        col2
        for col2 in columns
        if col2 not in [col, "geometry"] and not col2.startswith("_")
    ]
    random.shuffle(cols)
    if cols and not all_cols:
        cols = f"{col}, " + ", ".join(cols[: (min(3, len(cols)))])
    else:
        cols = "*"

    try:
        value = df.select(col).mean().collect().item()
        assert value is not None
        query = f"SELECT {cols} FROM df WHERE {col} > {value} ORDER BY {col} DESC"
    except Exception:
        value = df.select(pl.col(col).mode().first()).collect().item()
        query = f"SELECT {cols} FROM df WHERE {col}={maybe_to_string(value)}"
    results = pl.sql(query, eager=False).collect()
    if not len(results):
        raise ValueError(f"No rows after query {query}")
    if len(results) > 100_000:
        query += " LIMIT 100000"
    return query


@time_function_call(_PROFILE_DICT)
def read_file(
    path: str, file_system: AbstractFileSystem, **kwargs
) -> tuple[pl.LazyFrame, dict[str, pl.DataType]]:

    if not path.endswith(".parquet") and FILE_SPLITTER_TXT not in path:
        try:
            df = gpd.read_file(path, filesystem=file_system, **kwargs)
            df, dtypes = _geopandas_to_polars(df, path)
        except Exception:
            pd_read_func = getattr(pd, f"read_{Path(path).suffix.strip('.')}")
            df = pd_read_func(path, filesystem=file_system, **kwargs)
            df, dtypes = _pandas_to_polars(df, path)
        return df.lazy(), dtypes
    if FILE_SPLITTER_TXT not in path:
        try:
            # metadata = _get_geo_metadata(path, file_system)
            # primary_column = metadata["primary_column"]
            # df = _read_polars(
            #     path, file_system=file_system, primary_column=primary_column, **kwargs
            # )
            # return _prepare_df(df, path, metadata)
            table = _read_pyarrow(path, file_system=file_system, **kwargs)
            return _pyarrow_to_polars(table, path, file_system)
        except Exception:
            df = sg.read_geopandas(path, file_system=file_system, **kwargs)
            df, dtypes = _geopandas_to_polars(df, path)
            return df.lazy(), dtypes
    rows = path.split(FILE_SPLITTER_TXT)[-1]
    nrow, nth_batch = rows.split("-")
    nrow = int(nrow)
    nth_batch = int(nth_batch)
    path = path.split(FILE_SPLITTER_TXT)[0]
    try:
        table = read_nrows(path, nrow, nth_batch, file_system=file_system)
    except Exception:
        table = read_nrows(path, nrow, nth_batch, file_system=None)
    return _pyarrow_to_polars(table, path, file_system)


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
        max_read_size_per_callback: Defaults to 1e9 bytes (1 GB). Meaning max 1 GB is read at once, then the read
            function is cycled until all data is read. This is because long callbacks time out.
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
    _base_layers: ClassVar[dict[str, dl.BaseLayer]] = {
        "OpenStreetMap": dl.BaseLayer(
            dl.TileLayer("OpenStreetMap"),
            name="OpenStreetMap",
            checked=True,
        ),
        "CartoDB Dark Matter": dl.BaseLayer(
            dl.TileLayer(
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                attribution='&copy; <a href="https://carto.com/">CARTO</a>',
            ),
            name="CartoDB Dark Matter",
            checked=False,
        ),
        "Norge i bilder": dl.BaseLayer(
            dl.TileLayer(
                url="https://opencache.statkart.no/gatekeeper/gk/gk.open_nib_web_mercator_wmts_v2?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=Nibcache_web_mercator_v2&STYLE=default&FORMAT=image/jpgpng&tileMatrixSet=default028mm&tileMatrix={z}&tileRow={y}&tileCol={x}",
                attribution="© Geovekst",
            ),
            name="Norge i bilder",
            checked=False,
        ),
    }
    _map_children: ClassVar[list[Component]] = [
        dl.ScaleControl(position="bottomleft"),
        dl.MeasureControl(
            position="bottomright",
            primaryLengthUnit="meters",
        ),
    ]

    _file_formats_with_metadata: ClassVar[list[Component]] = [
        "parquet",
    ]

    read_func: ClassVar[Callable] = read_file

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
        max_read_size_per_callback: int = 1e9,
        **kwargs,
    ) -> None:
        """Initialiser."""
        self.start_dir = start_dir
        self.port = port
        self.maxZoom = kwargs.get("maxZoom", 40)
        self.minZoom = kwargs.get("minZoom", 4)
        self._kwargs = kwargs  # save kwargs for the "export" button
        self._bounds = None
        self.column = column
        self.color_dict = {
            key: (color if color.startswith("#") else _named_color_to_hex(color))
            for key, color in (color_dict or {}).items()
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
        self._bounds_series = GeoSeries()
        self.selected_files: dict[str, int] = {}
        self._loaded_data: dict[str, pl.LazyFrame] = {}
        self._dtypes: dict[str, dict[str, pl.DataType]] = {}
        self._max_unique_id_int: int = 0
        self._loaded_data_sizes: dict[str, int] = {}
        self._concatted_data: pl.DataFrame | None = None
        self._deleted_categories = set()
        self.selected_features = {}
        self._file_browser = FileBrowser(
            start_dir, file_system=file_system, favorites=favorites
        )
        self._current_table_view = None
        self.max_read_size_per_callback = max_read_size_per_callback
        self._force_categorical = False

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
            debug_print("\n\n\n\n\nget_layout", self._bounds, self.zoom, self.column)
            return dbc.Container(
                [
                    dcc.Location(id="url"),
                    dbc.Row(html.Div(id="alert")),
                    dbc.Row(html.Div(id="alert3")),
                    dbc.Row(html.Div(id="alert4")),
                    dbc.Row(html.Div(id="new-file-added")),
                    html.Div(id="file-deleted"),
                    html.Div(id="query-updated"),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(id="loading", style={"height": "3vh"}),
                                width={
                                    "size": 3,
                                    "order": "first",
                                },
                            ),
                        ],
                        justify="between",
                        id="urls",
                        style={
                            "width": "100vh",
                        },
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                self._map_constructor(
                                    html.Div(id="lc"),
                                    **(
                                        {
                                            "maxZoom": self.maxZoom,
                                            "minZoom": self.minZoom,
                                        }
                                        | kwargs
                                    ),
                                ),
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
                                    dbc.Row(
                                        get_button_with_tooltip(
                                            "Reload categories",
                                            id="reload-categories",
                                            n_clicks=0,
                                            tooltip_text="Get back categories that have been X-ed out",
                                        ),
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
                    html.Div(id="buffer-tip"),
                    *get_button_with_tooltip(
                        "Hard click",
                        id="hard-click",
                        n_clicks=int(self.hard_click),
                        tooltip_text="'Hard' click means that clicking on a geometry triggers all overlapping geometries to be marked",
                    ),
                    dbc.Row(html.Div(id="bottom-alert")),
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
                    dcc.Store(id="update-table", data=None),
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
                    dcc.Store(id="dummy-output"),
                    html.Button(id="query-examples-button", style={"display": "none"}),
                    dcc.Store(id="wms-added"),
                    html.Div(id="data-was-concatted", style={"display": "none"}),
                    html.Div(id="data-was-changed", style={"display": "none"}),
                    html.Div(id="new-data-read", style={"display": "none"}),
                    dbc.Input(id="max_rows_value", style={"display": "none"}),
                    dcc.Store(data=None, id="bins"),
                    dcc.Store(data=False, id="is-numeric"),
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

        self._queries = {}
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
                    self._queries[key] = value
                    continue
                value, dtypes = _geopandas_to_polars(value, key)
                bounds_series_dict[key] = shapely.box(
                    float(value["minx"].min()),
                    float(value["miny"].min()),
                    float(value["maxx"].max()),
                    float(value["maxy"].max()),
                )
                self._loaded_data[key] = value.lazy()
                self._dtypes[key] = dtypes | {"area": pl.Float64()}
                self.selected_files[key] = True

        self.selected_files = dict(reversed(self.selected_files.items()))
        self._bounds_series = GeoSeries(bounds_series_dict)

        # storing bounds here before file paths are loaded. To avoid setting center as the entire map bounds if large data
        if len(self._bounds_series):
            minx, miny, maxx, maxy = self._bounds_series.total_bounds
        else:
            minx, miny, maxx, maxy = None, None, None, None

        if not self.selected_files:
            self.center = center if center is not None else DEFAULT_CENTER
            self.zoom = zoom or DEFAULT_ZOOM
            self.app.layout = get_layout
            self._register_callbacks()
            return

        self._append_to_bounds_series(
            [x for x in self.selected_files if x not in self._loaded_data]
        )

        temp_center = center if center is not None else DEFAULT_CENTER
        _read_files(
            self,
            [x for x in self.selected_files if x not in self._loaded_data],
            mask=Point(reversed(temp_center)),
        )

        # dataframe dicts as input data are currently sorted first because they were added to loaded_data first.
        # now to get back original order
        # also resetting id count. Only needed in init
        self._max_unique_id_int: int = 0
        loaded_data_sorted = {}
        for x in reversed(data or []):
            if isinstance(x, dict):
                for key in reversed(x):
                    key = _standardize_path(key)
                    if key not in self._loaded_data:
                        continue
                    df = self._loaded_data[key]
                    loaded_data_sorted[key] = df.with_columns(
                        _unique_id=_get_unique_id(self._max_unique_id_int)
                    )
                    self._max_unique_id_int += 1
            else:
                x = _standardize_path(x)
                df = self._loaded_data[x]
                loaded_data_sorted[x] = df.with_columns(
                    _unique_id=_get_unique_id(self._max_unique_id_int)
                )
                self._max_unique_id_int += 1

        self._loaded_data = loaded_data_sorted

        if center is not None:
            self.center = center
        elif self._loaded_data and all((minx, miny, maxx, maxy)):
            self.center = ((maxy + miny) / 2, (maxx + minx) / 2)
        else:
            self.center = DEFAULT_CENTER

        if zoom is not None:
            self.zoom = zoom
        elif self._loaded_data and all((minx, miny, maxx, maxy)):
            self.zoom = get_zoom_from_bounds(minx, miny, maxx, maxy, 800, 600)
        else:
            self.zoom = DEFAULT_ZOOM

        self.app.layout = get_layout

        for unique_id in selected_features if selected_features is not None else []:
            i = int(float(unique_id))
            path = list(self._loaded_data)[i]
            properties, _ = self._get_selected_feature(unique_id, path, bounds=None)
            self.selected_features[unique_id] = properties

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

            if DEBUG:
                print("\nself.__dict__")
                for k, v in sorted(self.__dict__.items()):
                    print()
                    print(k)
                    print(k, v)

                print("\nprofile")
                for k, v in reversed(
                    sorted(_PROFILE_DICT.items(), key=lambda x: x[1][1])
                ):
                    print(k, v)

            data = self._get_self_as_dict()
            defaults = inspect.getfullargspec(self.__class__).kwonlydefaults
            data = {
                key: value for key, value in data.items() if value != defaults.get(key)
            } | self._kwargs
            txt = self._get_self_as_string(data)
            return html.Div(f"{txt}.run()"), True

        @callback(
            Output("buffer-tip", "children"),
            Input("alert", "children"),
            State("map", "zoom"),
        )
        def maybe_tip_about_buffer(_, zoom):
            if self._concatted_data is None:
                return None
            area_max = self._concatted_data.select("area").max().collect().item()
            if area_max is None:
                return None
            return (
                html.B(
                    "Tip: add query 'df.buffer(3000)' if geometries are difficult to see",
                    style={"font-size": 20},
                )
                if zoom <= 11 and 0 < area_max < 10000
                else None
            )

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
            self._bounds = bounds
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
        @time_method_call(_PROFILE_DICT)
        def append_path(load_parquet, load_parquet_ids, ids):
            triggered = dash.callback_context.triggered_id
            if not any(load_parquet) or not triggered:
                return dash.no_update
            selected_path = triggered["index"]
            n_clicks = get_index(load_parquet, load_parquet_ids, selected_path)
            if selected_path in self.selected_files or not n_clicks:
                return dash.no_update
            try:
                self._append_to_bounds_series([selected_path])
            except Exception as e:
                return dbc.Alert(
                    f"Couldn't read {selected_path}. {type(e)}: {e}",
                    color="warning",
                    dismissable=True,
                )
            self.selected_files[selected_path] = True
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
        @time_method_call(_PROFILE_DICT)
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
                f"{len(missing or [])=}, {len(self._loaded_data)=}",
            )

            if isinstance(triggered, dict) and triggered["type"] == "checked-btn":
                path = get_index_if_clicks(checked_clicks, checked_ids)
                if path is None:
                    return dash.no_update, dash.no_update, dash.no_update

            if triggered != "missing":
                box = shapely.box(*self._nested_bounds_to_bounds(bounds))
                files_in_bounds = set(sg.sfilter(self._bounds_series, box).index)
                files_in_bounds |= set(
                    self._bounds_series[lambda x: (x.isna()) | (x.is_empty)].index
                )

                def is_checked(path) -> bool:
                    return any(
                        is_checked
                        for sel_path, is_checked in self.selected_files.items()
                        if sel_path in path
                    )

                missing = list(
                    {
                        path
                        for path in files_in_bounds
                        if path not in self._loaded_data and is_checked(path)
                    }
                )
                if not all(path in self._loaded_data_sizes for path in missing):
                    paths_missing_size = [
                        path for path in missing if path not in self._loaded_data_sizes
                    ]
                    with ThreadPoolExecutor() as executor:
                        more_sizes = {
                            path: x["size"]
                            for path, x in zip(
                                paths_missing_size,
                                executor.map(self.file_system.info, paths_missing_size),
                                strict=True,
                            )
                        }
                    self._loaded_data_sizes |= more_sizes

            if triggered != "interval-component":
                new_missing = []
                for path in missing:
                    size = self._loaded_data_sizes[path]
                    if size < self.max_read_size_per_callback:
                        new_missing.append(path)
                        continue
                    with self.file_system.open(path, "rb") as file:
                        nrow = pq.read_metadata(file).num_rows
                    n = 30
                    rows_to_read = nrow // n
                    for i in range(n):
                        new_path = path + f"{FILE_SPLITTER_TXT}{rows_to_read}-{i}"
                        if new_path in missing:
                            continue
                        new_missing.append(new_path)
                        self._loaded_data_sizes[new_path] = size / n
                        self._bounds_series = pd.concat(
                            [
                                self._bounds_series,
                                GeoSeries({new_path: self._bounds_series.loc[path]}),
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
                f"{len(missing)=}, {len(self._loaded_data)=}, {new_data_read=}, {disabled=}",
            )

            return new_data_read, missing, disabled

        @callback(
            Output("is_splitted", "data"),
            Output("column-dropdown", "value"),
            Input("splitter", "n_clicks"),
        )
        def set_column_to_split_index(splitter_clicks):
            if not self.selected_files:
                return False, None
            triggered = dash.callback_context.triggered_id
            if triggered is not None:
                self.splitted = not self.splitted
                self.column = None if not self.splitted else self.column
            if self.splitted:
                self._deleted_categories = set()
                return self.splitted, "split_index"
            return self.splitted, self.column

        @callback(
            Output(
                {"type": "query", "index": dash.MATCH}, "value", allow_duplicate=True
            ),
            Input({"type": "query-view", "index": dash.ALL}, "is_open"),
            prevent_initial_call=True,
        )
        def apply_query(is_open):
            assert sum(is_open) in [0, 1], is_open
            if any(is_open):
                return dash.no_update
            triggered = dash.callback_context.triggered_id
            path = triggered["index"]
            return self._queries.get(path)

        @callback(
            Output(
                {"type": "query-copy", "index": dash.MATCH},
                "value",
                allow_duplicate=True,
            ),
            Input(
                {"type": "query-select-btn", "index": dash.ALL, "query": dash.ALL},
                "n_clicks",
            ),
            State(
                {"type": "query-select-btn", "index": dash.ALL, "query": dash.ALL},
                "id",
            ),
            prevent_initial_call=True,
        )
        def apply_query_copy(n_clicks, ids):
            path = get_index_if_clicks(n_clicks, ids)
            if not path:
                return dash.no_update
            triggered = dash.callback_context.triggered_id
            query = triggered["query"]
            old_query = self._queries.get(path)
            if (
                old_query
                and old_query.startswith("pl.col")
                and query.startswith("pl.col")
            ):
                query = f"{old_query}, {query}"
            self._queries[path] = query
            return query

        @callback(
            Input({"type": "query-copy", "index": dash.ALL}, "n_blur"),
            State({"type": "query-copy", "index": dash.ALL}, "value"),
            State({"type": "query-copy", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def update_query_from_copy(n_blur, queries: list[str], ids: list[str]):
            self._update_query(queries, ids)

        @callback(
            Output("query-updated", "children", allow_duplicate=True),
            Input({"type": "query", "index": dash.ALL}, "value"),
            Input({"type": "query", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def update_query(queries: list[str], ids: list[str]):
            alerts = self._update_query(queries, ids)
            return alerts

        @callback(
            Output("data-was-concatted", "children"),
            Output(
                "data-was-changed",
                "children",
            ),
            Output("alert3", "children", allow_duplicate=True),
            Output("update-table", "data", allow_duplicate=True),
            Input("new-data-read", "children"),
            Input("file-deleted", "children"),
            Input("is_splitted", "data"),
            Input("query-updated", "children"),
            State("debounced_bounds", "value"),
            prevent_initial_call=True,
        )
        @time_method_call(_PROFILE_DICT)
        def concat_data(
            new_data_read,
            file_deleted,
            is_splitted,
            query_updated,
            bounds,
        ):
            triggered = dash.callback_context.triggered_id
            debug_print("concat_data", triggered, new_data_read, self.splitted)

            t = perf_counter()
            if not new_data_read and not query_updated:
                return dash.no_update, 1, dash.no_update, dash.no_update

            bounds = self._nested_bounds_to_bounds(bounds)

            if triggered in ["file-deleted"]:
                self._max_unique_id_int = -1
                for path, df in self._loaded_data.items():
                    self._max_unique_id_int += 1
                    id_prev = df.select(pl.col("_unique_id").first()).collect().item()
                    self._loaded_data[path] = df.with_columns(
                        _unique_id=_get_unique_id(self._max_unique_id_int)
                    )
                    for idx in list(self.selected_features):
                        if idx[0] != id_prev[0]:
                            continue

                        # rounding values to avoid floating point precicion problems
                        new_idx = f"{self._max_unique_id_int}.{idx[2:]}"
                        feature = self.selected_features.pop(idx)
                        feature["id"] = new_idx
                        self.selected_features[new_idx] = feature

                update_table = True
            else:
                update_table = dash.no_update

            df, alerts = self._concat_data(bounds)
            self._concatted_data = df

            return 1, 1, alerts, update_table

        @callback(
            Output("file-control-panel", "children"),
            Input("data-was-changed", "children"),
            Input("file-deleted", "children"),
            Input({"type": "order-button-up", "index": dash.ALL}, "n_clicks"),
            Input({"type": "order-button-down", "index": dash.ALL}, "n_clicks"),
            State({"type": "order-button-up", "index": dash.ALL}, "id"),
            State({"type": "order-button-down", "index": dash.ALL}, "id"),
            State("file-control-panel", "children"),
            prevent_initial_call=True,
        )
        @time_method_call(_PROFILE_DICT)
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
                                dbc.Col(
                                    [
                                        dcc.Input(
                                            self._queries.get(path, None),
                                            placeholder="Query (with polars, pandas or sql). E.g. komm_nr == '0301'",
                                            id={
                                                "type": "query",
                                                "index": path,
                                            },
                                            style={"width": "100%"},
                                            debounce=2,
                                        ),
                                        dbc.Tooltip(
                                            "E.g. komm_nr == '0301' or pl.col('komm_nr') == '0301'",
                                            target={
                                                "type": "query",
                                                "index": path,
                                            },
                                            delay={"show": 500, "hide": 100},
                                        ),
                                    ],
                                    width=9,
                                ),
                                dbc.Col(
                                    [
                                        html.Button(
                                            "Expand query",
                                            id={
                                                "type": "query-expand-button",
                                                "index": path,
                                            },
                                        ),
                                        dbc.Modal(
                                            id={
                                                "type": "query-view",
                                                "index": path,
                                            },
                                            is_open=False,
                                            size="xl",
                                            backdrop="static",
                                        ),
                                    ],
                                    width=3,
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
                for path, checked in reversed(self.selected_files.items())
            ]

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
        def check_or_uncheck_wms(n_clicks_list, ids):
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
            Output("update-table", "data", allow_duplicate=True),
            Input({"type": "delete-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "delete-btn", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        @time_method_call(_PROFILE_DICT)
        def delete_file(n_clicks_list, delete_ids):
            return (
                *self._delete_file(n_clicks_list, delete_ids, delete_category=False),
                True,
            )

        @callback(
            Output("file-deleted", "children", allow_duplicate=True),
            Output("alert3", "children", allow_duplicate=True),
            Output("update-table", "data", allow_duplicate=True),
            Output("color-container", "children", allow_duplicate=True),
            Input({"type": "delete-cat-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "delete-cat-btn", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        @time_method_call(_PROFILE_DICT)
        def delete_category(n_clicks_list, delete_ids):
            path_to_delete = get_index_if_clicks(n_clicks_list, delete_ids)
            if path_to_delete is None:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            if not self.column:
                return (
                    *self._delete_file(n_clicks_list, delete_ids, delete_category=True),
                    True,
                    dash.no_update,
                )
            else:
                self._deleted_categories.add(path_to_delete)
                return None, None, True, dash.no_update

        @callback(
            Output({"type": "query-view", "index": dash.MATCH}, "children"),
            Output({"type": "query-view", "index": dash.MATCH}, "is_open"),
            # Input("query-examples-button", "n_clicks"),
            Input({"type": "query-examples-button", "index": dash.ALL}, "n_clicks"),
            Input({"type": "query-expand-button", "index": dash.ALL}, "n_clicks"),
            State({"type": "query-examples-button", "index": dash.ALL}, "id"),
            State({"type": "query-expand-button", "index": dash.ALL}, "id"),
            State("debounced_bounds", "value"),
            prevent_initial_call=True,
        )
        @time_method_call(_PROFILE_DICT)
        def expand_query_panel(examples_clicks, n_clicks, examples_ids, ids, bounds):
            triggered = dash.callback_context.triggered_id

            should_add_examples = (
                isinstance(triggered, dict)
                and triggered["type"] == "query-examples-button"
            )
            if should_add_examples:
                path = get_index_if_clicks(examples_clicks, examples_ids)
            else:
                path = get_index_if_clicks(n_clicks, ids)

            if not path:
                return dash.no_update, False

            if not should_add_examples:
                return (
                    self._query_panel_return_modal(path, queries=None),
                    True,
                )

            df_name = _get_stem(path)
            if sum(_get_stem(x) == df_name for x in self.selected_files) > 1:
                df_name = _get_stem_from_parent(path)

            queries = [
                html.Br(),
                html.Br(),
                html.B(
                    f"Example queries for table: {df_name}", style={"font-size": 16}
                ),
                html.Br(),
                html.B("Example join queries:"),
            ]

            n_queries = len(queries)

            bounds = self._nested_bounds_to_bounds(bounds)
            df, _ = self._concat_data(bounds=bounds, paths=[path], _filter=False)

            if df is None:
                return dash.no_update, dash.no_update

            df_cols = {
                col
                for key, dtypes in self._dtypes.items()
                for col in dtypes
                if path in key
            }

            for join_path in self.selected_files:
                if join_path == path:
                    continue
                join_df_name = _get_stem(join_path)
                if sum(_get_stem(x) == join_df_name for x in self.selected_files) > 1:
                    join_df_name = _get_stem_from_parent(join_path)

                join_df, _ = self._concat_data(
                    bounds=bounds, paths=[join_path], _filter=False
                )

                join_df_cols = {
                    col
                    for key, dtypes in self._dtypes.items()
                    for col in dtypes
                    if join_path in key
                }

                cols_to_keep = join_df_cols.difference(df_cols)
                if not cols_to_keep:
                    continue

                cols_to_keep = ", ".join(
                    f"{join_df_name}.{col}" for col in cols_to_keep
                )

                for col in set(join_df_cols).difference(ADDED_COLUMNS | {"area"}):

                    if col not in df_cols or self._get_dtype(join_path, col).is_float():
                        continue
                    debug_print(f"\n\nquery join: {path=}, {join_path=}, {col=}")
                    try:
                        joined = (
                            df.join(join_df, on=col, how="inner").select(col).collect()
                        )
                    except Exception as e:
                        debug_print(
                            f"\n\nquery join failed: {path=}, {join_path=}, {col=}", e
                        )
                        continue
                    if not len(joined):
                        continue

                    query = f"select {cols_to_keep}, df.* from df inner join {join_df_name} using ({col})"
                    queries.append(html.Br())
                    queries.extend(
                        get_button_with_tooltip(
                            f"inner join: {join_df_name} on {col}",
                            id={
                                "type": "query-select-btn",
                                "query": query,
                                "index": path,
                            },
                            n_clicks=0,
                            style=_unclicked_button_style(),
                            tooltip_text="Apply query",
                        )
                    )

            if n_queries == len(queries):
                queries.append(" No join query suggestions found")

            queries.append(html.Br())
            queries.append(html.B("Example single table queries:"))

            def maybe_to_string(value: Any):
                if isinstance(value, str):
                    return f"'{value}'"
                return value

            default_query = _get_default_sql_query(
                df, [col for col in df_cols if not col.startswith("_")]
            )
            queries.append(html.Br())
            queries.extend(
                get_button_with_tooltip(
                    default_query,
                    id={
                        "type": "query-select-btn",
                        "query": default_query,
                        "index": path,
                    },
                    n_clicks=0,
                    style=_unclicked_button_style(),
                    tooltip_text="Apply query",
                )
            )

            for i, col in enumerate(set(df_cols).difference(ADDED_COLUMNS)):
                try:
                    query = _get_sql_query_with_col(
                        df, col, df_cols, all_cols=i % 2 == 0
                    )
                    queries.append(html.Br())
                    queries.extend(
                        get_button_with_tooltip(
                            query,
                            id={
                                "type": "query-select-btn",
                                "query": query,
                                "index": path,
                            },
                            n_clicks=0,
                            style=_unclicked_button_style(),
                            tooltip_text="Apply query",
                        )
                    )
                except Exception as e:
                    debug_print("failed query", e)

            return (
                self._query_panel_return_modal(path, queries),
                True,
            )

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
                self._deleted_categories = set()
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
            Output("urls", "children"),
            Input("debounced_bounds", "value"),
            State("urls", "children"),
        )
        @time_method_call(_PROFILE_DICT)
        def update_urls(_, urls):
            return (
                [urls[0]]
                + [
                    dbc.Col(
                        [html.A(txt, href=url, target="_blank", id=txt)],
                        width={
                            "size": 2,
                            "order": "last",
                        },
                    )
                    for txt, url in [
                        ("Google earth", get_google_earth_url(self.center)),
                        ("Google maps", get_google_maps_url(self.center)),
                    ]
                ]
                + [
                    dbc.Col(
                        ", ".join(str(round(x, 4)) for x in self.center),
                        width={
                            "size": 3,
                            "order": "last",
                        },
                    )
                ]
            )

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
        @time_method_call(_PROFILE_DICT)
        def update_column_dropdown_options(_):
            if self._concatted_data is None:
                return dash.no_update
            cols_to_drop = (
                ADDED_COLUMNS
                if not DEBUG
                else ADDED_COLUMNS.difference({"_unique_id", "__file_path"})
            )
            columns = (
                {
                    col
                    for dtypes in self._dtypes.values()
                    for col, dtype in dtypes.items()
                    if not dtype.is_nested()
                    and not (col.startswith("__") and col.endswith("__"))
                }
                .difference(cols_to_drop)
                .union(["area"])
            )
            return [{"label": col, "value": col} for col in sorted(columns)]

        @callback(
            Output("numeric-options", "style"),
            Input("is-numeric", "data"),
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
            self._force_categorical = False
            return 0

        @callback(
            Output("dummy-output", "data"),
            Input({"type": "colorpicker", "column_value": dash.ALL}, "value"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        @time_method_call(_PROFILE_DICT)
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
            Output("bins", "data"),
            Output("is-numeric", "data"),
            Output("force-categorical", "children"),
            Output("colors-are-updated", "data"),
            Input("cmap-placeholder", "value"),
            Input("k", "value"),
            Input("force-categorical", "n_clicks"),
            Input("data-was-concatted", "children"),
            State("column-dropdown", "value"),
            State("debounced_bounds", "value"),
            State("bins", "data"),
        )
        @time_method_call(_PROFILE_DICT)
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

            if not self.selected_files:
                self.column = None
                self.color_dict = {}
                self._deleted_categories = set()
                return html.Div(), None, False, None, 1
            elif column != self.column or triggered in ["force-categorical"]:
                self.color_dict = {}
                self._deleted_categories = set()
            elif not column and triggered is None:
                column = self.column
            elif self._concatted_data is None:
                return (
                    [],
                    None,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )

            self.column = column

            default_colors = list(sg.maps.map._CATEGORICAL_CMAP.values())

            debug_print(
                "get_column_value_color_dict, column=",
                column,
                self.column,
                triggered,
                self.color_dict,
            )

            if not column or (
                self._concatted_data is not None and column not in self._concatted_data
            ):
                new_values = [_get_stem(value) for value in self.selected_files]
                if len(set(new_values)) < len(new_values):
                    new_values = [
                        (
                            _get_stem_from_parent(value)
                            if sum(x == name for x in new_values) > 1
                            else _get_stem(value)
                        )
                        for value, name in zip(
                            self.selected_files, new_values, strict=True
                        )
                    ]

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

                for label, path in zip(new_values, self.selected_files, strict=True):
                    name = _get_stem(path)
                    if name in self.color_dict:
                        color_dict[label] = self.color_dict.pop(name)
                    name = _get_stem_from_parent(path)
                    if name in self.color_dict:
                        color_dict[label] = self.color_dict.pop(name)

                self.color_dict = color_dict

                return (
                    _get_colorpicker_container(color_dict),
                    None,
                    False,
                    None,
                    1,
                )

            bounds = self._nested_bounds_to_bounds(bounds)

            values = (
                filter_by_bounds(
                    self._concatted_data.select(column, "minx", "miny", "maxx", "maxy"),
                    bounds,
                )
                .select(column)
                .collect()[column]
            )
            values_no_nans = values.drop_nans().drop_nulls()
            values_no_nans_unique = set(values_no_nans.unique())

            force_categorical_button = _get_force_categorical_button(
                values_no_nans, force_categorical_clicks
            )
            self._force_categorical = (force_categorical_clicks or 0) % 2 == 1
            is_numeric: bool = (
                not self._force_categorical and values_no_nans.dtype.is_numeric()
            )

            if is_numeric and len(values_no_nans):
                if len(values_no_nans_unique) <= k:
                    bins = list(values_no_nans_unique)
                else:
                    bins = jenks_breaks(values_no_nans.to_numpy(), n_classes=k)

                cmap_ = matplotlib.colormaps.get_cmap(cmap)
                colors_ = [
                    matplotlib.colors.to_hex(cmap_(int(i)))
                    for i in np.linspace(0, 255, num=k + 1)
                ]
                rounded_bins = [round(x, 1) for x in bins]
                color_dict = {
                    f"{round(min(values_no_nans), 1)} - {rounded_bins[0]}": colors_[0],
                    **{
                        f"{start} - {stop}": colors_[i + 1]
                        for i, (start, stop) in enumerate(
                            itertools.pairwise(rounded_bins[1:])
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

            self.color_dict = color_dict

            if not is_numeric:
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
            Output("loading", "children", allow_duplicate=True),
            Input("alert", "children"),
            prevent_initial_call=True,
        )
        def update_loading(_):
            if self._concatted_data is None:
                return None
            return "Finished loading"

        @callback(
            Output("lc", "children"),
            Output("alert", "children"),
            Output("max_rows", "children"),
            Output({"type": "wms-list", "index": dash.ALL}, "children"),
            Input("colors-are-updated", "data"),
            Input("dummy-output", "data"),
            Input("is-numeric", "data"),
            Input("wms-checklist", "value"),
            Input("wms-added", "data"),
            Input("max_rows_value", "value"),
            Input("file-control-panel", "children"),
            Input("alpha", "value"),
            Input({"type": "checked-btn", "index": dash.ALL}, "style"),
            Input(
                {"type": "checked-btn-wms", "wms_name": dash.ALL, "tile": dash.ALL},
                "style",
            ),
            State("debounced_bounds", "value"),
            State("column-dropdown", "value"),
            State("bins", "data"),
        )
        @time_method_call(_PROFILE_DICT)
        def add_data(
            currently_in_bounds,
            colorpicker_was_changed,
            is_numeric,
            # wms_items,
            wms_checklist,
            wms_added,
            max_rows_value,
            # data_was_changed,
            order_was_changed,
            alpha,
            checked_clicks,
            checked_wms_clicks,
            bounds,
            column,
            bins,
        ):
            triggered = dash.callback_context.triggered_id
            debug_print(
                "\nadd_data",
                dash.callback_context.triggered_id,
                len(self._loaded_data),
                f"{self.column=}",
            )
            t = perf_counter()

            # if isinstance(triggered, dict) and triggered["type"] == "colorpicker" and

            if max_rows_value is not None:
                self.max_rows = max_rows_value

            bounds = self._nested_bounds_to_bounds(bounds)

            color_dict = self.color_dict

            wms_layers, all_tiles_lists = self._add_wms(wms_checklist, bounds)

            if is_numeric:
                color_dict = {i: color for i, color in enumerate(color_dict.values())}

            if self._concatted_data is None:
                data = [_get_leaflet_overlay(data=None, path="none")]
                rows_are_not_hidden = True
            else:
                n_rows_per_path = dict(
                    self._concatted_data.select("__file_path")
                    .collect()["__file_path"]
                    .value_counts()
                    .iter_rows()
                )
                current_columns = set(self._concatted_data.collect_schema().names())
                add_data_func = partial(
                    _add_data_one_path,
                    max_rows=self.max_rows,
                    concatted_data=self._concatted_data,
                    nan_color=self.nan_color,
                    nan_label=self.nan_label,
                    column=column,
                    is_numeric=is_numeric,
                    color_dict=color_dict,
                    bins=bins,
                    alpha=alpha,
                    n_rows_per_path=n_rows_per_path,
                    columns=self._columns,
                    current_columns=current_columns,
                )
                results = [
                    add_data_func(path)
                    for path, checked in self.selected_files.items()
                    if checked
                ]
                rows_are_not_hidden = not any(x[0] for x in results)
                data = list(
                    itertools.chain.from_iterable([x[1] for x in results if x[1]])
                )

            if rows_are_not_hidden:
                max_rows_component = None
            else:
                max_rows_component = _get_max_rows_displayed_component(self.max_rows)

            return (
                dl.LayersControl(list(self._base_layers.values()) + wms_layers + data),
                None,
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
        @time_method_call(_PROFILE_DICT)
        def update_clicked_features_title(features):
            if not features:
                return dash.no_update
            return (f"Clicked features (n={len(features)})",)

        @callback(
            Output("all-features-title", "children"),
            Input("all-features", "data"),
        )
        @time_method_call(_PROFILE_DICT)
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
            Input("update-table", "data"),
            Input({"type": "geojson", "filename": dash.ALL}, "n_clicks"),
            State({"type": "geojson", "filename": dash.ALL}, "clickData"),
            State({"type": "geojson", "filename": dash.ALL}, "id"),
            State("clicked-features", "data"),
            State("clicked-ids", "data"),
            State("debounced_bounds", "value"),
        )
        @time_method_call(_PROFILE_DICT)
        def display_clicked_feature_attributes(
            clear_table,
            update_table,
            geojson_n_clicks,
            features,
            feature_ids,
            clicked_features,
            clicked_ids,
            bounds,
        ):
            triggered = dash.callback_context.triggered_id
            debug_print("display_clicked_feature_attributes", triggered)
            if triggered == "clear-table-clicked":
                self.selected_features = {}
                return [], [], None
            if (
                triggered is None
                or triggered == "update-table"
                or (
                    (self.selected_features and not features)
                    or all(x is None for x in features)
                )
            ):
                clicked_ids = list(self.selected_features)
                clicked_features = list(self.selected_features.values())
                return clicked_features, clicked_ids, None

            if not features or all(x is None for x in features):
                return dash.no_update, dash.no_update, None

            if triggered == "update-table":
                # get path of table displayed
                clicked_path = self._current_table_view
                if clicked_path is None:
                    return dash.no_update, dash.no_update, None
            else:
                clicked_path = triggered["filename"]

            index = next(
                iter(
                    i
                    for i, id_ in enumerate(feature_ids)
                    if id_["filename"] == clicked_path
                )
            )
            feature = features[index]
            if feature is None:
                # feature is None probably because of zoom/panning quickly after clicking feature
                return dash.no_update, dash.no_update, None
            unique_id = feature["properties"]["_unique_id"]
            i = int(float(unique_id))
            try:
                path = list(self._loaded_data)[i]
            except IndexError as e:
                raise type(e)(f"{e}: {i=}, {self._loaded_data.keys()=}")
            bounds = self._nested_bounds_to_bounds(bounds)
            feature, geometry = self._get_selected_feature(
                unique_id, path, bounds=bounds
            )
            if self.hard_click:
                geom = shapely.from_wkb(geometry)

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
                    filter_by_bounds(self._concatted_data, geom.bounds)
                    .filter(
                        pl.col("geometry").map_elements(
                            geoms_relate, return_dtype=pl.Boolean
                        )
                    )
                    .drop(
                        *ADDED_COLUMNS.difference({"_unique_id"}).union(
                            {"split_index"}
                        ),
                        strict=False,
                    )
                )
                if DEBUG:
                    intersecting = intersecting.with_columns(
                        pl.col("_unique_id").alias("id")
                    )
                else:
                    intersecting = intersecting.rename({"_unique_id": "id"})

                intersecting = intersecting.collect()
                all_null_cols = [
                    col
                    for col in intersecting.columns
                    if intersecting[col].is_null().all()
                ]
                properties = intersecting.drop(*all_null_cols).to_dicts()
            else:
                properties = [{key: value for key, value in feature.items()}]
            for props in properties:
                if props["id"] not in clicked_ids:
                    clicked_features.append(props)
            clicked_ids = [x["id"] for x in clicked_features]
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
            Output("bottom-alert", "children"),
            Input("clear-table", "n_clicks"),
            Input("update-table", "data"),
            Input({"type": "table-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "table-btn", "index": dash.ALL}, "id"),
        )
        @time_method_call(_PROFILE_DICT)
        def display_all_feature_attributes(
            clear_table, update_table, table_btn_n_clicks, table_btn_ids
        ):
            triggered = dash.callback_context.triggered_id
            debug_print("display_all_feature_attributes", triggered)
            if triggered == "clear-table" or not self.selected_files:
                self._current_table_view = None
                return [], None
            if triggered is None:
                return dash.no_update, None

            if triggered == "update-table":
                # get path of table displayed
                clicked_path = self._current_table_view
            else:
                clicked_path = get_index_if_clicks(table_btn_n_clicks, table_btn_ids)
            if clicked_path is None:
                return dash.no_update, None

            self._current_table_view = clicked_path

            out_alert = dbc.Alert(
                f"No rows in '{self._get_unique_stem(clicked_path)}' after queries.",
                color="info",
                dismissable=True,
                duration=10_000,
            )

            df, _ = self._concat_data(bounds=None, paths=[clicked_path])
            if df is not None:
                df = df.drop(ADDED_COLUMNS.difference({"_unique_id"})).collect()
            if df is None or not len(df):
                # read data out of bounds to get table
                _read_files(
                    self,
                    [
                        x
                        for x in self._bounds_series[
                            lambda x: x.index.str.contains(clicked_path)
                        ].index
                        if x not in self._loaded_data
                    ],
                )
                df, _ = self._concat_data(bounds=None, paths=[clicked_path])
                if df is None:
                    self._current_table_view = None
                    return None, out_alert
                df = df.drop(ADDED_COLUMNS.difference({"_unique_id"})).collect()

            if not len(df) or not len(df.columns):
                self._current_table_view = None
                return None, out_alert
            if DEBUG:
                df = df.with_columns(pl.col("_unique_id").alias("id"))
            else:
                df = df.rename({"_unique_id": "id"})

            if (
                len(df) * len(df.columns) > 5_000_000
                and " limit " not in self._queries.get(clicked_path, "").lower()
            ):
                cols = set(df.columns).difference(ADDED_COLUMNS).union({"area"})
                query = _get_default_sql_query(df, cols)
                for col in reversed(sorted(cols)):
                    try:
                        query = _get_sql_query_with_col(
                            df.lazy(), col, cols, all_cols=False
                        )
                        break
                    except Exception as e:
                        debug_print("failed query", e)

                alert = dbc.Alert(
                    html.Div(
                        [
                            f"Cannot display table of shape {len(df), len(df.columns)}. "
                            f"Consider using an SQL query, e.g.",
                            html.Br(),
                            html.B(query),
                        ]
                    ),
                    color="warning",
                    dismissable=True,
                )
                return None, alert

            clicked_features = df.to_dicts()
            return clicked_features, None

        @callback(
            Output("feature-table-rows", "columns"),
            Output("feature-table-rows", "data"),
            Output("feature-table-rows", "style_table"),
            Output("feature-table-rows", "hidden_columns"),
            Input("all-features", "data"),
            State("feature-table-rows", "style_table"),
        )
        @time_method_call(_PROFILE_DICT)
        def update_table(data, style_table):
            return self._update_table(data, style_table=style_table)

        @callback(
            Output("feature-table-rows-clicked", "columns"),
            Output("feature-table-rows-clicked", "data"),
            Output("feature-table-rows-clicked", "style_table"),
            Output("feature-table-rows-clicked", "hidden_columns"),
            Input("clicked-features", "data"),
            State("feature-table-rows-clicked", "style_table"),
        )
        @time_method_call(_PROFILE_DICT)
        def update_table_clicked(data, style_table):
            return self._update_table(data, style_table)

        @callback(
            Output("map-bounds", "data"),
            Output("map-zoom", "data"),
            Output("map-center", "data"),
            Input("feature-table-rows", "active_cell"),
            Input("feature-table-rows-clicked", "active_cell"),
            State("viewport-container", "data"),
            prevent_initial_call=True,
        )
        @time_method_call(_PROFILE_DICT)
        def zoom_to_feature(active: dict, active_clicked: dict, viewport):
            triggered = dash.callback_context.triggered_id
            if triggered == "feature-table-rows":
                if active is None:
                    return dash.no_update, dash.no_update, dash.no_update
                unique_id = active["row_id"]
            else:
                if active_clicked is None:
                    return dash.no_update, dash.no_update, dash.no_update
                unique_id = active_clicked["row_id"]

            i = int(float(unique_id))

            try:
                df = list(self._loaded_data.values())[i]
            except IndexError as e:
                raise IndexError(f"{e} for {i=} and {self._loaded_data=}")
            matches = (
                df.filter(pl.col("_unique_id") == unique_id)
                .select("minx", "miny", "maxx", "maxy")
                .collect()
            )
            if not len(matches) or any(pd.isna(x) for x in matches.row(0)):
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
        @time_method_call(_PROFILE_DICT)
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

    @time_method_call(_PROFILE_DICT)
    def _update_table(self, data, style_table):
        if not data:
            return None, None, style_table | {"height": "1vh"}, None
        columns_union = set()
        for x in data:
            columns_union |= set(x)
        columns = [{"name": k, "id": k} for k in columns_union]
        height = min(40, len(data) * 5 + 5)
        return (
            columns,
            data,
            style_table | {"height": f"{height}vh"},
            ["id"],
        )

    @time_method_call(_PROFILE_DICT)
    def _delete_file(self, n_clicks_list, delete_ids, delete_category: bool):
        path_to_delete = get_index_if_clicks(n_clicks_list, delete_ids)
        debug_print("_delete_file", locals())
        if path_to_delete is None:
            return dash.no_update, dash.no_update
        deleted_files = set()
        for path in dict(self.selected_files):
            name = self._get_unique_stem(path) if delete_category else path
            if path_to_delete == name:
                self.selected_files.pop(path)
                deleted_files.add(path)
                break

        assert len(deleted_files) == 1, deleted_files
        deleted_files2 = set()
        for i, path2 in enumerate(list(self._loaded_data)):
            parts = Path(path2).parts
            if not all(part in parts for part in Path(path).parts):
                continue
            for idx in list(self.selected_features):
                if int(float(idx)) == i:
                    self.selected_features.pop(idx)
            del self._loaded_data[path2]
            deleted_files2.add(path2)

        debug_print(f"{deleted_files2=}")
        self._bounds_series = self._bounds_series[
            lambda x: ~x.index.isin(deleted_files2)
        ]

        return None, None

    @time_method_call(_PROFILE_DICT)
    def _nested_bounds_to_bounds(
        self,
        bounds: list[list[float]],
    ) -> tuple[float, float, float, float]:
        if bounds is None and self._bounds is None:
            return (
                sg.to_gdf(reversed(self.center), 4326)
                .to_crs(3035)
                .buffer(165_000 / (self.zoom**1.25))
                .to_crs(4326)
                .total_bounds
            )
        elif self._bounds is not None:
            bounds = self._bounds
        if isinstance(bounds, str):
            bounds = json.loads(bounds)
        mins, maxs = bounds
        miny, minx = mins
        maxy, maxx = maxs
        return minx, miny, maxx, maxy

    @time_method_call(_PROFILE_DICT)
    def _get_selected_feature(
        self, unique_id: float, path: str, bounds: tuple[float], recurse: bool = True
    ) -> tuple[dict[str, str | Number], Geometry]:
        df, _ = self._concat_data(bounds=bounds, paths=[path])
        row = df.filter(pl.col("_unique_id") == unique_id)
        geometries = row.select("geometry").collect()["geometry"]
        if not len(geometries):
            time.sleep(0.1)
            return self._get_selected_feature(
                unique_id, path, bounds=None, recurse=False
            )
        geometry = next(iter(geometries))

        if DEBUG:
            row = row.with_columns(pl.col("_unique_id").alias("id"))
        else:
            row = row.rename({"_unique_id": "id"})

        row = row.drop(
            *ADDED_COLUMNS.difference({"_unique_id"}).union({"split_index"}),
            strict=False,
        ).collect()

        if not len(row) and recurse:
            time.sleep(0.1)
            return self._get_selected_feature(
                unique_id, path, bounds=None, recurse=False
            )
        assert len(row) == 1, (
            unique_id,
            f"{recurse=}",
            row,
            row["id"],
            df.collect()["_unique_id"],
        )
        return row.row(0, named=True), geometry

    @time_method_call(_PROFILE_DICT)
    def _check_for_circular_queries(self, query, path):
        if query is None:
            return
        n = 0
        for path2, query2 in self._queries.items():
            if (
                query2 is None
                or path == path2
                or " join " not in query
                or " join " not in query2
            ):
                continue

            stripped_query = query.replace(")", "").replace("(", "")
            stripped_query2 = query2.replace(")", "").replace("(", "")
            if (
                f" {_get_stem(path)} " in stripped_query2
                or f" {_get_stem_from_parent(path)} " in stripped_query2
            ):
                n += 1

            if (
                f" {_get_stem(path2)} " in stripped_query
                or f" {_get_stem_from_parent(path2)} " in stripped_query
            ):
                n += 1
        if n >= 2:
            raise RecursionError(
                f"Recursion error: Circular joins on {path} and {path}",
            )

    @time_method_call(_PROFILE_DICT)
    def _update_query(self, queries: list[str | None], ids):
        out_alerts = []
        for path in self.selected_files:
            try:
                query = get_index(queries, ids, path)
                if query == self._queries.get(path):
                    continue
                self._check_for_circular_queries(query, path)
                self._queries[path] = query
            except RecursionError as e:
                out_alerts.append(
                    dbc.Alert(
                        str(e),
                        color="warning",
                        dismissable=True,
                    )
                )
                break
            except ValueError:
                pass
        if out_alerts:
            return out_alerts
        return None

    @time_method_call(_PROFILE_DICT)
    def _get_unique_stem(self, path) -> str:
        name = _get_stem(path)
        if sum(_get_stem(x) == name for x in self.selected_files) > 1:
            name = _get_stem_from_parent(path)
        return name

    @property
    def _columns(self) -> dict[str, set[str]]:
        return {path: set(dtypes) for path, dtypes in self._dtypes.items()}

    def _has_column(self, path: str, column: str) -> bool:
        return bool(
            {
                True
                for key, dtypes in self._dtypes.items()
                for col in dtypes
                if path in key and col == column
            }
        )

    def _get_dtype(self, path: str, column: str) -> pl.DataType:
        relevant_dtypes: set[pl.DataType] = {
            dtype
            for key, dtypes in self._dtypes.items()
            for col, dtype in dtypes.items()
            if path in key and col == column
        }
        if not relevant_dtypes:
            raise ValueError(f"No column '{column}' in {path}")
        if len(relevant_dtypes) > 1:
            raise ValueError(f"Multiple dtypes for '{column}': {relevant_dtypes}")
        return next(iter(relevant_dtypes))

    def _query_panel_return_modal(self, path, queries):
        queries = queries or []
        example_button_text = "Get query examples"
        if (
            queries
            and sum(
                size for key, size in self._loaded_data_sizes.items() if path in key
            )
            > 100_000_000
        ):
            info_about_overhead = [
                html.Div(
                    [
                        "Note that for large datasets like this, queries with polars/SQL might be significantly faster than pandas and especially geopandas.",
                        html.Br(),
                        f"Also note that clicking the '{example_button_text}' button might be slow and memory consuming.",
                    ],
                )
            ]
        else:
            info_about_overhead = []
        return [
            dbc.ModalHeader(
                dbc.ModalTitle(f"Query {self._get_unique_stem(path)}"),
                close_button=True,
            ),
            dbc.ModalBody(
                html.Div(
                    [
                        dcc.Textarea(
                            value=self._queries.get(path, None),
                            placeholder="Write query here...",
                            id={
                                "type": "query-copy",
                                "index": path,
                            },
                            style={"width": "100%", "height": "20vh"},
                            autoFocus="autoFocus",
                        ),
                        *info_about_overhead,
                        *get_button_with_tooltip(
                            example_button_text,
                            id={
                                "type": "query-examples-button",
                                "index": path,
                            },
                            tooltip_text="Note that this might be slow and memory heavy for large datasets",
                        ),
                        *queries,
                    ],
                ),
            ),
        ]

    @time_method_call(_PROFILE_DICT)
    def _concat_data(
        self,
        bounds,
        paths: list[str] | None = None,
        _filter: bool = True,
    ) -> tuple[pl.LazyFrame | None, list[dbc.Alert] | None]:
        dfs = {}
        alerts = set()
        for path in self.selected_files:
            path_parts = Path(path).parts
            for key in self._loaded_data:
                if paths and (path not in paths and key not in paths) or key in dfs:
                    continue
                key_parts = Path(key).parts
                if not all(part in key_parts for part in path_parts):
                    continue
                df = self._loaded_data[key]
                if bounds is not None:
                    df = filter_by_bounds(df, bounds)
                if (
                    self._deleted_categories
                    and self.column
                    and not self._force_categorical
                    and self._has_column(key, self.column)
                    and self._get_dtype(key, self.column).is_numeric()
                ):
                    try:
                        error_mess = "Cannot remove categories from numeric columns. Use an SQL query instead"
                        # make sure we only give one warning
                        assert not any(
                            x.children == error_mess for x in alerts if x is not None
                        )
                        alerts.add(
                            dbc.Alert(
                                error_mess,
                                color="warning",
                                dismissable=True,
                                duration=5_000,
                            )
                        )
                    except AssertionError:
                        pass
                elif self._deleted_categories and self.column in df:
                    try:
                        expression = (
                            pl.col(self.column).is_in(list(self._deleted_categories))
                            == False
                        )
                    except Exception as e:
                        raise type(e)(
                            f"{e}. {self.column=}, {self._deleted_categories=}"
                        )
                    if self.nan_label in self._deleted_categories:
                        expression &= pl.col(self.column).is_not_null()
                    df = df.filter(expression)
                elif (
                    self.nan_label in self._deleted_categories and self.column not in df
                ):
                    if self.splitted:
                        df = get_split_index(df)
                    continue
                if _filter and self._queries.get(path, None) is not None:
                    df, alert = self._filter_data(df, self._queries[path], key)
                    alerts.add(alert)

                if self.splitted:
                    df = get_split_index(df)

                dfs[key] = df

        if dfs:
            df = pl.concat(list(dfs.values()), how="diagonal_relaxed")
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

    @time_method_call(_PROFILE_DICT)
    def _filter_data(
        self, df: pl.DataFrame, query: str | None, path: str
    ) -> pl.DataFrame:
        query = query.strip()

        if query is None or (isinstance(query, str) and query == ""):
            return df, None

        alert = None

        # try to filter with polars, then pandas.loc, then pandas.query
        # no need for pretty code and specific exception handling here, as this a convenience feature
        if _is_sql(query):
            try:
                return self._run_polars_sql(df, query), None
            except Exception as e:
                return df, (
                    f"Query function failed with polars sql ({type(e).__name__}: {e}) "
                )
        elif _is_polars_expression(query):
            try:
                return (
                    df.filter(eval(query) if isinstance(query, str) else query),
                    None,
                )
            except Exception as e:
                return df, (
                    f"Query function failed with polars filter ({type(e).__name__}: {e}. query: {query}) "
                )

        # df_with_added_cols = df.select(ADDED_COLUMNS.difference({"geometry"}))
        df2 = df.drop(ADDED_COLUMNS.difference({"geometry", "_unique_id"})).collect()
        df2, alert = self._run_df_function(df2, df, query, path)
        if df2 is None:
            return df, alert
        if "_unique_id" not in df2:
            return df, (
                "Cannot drop internal column '_unique_id' from df. This is added to keep track of data"
            )
        return df2, alert

    @time_method_call(_PROFILE_DICT)
    def _run_df_function(
        self, df: pl.DataFrame, df_orig: pl.LazyFrame, query: str | Callable, path: str
    ):
        error_mess = ""

        try:
            called = eval(query)
        except Exception as e:
            debug_print(f"eval(query) failed: {e}")
            called = query
        if callable(called):
            try:
                called = called(df)
            except Exception as e:
                debug_print(f"query(df) failed: {e}")
        try:
            return self._callable_results_to_polars(df_orig, called, path), None
        except Exception as e:
            error_mess += (
                f"Query function failed with polars ({type(e).__name__}: {e}) "
            )

        try:
            return (
                self._callable_results_to_polars(
                    df_orig, df.to_pandas().query(query), path
                ),
                None,
            )
        except Exception as e:
            debug_print(f"pandas query failed: {e}")
            query_error = f"Query function failed with pandas.DataFrame.query: ({type(e).__name__}: '{e}' for query: '{query}'"

        is_likely_geopandas = _is_likely_geopandas_func(df, query)
        if is_likely_geopandas:
            df = _polars_to_gdf(df)
        else:
            df = df.to_pandas()
        added_to_globals = set()
        for path_ in self.selected_files:
            name = self._get_unique_stem(path_)
            if (
                name in ["df", "self"]
                or not isinstance(query, str)
                or name not in query
                or name in globals()
            ):
                continue
            df2, _ = self._concat_data(
                bounds=self._nested_bounds_to_bounds(self._bounds), paths=[path_]
            )
            df2 = (
                _polars_to_gdf(
                    df2.drop(*ADDED_COLUMNS.difference({"geometry"})).collect()
                )
                if is_likely_geopandas
                else df2.drop(*ADDED_COLUMNS).collect().to_pandas()
            )
            locals()[name] = df2
            globals()[name] = df2
            added_to_globals.add(name)

        if isinstance(query, str):
            try:
                query = eval(query)
            except NameError:
                error_mess += query_error
            except Exception as e:
                error_mess += (
                    f"Query function failed with pandas ({type(e).__name__}: {e}) "
                )
                for name in added_to_globals:
                    globals().pop(name)
                return None, error_mess

        for name in added_to_globals:
            globals().pop(name)

            # if is_likely_geopandas:
            #     return None, error_mess
            # try:
            #     called = query(df.to_pandas())
            #     return self._callable_results_to_polars(df_orig, called, path), None
            # except Exception as e:

        try:
            if callable(query):
                query = query(df)
            return self._callable_results_to_polars(df_orig, query, path), None
        except Exception as e:
            error_mess += (
                f"Query function failed with pandas ({type(e).__name__}: {e}) "
            )

        return None, error_mess

    @time_method_call(_PROFILE_DICT)
    def _callable_results_to_polars(
        self, df: pl.LazyFrame, called: Any, path: str
    ) -> pl.LazyFrame:
        debug_print("_callable_results_to_polars", type(called), path)
        if isinstance(called, pl.LazyFrame):
            return called
        if isinstance(called, pl.DataFrame):
            return called.lazy()
        if isinstance(called, GeoDataFrame):
            called, _ = _geopandas_to_polars(called, path)
            called = called.with_columns(
                _unique_id=_get_unique_id(list(self._loaded_data).index(path))
            ).lazy()
            return called
        if isinstance(called, GeoSeries):
            geometries, areas, bounds = _get_area_and_bounds(geometries=called.values)
            called = _add_columns(
                df.drop(ADDED_COLUMNS),
                geometries,
                areas,
                bounds,
                path,
            ).with_columns(
                _unique_id=_get_unique_id(list(self._loaded_data).index(path))
            )
            return called
        if isinstance(called, pd.DataFrame):
            return df.filter(pl.col("_unique_id").is_in(called["_unique_id"].values))
        if pd.api.types.is_list_like(called):
            return df.filter(np.array(called))
        if isinstance(called, pl.Expr):
            return df.filter(called)
        if callable(called):
            raise ValueError(f"Could't call function: {called}")
        raise ValueError(f"Didn't understand return value {type(called)}: {called}")

    @time_method_call(_PROFILE_DICT)
    def _run_polars_sql(self, df: pl.LazyFrame, query):
        formatted_query = _add_cols_to_sql_query(query.replace('"', "'"))
        if " join " in formatted_query.lower():
            return self._polars_sql_join(df, formatted_query)
        if " df " in query:
            return pl.sql(formatted_query, eager=False)
        return df.sql(formatted_query)

    @time_method_call(_PROFILE_DICT)
    def _polars_sql_join(self, df, query):
        if " df " not in query and "df\n" not in query and "\ndf" not in query:
            raise ValueError(
                f"Table to be queried must be referenced to as 'df'. {query}"
            )
        join_df_name: str = (
            query.replace(" JOIN ", " join ").split(" join ")[-1].split()[0]
        )
        if join_df_name in self.selected_files:
            path = join_df_name
        elif sum(join_df_name == _get_stem(x) for x in self.selected_files) == 1:
            path = next(
                iter(x for x in self.selected_files if join_df_name == _get_stem(x))
            )
        elif (
            sum(join_df_name == _get_stem_from_parent(x) for x in self.selected_files)
            == 1
        ):
            path = next(
                iter(
                    x
                    for x in self.selected_files
                    if join_df_name == _get_stem_from_parent(x)
                )
            )
        else:
            try:
                example_stem = Path(
                    next(
                        iter(
                            {
                                x
                                for x in self.selected_files
                                if len(list(Path(x).parts)) > 1
                            }
                        )
                    )
                ).stem
            except StopIteration:
                example_stem = "ABAS_kommune_flate_p2025_v1"
            raise ValueError(
                f"Join data must be referenced to by the file's stem without quotation marks, e.g. {example_stem}. "
                "If multiple tables have same stem, the parent directory must be included as well.",
            )
        join_df, _ = self._concat_data(bounds=None, paths=[path])
        if join_df is None:
            if self._queries.get(path) and len(
                self._concat_data(bounds=None, paths=[path], _filter=False)[0]
            ):
                query_tip = f" after query: {self._queries[path]}"
            else:
                query_tip = ""
            this_name = self._get_unique_stem(next(iter(df["__file_path"])))
            raise NoRowsError(
                f"SQL query error for {this_name}: No rows in join table {self._get_unique_stem(path)}{query_tip}"
            )
        # using literal 'join_df' in case 'i' is negative index
        query = query.replace(join_df_name, "join_df")
        ctx = pl.SQLContext(**{"df": df, "join_df": join_df})
        return ctx.execute(query, eager=False)

    @time_method_call(_PROFILE_DICT)
    def _map_constructor(
        self, data: dl.LayersControl, preferCanvas=True, zoomAnimation=False, **kwargs
    ) -> dl.Map:
        return dl.Map(
            center=self.center,
            bounds=self._bounds,
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

    @time_method_call(_PROFILE_DICT)
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

    @time_method_call(_PROFILE_DICT)
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

    @time_method_call(_PROFILE_DICT)
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

    @time_method_call(_PROFILE_DICT)
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

    def _append_to_bounds_series(self, paths) -> None:
        try:
            child_paths = _get_child_paths(paths, self.file_system)
            paths_with_meta, paths_without_meta = (
                self._get_paths_with_and_without_metadata(child_paths)
            )
            more_bounds = _get_bounds_series_as_4326(
                paths_with_meta,
                file_system=self.file_system,
            )
        except Exception:
            # reload file system to avoid cached reading of old files that don't exist any more
            self.file_system = self.file_system.__class__()
            child_paths = _get_child_paths(paths, self.file_system)
            paths_with_meta, paths_without_meta = (
                self._get_paths_with_and_without_metadata(child_paths)
            )
            more_bounds = _get_bounds_series_as_4326(
                paths_with_meta, file_system=self.file_system
            )
        self._bounds_series = pd.concat(
            [
                self._bounds_series,
                more_bounds,
                pd.Series(
                    [None for _ in range(len(paths_without_meta))],
                    index=paths_without_meta,
                ),
            ]
        )[lambda x: ~x.index.duplicated()]

    def _get_paths_with_and_without_metadata(self, paths):
        filt = [
            any(path.endswith(f".{txt}") for txt in self._file_formats_with_metadata)
            for path in paths
        ]
        return [path for path in paths if filt], [path for path in paths if not filt]

    def _get_self_as_dict(self) -> dict[str, Any]:
        data = {
            key: value
            for key, value in self.__dict__.items()
            if key
            not in [
                "app",
                "bounds",
                "logger",
            ]
            and not key.startswith("_")
            and not (isinstance(value, (dict, list, tuple)) and not value)
        }

        if self.selected_files:
            data = {
                "data": {
                    key: _unformat_query(self._queries.get(key, "")) or None
                    for key in reversed(data.pop("selected_files", []))
                },
                **data,
            }
        else:
            data.pop("selected_files", [])

        if self._file_browser.favorites:
            data["favorites"] = self._file_browser.favorites

        data["file_system"] = data["file_system"].__class__.__name__ + "()"

        if "selected_features" in data:
            data["selected_features"] = list(self.selected_features)
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

    def __getstate__(self):
        for variable_name, value in vars(self).items():
            try:
                pickle.dumps(value)
            except pickle.PicklingError:
                print(f"{variable_name} with value {value} is not pickable")


@time_function_call(_PROFILE_DICT)
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


@time_function_call(_PROFILE_DICT)
def _change_order(explorer, n_clicks_list, ids, buttons, what: str):
    if what not in ["up", "down"]:
        raise ValueError(what)
    path = get_index_if_clicks(n_clicks_list, ids)
    if path is None or not buttons:
        return dash.no_update
    i = list(reversed(explorer.selected_files)).index(path)
    if (what == "up" and i == 0) or (what == "down" and i == len(buttons) - 1):
        return dash.no_update
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
    return buttons


def _named_color_to_hex(color: str) -> str:
    return mcolors.to_hex(color)


@time_function_call(_PROFILE_DICT)
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


@time_function_call(_PROFILE_DICT)
def _add_data_one_path(
    path,
    column,
    is_numeric,
    color_dict,
    bins,
    max_rows,
    concatted_data,
    nan_color,
    nan_label,
    alpha,
    n_rows_per_path,
    columns: dict[str, set[str]],
    current_columns: set[str],
):
    columns: set[str] = {
        col
        for key, cols in columns.items()
        for col in cols
        if path in key and col in current_columns
    } | {"split_index"}

    df = concatted_data.filter(pl.col("__file_path").str.contains(path)).select(
        "geometry", "_unique_id", *((column,) if column and column in columns else ())
    )
    n_rows = sum(count for key, count in n_rows_per_path.items() if path in key)
    if not n_rows:
        return (
            False,
            [_get_leaflet_overlay(data=None, path=path)],
        )

    rows_are_hidden = n_rows > max_rows
    if rows_are_hidden:
        indices = np.random.choice(n_rows, size=max_rows, replace=False)
        df = df.filter(pl.int_range(pl.len()).is_in(indices))

    if column is not None and column in columns:
        df = _fix_colors(df, column, bins, is_numeric, color_dict, nan_color, nan_label)

    if column and column not in columns:
        return rows_are_hidden, [
            _get_leaflet_overlay(
                data=_cheap_geo_interface(df.collect()),
                path=path,
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
            )
        ]
    elif column:
        return rows_are_hidden, [
            _get_multiple_leaflet_overlay(
                df,
                path,
                column,
                nan_color,
                alpha,
                onEachFeature=ns("yellowIfHighlighted"),
                pointToLayer=ns("pointToLayerCircle"),
                hideout=dict(
                    circleOptions=dict(fillOpacity=1, stroke=False, radius=5),
                ),
            )
        ]
    else:
        # no column
        try:
            color = color_dict[_get_stem(path)]
        except KeyError:
            color = color_dict[_get_stem_from_parent(path)]
        return rows_are_hidden, [
            _get_leaflet_overlay(
                data=_cheap_geo_interface(df.collect()),
                path=path,
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
            )
        ]


@time_function_call(_PROFILE_DICT)
def _fix_colors(df, column, bins, is_numeric, color_dict, nan_color, nan_label):
    if not is_numeric:
        return df.with_columns(
            _color=pl.col(column).replace(
                {
                    value: color
                    for value, color in color_dict.items()
                    if value != nan_label
                },
                default=pl.lit(nan_color),
                return_dtype=pl.String(),
            )
        )
    elif bins is None:
        return df.with_columns(_color=pl.lit(nan_color))

    notnas = df.filter((pl.col(column).is_not_null()) & (pl.col(column).is_not_nan()))
    if bins is not None and len(bins) == 1:
        notnas = notnas.with_columns(
            _color=pl.lit(
                next(iter(color for color in color_dict.values() if color != nan_color))
            )
        )
    else:
        conditions = [
            (pl.col(column) < bins[1]) & (pl.col(column).is_not_null()),
            *[
                (pl.col(column) >= bins[i])
                & (pl.col(column) < bins[i + 1])
                & (pl.col(column).is_not_null())
                for i in range(1, len(bins) - 1)
            ],
            (pl.col(column) >= bins[-1]) & (pl.col(column).is_not_null()),
        ]
        bin_index_expr = pl.when(conditions[0]).then(pl.lit(color_dict[0]))
        for i, cond in enumerate(conditions[1:], start=1):
            bin_index_expr = bin_index_expr.when(cond).then(pl.lit(color_dict[i]))
        notnas = notnas.with_columns(bin_index_expr.alias("_color"))

    return pl.concat(
        [notnas, df.filter((pl.col(column).is_null()) | (pl.col(column).is_nan()))],
        how="diagonal_relaxed",
    )


def polars_isna(df):
    try:
        return (df.is_nan()) | (df.is_null())
    except pl.exceptions.InvalidOperationError:
        return df.is_null()


@time_function_call(_PROFILE_DICT)
def filter_by_bounds(df: pl.LazyFrame, bounds: tuple[float]) -> pl.LazyFrame:
    minx, miny, maxx, maxy = bounds

    df = df.filter(
        (
            (pl.col("minx") <= float(maxx))
            & (pl.col("maxx") >= float(minx))
            & (pl.col("miny") <= float(maxy))
            & (pl.col("maxy") >= float(miny))
        )
        | (pl.col("minx").is_null())
    )
    return df


@time_function_call(_PROFILE_DICT)
def read_nrows(file, nrow: int, nth_batch: int, file_system) -> pyarrow.Table:
    """Read first n rows of a parquet file."""
    for _, batch in zip(
        range(nth_batch + 1),
        pq.ParquetFile(file, filesystem=file_system).iter_batches(nrow),
        strict=False,
    ):
        pass
    return pyarrow.Table.from_batches([batch])


@time_function_call(_PROFILE_DICT)
def _read_polars(path, file_system, primary_column, **kwargs):
    with file_system.open(path, "rb") as file:
        return pl.scan_parquet(
            file,
            schema={primary_column: pl.Binary()},
            missing_columns="insert",
            **kwargs,
        )


@time_function_call(_PROFILE_DICT)
def _pyarrow_to_polars(
    table: pyarrow.Table,
    path: str,
    file_system: AbstractFileSystem,
    pandas_fallback: bool = True,
) -> tuple[pl.LazyFrame, dict[str, pl.DataType]]:
    """Convert pyarrow.Table with geo-metadata to polars.LazyFrame.

    The geometry column must have no metadata for it to be accepted by polars.

    Turning the frame lazy might have no performance benefit. Ideally should
    use polars.scan_parquet, but
    """
    metadata = _get_geo_metadata(path, file_system)
    primary_column = metadata["primary_column"]
    try:
        table = table.cast(
            pyarrow.schema(
                [
                    *[
                        (
                            (col, table.schema.field(col).type)
                            if col != primary_column
                            else (col, pyarrow.binary())
                        )
                        for col in table.schema.names
                    ],
                ]
            )
        )
        df = pl.from_arrow(table, schema_overrides={primary_column: pl.Binary()})
    except Exception as e:
        if DEBUG or not pandas_fallback:
            raise e
        df = pl.from_pandas(table.to_pandas())
    dtypes = dict(zip(df.columns, df.dtypes, strict=False))
    return _prepare_df(df.lazy(), path, metadata), dtypes


@time_function_call(_PROFILE_DICT)
def _get_area_and_bounds(geometries: GeometryArray | GeoSeries):
    areas = shapely.area(geometries)
    if np.median(areas) > 10:
        # as int because easier to read
        areas = areas.astype(np.int64)
    if geometries.crs is not None:
        geometries = geometries.to_crs(4326)
    return geometries, areas, shapely.bounds(geometries)


@time_function_call(_PROFILE_DICT)
def _prepare_df(df: pl.LazyFrame, path, metadata) -> pl.LazyFrame:
    primary_column = metadata["primary_column"]
    geo_metadata = metadata["columns"][primary_column]
    crs = geo_metadata["crs"]
    geometries, areas, bounds = _get_area_and_bounds(
        geometries=GeometryArray(
            shapely.from_wkb(df.select(primary_column).collect()[primary_column]),
            crs=crs,
        )
    )
    df = df.drop(primary_column)
    df = _add_columns(df, geometries, areas, bounds, path)
    # collecting, then back to lazy to only do these calculations once
    return df.collect().lazy()


@time_function_call(_PROFILE_DICT)
def _geopandas_to_polars(df: GeoDataFrame, path) -> pl.DataFrame:
    geometries, areas, bounds = _get_area_and_bounds(geometries=df.geometry.values)
    df = df.drop(columns=df.geometry.name)
    df = pl.from_pandas(df)
    dtypes = dict(zip(df.columns, df.dtypes, strict=False))
    df = _add_columns(df, geometries, areas, bounds, path)
    return df, dtypes


@time_function_call(_PROFILE_DICT)
def _pandas_to_polars(df: pd.DataFrame, path) -> pl.DataFrame:
    df = pl.from_pandas(df)
    dtypes = dict(zip(df.columns, df.dtypes, strict=False))
    df = df.with_columns(
        area=pl.lit(None).cast(pl.Float64()),
        geometry=pl.lit(None).cast(pl.Binary()),
        minx=pl.lit(None).cast(pl.Float64()),
        miny=pl.lit(None).cast(pl.Float64()),
        maxx=pl.lit(None).cast(pl.Float64()),
        maxy=pl.lit(None).cast(pl.Float64()),
        __file_path=pl.lit(path),
    )
    return df, dtypes


def _add_columns(df, geometries, areas, bounds, path):
    return df.with_columns(
        pl.Series("area", areas),
        pl.Series("geometry", shapely.to_wkb(geometries), dtype=pl.Binary()),
        pl.Series("minx", bounds[:, 0]),
        pl.Series("miny", bounds[:, 1]),
        pl.Series("maxx", bounds[:, 2]),
        pl.Series("maxy", bounds[:, 3]),
        __file_path=pl.lit(path),
    )


@time_function_call(_PROFILE_DICT)
def _polars_to_gdf(df: pl.LazyFrame) -> GeoDataFrame:
    return GeoDataFrame(
        df.drop("geometry").to_pandas(),
        geometry=shapely.from_wkb(df["geometry"].to_pandas()),
        crs=4326,
    ).to_crs(3035)


def _get_unique_id(i: float) -> pl.Expr:
    """Lazy float column: 0.0, 0.01, ..., N / divider + i."""
    return pl.lit(f"{i}.") + (pl.int_range(pl.len(), eager=False)).cast(pl.Utf8)


@time_function_call(_PROFILE_DICT)
def _read_files(explorer, paths: list[str], mask=None, **kwargs) -> None:
    if not paths:
        return
    paths = [
        path
        for path in paths
        if mask is None or shapely.intersects(mask, explorer._bounds_series[path])
    ]
    if not paths:
        return
    # loky because to_crs is slow with threading
    backend = "threading" if len(paths) <= 3 else "loky"
    file_system = explorer.file_system
    with joblib.Parallel(len(paths), backend=backend) as parallel:
        more_data = parallel(
            joblib.delayed(explorer.__class__.read_func)(
                path=path, file_system=file_system, **kwargs
            )
            for path in paths
        )
    for path, (df, dtypes) in zip(paths, more_data, strict=True):
        explorer._loaded_data[path] = df.with_columns(
            _unique_id=_get_unique_id(explorer._max_unique_id_int)
        )
        explorer._dtypes[path] = dtypes | {"area": pl.Float64()}
        explorer._max_unique_id_int += 1


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


def _get_stem(path):
    return Path(path).stem


def _get_stem_from_parent(path):
    name = Path(path).stem
    parent_name = Path(path).parent.stem
    return f"{parent_name}/{name}"


def _get_child_paths(paths, file_system):
    child_paths = set()
    for path in paths:
        path = _standardize_path(path)
        suffix = Path(path).suffix
        if suffix:
            these_child_paths = {
                _standardize_path(x)
                for x in file_system.glob(str(Path(path) / f"**/*{suffix}"))
                if Path(path).parts != Path(x).parts
            }
        else:
            these_child_paths = {
                _standardize_path(x)
                for x in file_system.glob(str(Path(path) / f"**/*.*"))
                if Path(path).parts != Path(x).parts
            }
        if not these_child_paths:
            child_paths.add(path)
        else:
            child_paths |= these_child_paths
    return child_paths


def _try_to_get_bounds_else_none(
    path, file_system
) -> tuple[tuple[float] | None, str | None]:
    try:
        return _get_bounds_parquet(path, file_system, pandas_fallback=True)
    except Exception:
        try:
            return _get_bounds_parquet_from_open_file(path, file_system)
        except Exception:
            return None, None


def _get_bounds_series_as_4326(paths, file_system):
    # bounds_series = sg.get_bounds_series(paths, file_system=file_system)
    # return bounds_series.to_crs(4326)

    func = partial(_try_to_get_bounds_else_none, file_system=file_system)
    with ThreadPoolExecutor() as executor:
        bounds_and_crs = list(executor.map(func, paths))

    crss = {json.dumps(x[1]) for x in bounds_and_crs}
    crs = {
        crs
        for crs in crss
        if not any(str(crs).lower() == txt for txt in ["none", "null"])
    }
    if not crs:
        return GeoSeries([None for _ in range(len(paths))], index=paths)
    crs = get_common_crs(crss)
    return GeoSeries(
        [
            shapely.box(*bbox[0]) if bbox[0] is not None else None
            for bbox in bounds_and_crs
        ],
        index=paths,
        crs=crs,
    ).to_crs(4326)


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

    if not lon_delta + lon_delta:
        # if point geometries
        return 16

    # Adjusted width in meters at that latitude
    width_m = lon_delta * (C / 360.0) * math.cos(lat_rad)
    height_m = lat_delta * (C / 360.0)

    # Meters per pixel required
    meters_per_pixel_w = width_m / map_width_px
    meters_per_pixel_h = height_m / map_height_px
    meters_per_pixel = max(meters_per_pixel_w, meters_per_pixel_h)

    # Invert the meters/pixel formula:
    zoom = math.log2(C * math.cos(lat_rad) / (meters_per_pixel * 256))
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


def get_split_index(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        (
            pl.col("__file_path").map_elements(_get_stem, return_dtype=pl.Utf8)
            + " "
            + pl.col("__file_path").cum_count().over("__file_path").cast(pl.Utf8)
        ).alias("split_index")
    )


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


class NoRowsError(ValueError):
    pass


def _get_file_system(path, file_system) -> AbstractFileSystem:
    if file_system is not None:
        return file_system
    if str(path).startswith("gs://"):
        from gcsfs import GCSFileSystem

        return GCSFileSystem()
    return LocalFileSystem()


def get_google_earth_url(center, zoom_m: int = 150) -> str:
    y, x = center
    return f"https://earth.google.com/web/@{y},{x},{zoom_m}a,70.30108914d,35y,0h,0t,0r/data=CgwqBggBEgAYAUICCAE6AwoBMEICCABKDQj___________8BEAA"


def get_google_maps_url(center, zoom_m: int = 150) -> str:
    y, x = center
    url = f"https://www.google.com/maps/@{y},{x},{zoom_m}m/data=!3m1!1e3?entry=ttu&g_ep=EgoyMDI0MTEyNC4xIKXMDSoASAFQAw%3D%3D"
    return url


def _add_cols_to_sql_query(query: str) -> str:
    query = _unformat_query(query)
    if "*" in query:
        return query
    pat = r"\b(SELECT DISTINCT|SELECT)\b"
    select_statement = re.search(pat, query, re.IGNORECASE).group(1)
    _, rest = query.split(select_statement)
    rest = rest.replace("\n", " ")
    cols = ", ".join(col for col in ADDED_COLUMNS if col not in rest)
    return f"{select_statement} {cols}, {rest}"


def _is_polars_expression(txt: Any):
    return (isinstance(txt, str) and "pl.col" in txt) or (isinstance(txt, pl.Expr))


def _is_sql(txt: Any) -> bool:
    return isinstance(txt, str) and txt.replace("\n", "").strip().lower().startswith(
        "select"
    )


@time_function_call(_PROFILE_DICT)
def _is_likely_geopandas_func(df, txt: Any):
    if not isinstance(txt, str) or "pl." in txt:
        return False
    # geopandas_methods = {
    #     # "buffer", "area", "length", "is_empty", "sjoin", "overlay", "clip", "sjoin_nearest", "bounds", "boundary", "geom_type", "set_precision", "make_valid", "is_valid", "centroid", ""
    # }
    geopandas_methods = {
        x
        for x in set(dir(GeoDataFrame))
        .union(set(dir(GeoSeries)))
        .difference(set(dir(pd.DataFrame)))
        .difference(set(dir(pd.Series)))
        .difference(set(dir(pd.Series.str)))
        .difference(set(dir(pl.DataFrame)))
        .difference(set(dir(pl.Series)))
        .difference(set(dir(pl.Series.str)))
        if not x.startswith("__")
    }
    cols = set(df.columns)
    return any(x in txt and len(x) > 2 and x not in cols for x in geopandas_methods)


def _unformat_query(query: str) -> str:
    """Remove newlines and multiple whitespaces from SQL query."""
    query = query.replace("\n", " ").strip()
    while "  " in query:
        query = query.replace("  ", " ")
    return query


@time_function_call(_PROFILE_DICT)
def _cheap_geo_interface(df: pl.DataFrame) -> dict:
    debug_print("_cheap_geo_interface", len(df))
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "id": str(i),
                "type": "Feature",
                "properties": {"_unique_id": id_},
                "geometry": msgspec.json.decode(geom),
            }
            for i, (geom, id_) in enumerate(
                zip(
                    shapely.to_geojson(shapely.from_wkb(df["geometry"])),
                    df["_unique_id"],
                    strict=True,
                )
            )
        ],
    }


@time_function_call(_PROFILE_DICT)
def _get_leaflet_overlay(data, path, **kwargs):
    return dl.Overlay(
        dl.GeoJSON(data=data, id={"type": "geojson", "filename": path}, **kwargs),
        name=_get_stem(path),
        id={"type": "geoj   son-overlay", "filename": path},
        checked=True,
    )


@time_function_call(_PROFILE_DICT)
def _get_multiple_leaflet_overlay(df, path, column, nan_color, alpha, **kwargs):
    values = df.select("_color").unique().collect()["_color"]
    return dl.Overlay(
        dl.LayerGroup(
            [
                dl.GeoJSON(
                    data=_cheap_geo_interface(
                        df.filter(pl.col("_color") == color_).collect()
                    ),
                    style={
                        "color": color_,
                        "fillColor": color_,
                        "weight": 2,
                        "fillOpacity": alpha,
                    },
                    id={
                        "type": "geojson",
                        "filename": path + color_,
                    },
                    **kwargs,
                )
                for color_ in values.drop_nulls()  # .drop_nans()
            ]
            + (
                []
                if not values.is_null().any()
                else [
                    dl.GeoJSON(
                        data=_cheap_geo_interface(
                            df.filter(pl.col(column).is_null()).collect()
                        ),
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
                        **kwargs,
                    )
                ]
            )
        ),
        name=_get_stem(path),
        checked=True,
        id={"type": "geojson-overlay", "filename": path},
    )
