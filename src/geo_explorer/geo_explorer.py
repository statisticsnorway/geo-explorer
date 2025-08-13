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

import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import joblib
import matplotlib
import numpy as np
import pandas as pd
import polars as pl
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
from jenkspy import jenks_breaks
from pandas.api.types import is_datetime64_any_dtype
from shapely.errors import GEOSException
from shapely.geometry import Polygon

from .file_browser import FileBrowser
from .fs import LocalFileSystem
from .utils import _clicked_button_style
from .utils import _standardize_path
from .utils import _unclicked_button_style
from .utils import get_button_with_tooltip

OFFWHITE: str = "#ebebeb"
FILE_CHECKED_COLOR: str = "#3e82ff"
TABLE_TITLE_SUFFIX: str = (
    "(NOTE: to properly zoom to a feature, you may need to click on two separate cells on the same row)"
)

DEBUG: bool = False

if DEBUG:

    def debug_print(*args):
        print(*args)

else:

    def debug_print(*args):
        pass


def _buffer_box(box: Polygon, meters: int) -> Polygon:
    try:
        return sg.to_gdf(box, 4326).to_crs(3035).buffer(meters).to_crs(4326).union_all()
    except GEOSException:
        return box


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


def get_colorpicker_container(color_dict: dict[str, str]) -> html.Div:
    def to_python_type(x):
        if isinstance(x, Number):
            return float(x)
        return x

    color_dict = {to_python_type(key): value for key, value in color_dict.items()}

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            type="color",
                            id={
                                "type": "colorpicker",
                                "column_value": value,
                            },
                            value=color,
                            style={"width": 50, "height": 50},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Label([value]),
                        width="auto",
                    ),
                    dbc.Col(
                        get_button_with_tooltip(
                            "❌",
                            id={
                                "type": "delete-cat-btn",
                                "index": value,
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
            for value, color in color_dict.items()
        ],
        id="color-container",
    )


def _get_df(path, loaded_data, paths_concatted, override: bool = False):
    # cols_to_keep = ["_unique_id", "minx", "miny", "maxx", "maxy", "geometry"]

    t = perf_counter()
    debug_print("_get_df", path, path in loaded_data, path in paths_concatted)
    if path in loaded_data and not override and path in paths_concatted:
        debug_print("_get_df", "00000")
        return []
    if path in loaded_data:
        debug_print("_get_df", "11111")
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
    loaded_data,
    bounds,
    column,
    zoom,
    is_numeric,
    color_dict,
    bins,
    max_rows,
    currently_in_bounds,
    concatted_data,
    nan_color,
    alpha,
):
    debug_print("add_data_one_path", path)
    ns = Namespace("onEachFeatureToggleHighlight", "default")
    data = []
    if concatted_data is None:
        return [None], None, False
    try:
        df = concatted_data.filter(pl.col("__file_path").str.contains(path))
    except Exception as e:
        raise type(e)(f"{e}: {path} - {concatted_data['__file_path']}")

    debug_print(len(df))

    if not len(df):
        return [None], None, False

    df = filter_by_bounds(df, bounds)

    rows_are_hidden = len(df) > max_rows

    if len(df) > max_rows:
        df = df.sample(max_rows)
    df = df.to_pandas()
    df["geometry"] = shapely.from_wkb(df["geometry"])
    df = GeoDataFrame(df, crs=4326)

    debug_print(len(df))

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
    if not any(path in x for x in currently_in_bounds):
        data.append(
            dl.Overlay(
                dl.GeoJSON(id={"type": "geojson", "filename": path}),
                name=_get_name(path),
                checked=True,
                id={"type": "geojson-overlay", "filename": path},
            )
        )
        return data, out_alert, False
    if column and column not in df:
        debug_print("_add_data_one_path111", column)
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
        debug_print("_add_data_one_path222", column)
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
        debug_print("_add_data_one_path333", color, "-", color_dict)
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
    t = perf_counter()
    minx, miny, maxx, maxy = bounds

    df = df.filter(
        (pl.col("minx") <= float(maxx))
        & (pl.col("maxx") >= float(minx))
        & (pl.col("miny") <= float(maxy))
        & (pl.col("maxy") >= float(miny))
    )
    debug_print("filter_by_bounds finished", perf_counter() - t, len(df))
    return df


def _read_and_to_4326(path: str, file_system) -> GeoDataFrame:
    df = sg.read_geopandas(path, file_system=file_system)
    return _fix_df(df)


def _fix_df(df: GeoDataFrame) -> GeoDataFrame:
    df["area"] = df.area
    if df["area"].median() > 10:
        df["area"] = df["area"].astype(int)
    if len(df):
        df = df.to_crs(4326)
    bounds = df.geometry.bounds.astype("float32[pyarrow]")
    df[["minx", "miny", "maxx", "maxy"]] = bounds[["minx", "miny", "maxx", "maxy"]]
    return df


def _get_unique_id(df, i):
    """Float column of 0.0, 0.01, ..., 3.1211 etc."""
    divider = 10 ** len(str(len(df)))
    return (np.array(range(len(df))) / divider) + i


def _read_files(explorer, paths: list[str]) -> None:
    if not paths:
        return
    backend = "threading" if len(paths) <= 3 else "loky"
    with joblib.Parallel(len(paths), backend=backend) as parallel:
        more_data = parallel(
            joblib.delayed(_read_and_to_4326)(path, file_system=explorer.file_system)
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
        df["_unique_id"] = _get_unique_id(
            df, explorer._max_unique_id_int
        )  # len(explorer.loaded_data))
        df = pl.from_pandas(df.assign(geometry=df.geometry.to_wkb()))
        if explorer.splitted:
            df = get_split_index(df)
        explorer.loaded_data[path] = df


def _random_color() -> str:
    r, g, b = np.random.choice(range(256), size=3)
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


class _EmptyColumnContainer:
    """Class with attribute 'columns' as an empty list."""

    columns: ClassVar[list] = []


class GeoExplorer:
    """Class for exploring geodata interactively."""

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
        start_dir: str = "/buckets",
        favorites: list[str] | None = None,
        port: int = 8055,
        file_system: AbstractFileSystem | None = None,
        data: list[str] | None = None,
        selected_features: list[str] | None = None,
        column: str | None = None,
        wms=None,
        center: tuple[float, float] | None = None,
        zoom: int = 10,
        nan_color: str = "#969696",
        nan_label: str = "Missing",
        color_dict: dict | None = None,
        max_zoom: int = 40,
        min_zoom: int = 4,
        max_rows: int = 10_000,
        alpha: float = 0.7,
        zoom_animation: bool = False,
        splitted: bool = False,
        hard_click: bool = False,
    ) -> None:
        """Initialiser."""
        self.start_dir = start_dir
        self.port = port
        if center is not None:
            self.center = center
        else:
            self.center = (59.91740845, 10.71394444)
        self.zoom = zoom
        self.max_zoom = max_zoom
        self.min_zoom = min_zoom
        self.bounds = None
        self.column = column
        self.color_dict = color_dict or {}
        self._color_dict2 = {}
        self.wms = wms or {}
        self.file_system = file_system or LocalFileSystem()
        self.nan_color = nan_color
        self.nan_label = nan_label
        self.splitted = splitted
        self.hard_click = hard_click
        self.max_rows = max_rows
        self.alpha = alpha
        self.file_system = file_system
        self.bounds_series = GeoSeries()
        self.selected_files: dict[str, int] = {}
        self._paths_concatted: set[str] = set()
        self.loaded_data: dict[str, pl.DataFrame] = {}
        self._max_unique_id_int: int = -1
        self._loaded_data_sizes: dict[str, int] = {}
        self.concatted_data: pl.DataFrame | None = None
        self.tile_names: list[str] = []
        self.currently_in_bounds: list[str] = []
        self._file_browser = FileBrowser(
            start_dir, file_system=file_system, favorites=favorites
        )
        self.selected_features = {}

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
            debug_print("\n\n\n\n\nget_layout", self.bounds)
            return dbc.Container(
                [
                    dcc.Location(id="url"),
                    dbc.Row(html.Div(id="alert")),
                    dbc.Row(html.Div(id="alert2")),
                    dbc.Row(html.Div(id="alert3")),
                    dbc.Row(html.Div(id="alert4")),
                    dbc.Row(html.Div(id="new-file-added")),
                    dbc.Row(html.Div(id="loading", style={"height": "3vh"})),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dl.Map(
                                        center=self.center,
                                        bounds=self.bounds,
                                        zoom=self.zoom,
                                        maxZoom=max_zoom,
                                        minZoom=min_zoom,
                                        children=self._map_children
                                        + [
                                            html.Div(id="lc"),
                                        ],
                                        preferCanvas=True,
                                        zoomAnimation=zoom_animation,
                                        id="map",
                                        style={
                                            "height": "90vh",
                                        },
                                    ),
                                ],
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
                                                    tooltip_text="Split all data into separate colors",
                                                ),
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        *get_button_with_tooltip(
                                                            "Export as code",
                                                            id="export",
                                                            style={"color": "#285cd4"},
                                                            tooltip_text="Get code to reproduce current view.",
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
                                                                        "Code to reproduce explore view"
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
                                                            id="hide-wms-button",
                                                            n_clicks=1,
                                                            style={"display": "none"},
                                                        ),
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        dcc.Checklist(
                                                            options=["Norge i bilder"],
                                                            value=[],
                                                            id="wms-checklist",
                                                        ),
                                                    ),
                                                    html.Div(id="wms-items"),
                                                ],
                                                id="hide-wms-div",
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
                                            "borderRadius": "3px",
                                            "padding": "0px",
                                            "backgroundColor": OFFWHITE,
                                            "margin-bottom": "7px",
                                            "margin-top": "7px",
                                            "margin-left": "0px",
                                            "margin-right": "0px",
                                        },
                                    ),
                                    dbc.Row(id="colorpicker-container"),
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
                    dcc.Input(
                        id="debounced_bounds",
                        value=None,
                        style={"display": "none"},
                        debounce=0.25,
                    ),
                    dcc.Store(id="viewport-container", data=None),
                    html.Div(id="color-container", style={"display": "none"}),
                    html.Div(id="currently-in-bounds", style={"display": "none"}),
                    html.Div(id="missing", style={"display": "none"}),
                    html.Div(id="currently-in-bounds2", style={"display": "none"}),
                    html.Div(id="new-file-added2", style={"display": "none"}),
                    html.Div(id="data-was-concatted", style={"display": "none"}),
                    html.Div(id="data-was-changed", style={"display": "none"}),
                    dcc.Store(id="order-was-changed", data=None),
                    html.Div(id="new-data-read", style={"display": "none"}),
                    html.Div(id="max_rows_was_changed", style={"display": "none"}),
                    dbc.Input(id="max_rows_value", style={"display": "none"}),
                    html.Div(id="file-deleted", style={"display": "none"}),
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

        self.app.layout = get_layout

        error_mess = "'data' must be a list of file paths or a dict of GeoDataFrames."
        bounds_series_dict = {}
        for x in data or []:
            if isinstance(x, dict):
                for key, df in x.items():
                    if not isinstance(df, GeoDataFrame):
                        raise ValueError(error_mess)
                    key = _standardize_path(key)
                    df = _fix_df(df)
                    bounds_series_dict[key] = shapely.box(*df.total_bounds)
                    df = pl.from_pandas(df.assign(geometry=df.geometry.to_wkb()))
                    self.loaded_data[key] = df
                    self.selected_files[key] = True
            elif isinstance(x, (str | os.PathLike | PurePath)):
                self.selected_files[_standardize_path(x)] = True
            else:
                raise ValueError(error_mess)

        self.bounds_series = GeoSeries(bounds_series_dict)

        if not self.selected_files:
            self._register_callbacks()
            return

        child_paths = _get_child_paths(
            [x for x in self.selected_files if x not in bounds_series_dict],
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

        _read_files(
            self,
            [x for x in self.selected_files if x not in bounds_series_dict],
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
            # make sure there's two slashes to make link clickable in print
            # (env variable might only have one slash, which redirects to two-slash-url)
            display_url = display_url.replace("https:/", "https://").replace(
                "https:///", "https://"
            )
            self.logger.info(f"\n\nDash is running on {display_url}\n\n")

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
            return html.Div(f"{self}.run()"), True

        @callback(
            Output("file-control-panel", "children"),  # , allow_duplicate=True),
            Output("order-was-changed", "data"),
            Input("data-was-changed", "children"),
            # Input("map", "bounds"),
            Input("file-deleted", "children"),
            Input({"type": "order-button-up", "index": dash.ALL}, "n_clicks"),
            Input({"type": "order-button-down", "index": dash.ALL}, "n_clicks"),
            State({"type": "order-button-up", "index": dash.ALL}, "id"),
            State({"type": "order-button-down", "index": dash.ALL}, "id"),
            State({"type": "filter", "index": dash.ALL}, "value"),
            State({"type": "filter", "index": dash.ALL}, "id"),
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
            filter_functions,
            filter_ids,
            buttons,
        ):
            triggered = dash.callback_context.triggered_id
            if isinstance(triggered, dict) and triggered["type"] == "order-button-up":
                return _change_order(self, n_clicks_up, ids_up, buttons, "up")
            if isinstance(triggered, dict) and triggered["type"] == "order-button-down":
                return _change_order(self, n_clicks_down, ids_down, buttons, "down")

            def get_filter_function_if_any(path):
                try:
                    return get_index(filter_functions, filter_ids, path)
                except ValueError:
                    return None

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
                                        "backgroundColor": FILE_CHECKED_COLOR,
                                    }
                                    if checked
                                    else {
                                        "color": OFFWHITE,
                                        "backgroundColor": OFFWHITE,
                                    }
                                ),
                                tooltip_text="Show/hide data",
                            )
                        ),
                        dbc.Col(
                            get_button_with_tooltip(
                                "Reload",
                                id={
                                    "type": "reload-btn",
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
                                tooltip_text="Reload data (in case categories have been X-ed out)",
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
                            dcc.Input(
                                get_filter_function_if_any(path),
                                placeholder="Filter (with polars or pandas)",
                                id={
                                    "type": "filter",
                                    "index": path,
                                },
                                debounce=3,
                            ),
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "marginBottom": "5px",
                    },
                )
                for i, (path, checked) in enumerate(
                    reversed(self.selected_files.items())
                )
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
                    "backgroundColor": FILE_CHECKED_COLOR,
                }, 0
            else:
                self.selected_files[path] = False
                return {
                    "color": OFFWHITE,
                    "backgroundColor": OFFWHITE,
                }, 1

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
            debug_print("\n\n\n\ndelete_category")
            path_to_delete = get_index_if_clicks(n_clicks_list, delete_ids)
            if path_to_delete is None:
                debug_print("no path to delete\n\n\n\n\n\n\n")
                return dash.no_update, dash.no_update, dash.no_update
            i: int = next(
                iter(
                    i for i, x in enumerate(delete_ids) if x["index"] == path_to_delete
                )
            )
            debug_print(f"path to delete: {path_to_delete}")

            if not self.column:
                self._paths_concatted = {
                    path
                    for path in self._paths_concatted
                    if Path(path).stem != path_to_delete
                }
                return (*self._delete_file(n_clicks_list, delete_ids), dash.no_update)
            else:
                if self.concatted_data[self.column].dtype.is_numeric():
                    return (
                        dash.no_update,
                        dbc.Alert(
                            "Removing categories in numeric columns is not supported",
                            color="warning",
                            dismissable=True,
                        ),
                        dash.no_update,
                    )
                if path_to_delete == self.nan_label:
                    expression = pl.col(self.column).is_not_null()
                else:
                    expression = pl.col(self.column) != path_to_delete

                debug_print(self.concatted_data[self.column].value_counts())
                debug_print(len(self.concatted_data))
                self.concatted_data = self.concatted_data.filter(expression)
                debug_print(len(self.concatted_data))
                debug_print(self.concatted_data[self.column].value_counts())
                self._paths_concatted = set(self.concatted_data["__file_path"].unique())
                debug_print("pop", self._color_dict2.pop(path_to_delete, None))
                self._color_dict2.pop(path_to_delete, None)
                color_container.pop(i)
                for path in list(self.loaded_data):
                    if path not in self._paths_concatted:
                        del self.loaded_data[path]

            return 1, None, color_container

        @callback(
            Output("file-deleted", "children", allow_duplicate=True),
            Input({"type": "reload-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "reload-btn", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def reload_data(n_clicks_list, reload_ids):
            path_to_reload = get_index_if_clicks(n_clicks_list, reload_ids)
            if path_to_reload is None:
                return dash.no_update
            this_data = pl.concat(
                _get_df(
                    path_to_reload,
                    self.loaded_data,
                    self._paths_concatted,
                    override=True,
                )
            )
            other_data = self.concatted_data.filter(
                pl.col("__file_path").str.contains(path_to_reload) == False
            )

            self.concatted_data = pl.concat(
                [this_data, other_data], how="diagonal_relaxed"
            )

        @callback(
            Output("column-dropdown", "value", allow_duplicate=True),
            Input("file-deleted", "children"),
            prevent_initial_call=True,
        )
        def reset_columns(_):
            if not self.selected_files:
                return ""
            return dash.no_update

        @callback(
            Output("splitter", "n_clicks"),
            Output("splitter", "style"),
            Output("is_splitted", "data"),
            Output("column-dropdown", "value", allow_duplicate=True),
            Input("splitter", "n_clicks"),
            Input("column-dropdown", "value"),
            # Input("file-control-panel", "children"),
            prevent_initial_call=True,
        )
        def is_splitted(n_clicks: int, column):
            triggered = dash.callback_context.triggered_id
            if self.concatted_data is None or triggered is None:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            is_splitted: bool = n_clicks % 2 == 1 and not (
                triggered == "column-dropdown" and not column
            )
            if is_splitted:
                style = _clicked_button_style()
            else:
                style = _unclicked_button_style()
            if is_splitted:
                column = "split_index"
            elif column == "split_index":
                column = None
            else:
                column = dash.no_update
            n_clicks = 1 if is_splitted else 0
            return n_clicks, style, is_splitted, column

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
            Output("new-file-added", "children"),
            Output("new-file-added", "style"),
            Input({"type": "load-parquet", "index": dash.ALL}, "n_clicks"),
            Input({"type": "load-parquet", "index": dash.ALL}, "id"),
            Input("is_splitted", "data"),
            State({"type": "file-path", "index": dash.ALL}, "id"),
        )
        def append_path(load_parquet, load_parquet_ids, is_splitted, ids):
            triggered = dash.callback_context.triggered_id
            debug_print("append_path", triggered)
            if triggered == "is_splitted":
                if not is_splitted:
                    return dash.no_update, dash.no_update
                self.splitted = True
                for key, df in self.loaded_data.items():
                    self.loaded_data[key] = self.loaded_data[key].with_columns(
                        split_index=[f"{_get_name(key)} {i}" for i in range(len(df))]
                    )
                self.concatted_data = get_split_index(self.concatted_data)
                return 1, {"display": "none"}
            if not any(load_parquet) or not triggered:
                return dash.no_update, dash.no_update
            try:
                selected_path = triggered["index"]
            except Exception as e:
                raise type(e)(f"{e}: {triggered}") from e
            n_clicks = get_index(load_parquet, load_parquet_ids, selected_path)
            if selected_path in self.selected_files or not n_clicks:
                return dash.no_update, dash.no_update
            try:
                more_bounds = _get_bounds_series(
                    selected_path, file_system=self.file_system
                )
            except Exception as e:
                return (
                    dbc.Alert(
                        f"Couldn't read {selected_path}. {type(e)}: {e}",
                        color="warning",
                        dismissable=True,
                    ),
                    None,
                )
            self.selected_files[selected_path] = True
            self.bounds_series = pd.concat(
                [
                    self.bounds_series,
                    more_bounds,
                ]
            )
            return 1, {"display": "none"}

        @callback(
            Output("debounced_bounds", "value"),
            Input("map", "bounds"),
            Input("map", "zoom"),
            State("map-bounds", "data"),
            prevent_initial_call=True,
        )
        def update_bounds(bounds, zoom, bounds2):
            debug_print("update_bounds", bounds, bounds2)
            if bounds is None:
                return dash.no_update
            self.bounds = bounds
            centroid = shapely.box(*self._nested_bounds_to_bounds(bounds)).centroid
            self.center = (centroid.y, centroid.x)
            self.zoom = zoom
            return json.dumps(bounds)

        @callback(
            Output("new-data-read", "children"),
            Output("missing", "children"),
            Output("interval-component", "disabled"),
            Input("debounced_bounds", "value"),
            Input("new-file-added", "children"),
            Input("file-deleted", "children"),
            Input("interval-component", "n_intervals"),
            Input("missing", "children"),
            Input({"type": "checked-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "checked-btn", "index": dash.ALL}, "id"),
        )
        def get_files_in_bounds(
            bounds,
            file_added,
            file_deleted,
            n_intervals,
            missing,
            checked_clicks,
            checked_ids,
        ):
            t = perf_counter()

            triggered = dash.callback_context.triggered_id
            debug_print("get_files_in_bounds", triggered, len(missing or []), bounds)

            if isinstance(triggered, dict) and triggered["type"] == "checked-btn":
                path = get_index_if_clicks(checked_clicks, checked_ids)
                if path is None:
                    return dash.no_update, dash.no_update, dash.no_update

            if triggered != "missing":
                box = shapely.box(*self._nested_bounds_to_bounds(bounds))
                files_in_bounds = sg.sfilter(self.bounds_series, box)
                self.currently_in_bounds = list(set(files_in_bounds.index))

                def is_checked(path):
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
            if missing:
                if len(missing) > 10:
                    to_read = 0
                    cumsum = 0
                    if not all(path in self._loaded_data_sizes for path in missing):
                        with ThreadPoolExecutor() as executor:
                            more_sizes = {
                                path: x["size"]
                                for path, x in zip(
                                    missing,
                                    executor.map(self.file_system.info, missing),
                                    strict=True,
                                )
                            }
                        self._loaded_data_sizes |= more_sizes
                    for path in missing:
                        size = self._loaded_data_sizes[path]
                        cumsum += size
                        to_read += 1
                        if cumsum > 500_000_000 or to_read > cpu_count() * 2:
                            break
                else:
                    to_read = min(10, len(missing))
                debug_print("to_read", to_read, len(missing))
                if len(missing) > to_read:
                    _read_files(self, missing[:to_read])
                    missing = missing[to_read:]
                    disabled = False
                    new_data_read = dash.no_update if not len(missing) else True
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
                len(missing),
                len(self.loaded_data),
                new_data_read,
                len(missing),
                disabled,
            )

            return new_data_read, missing, disabled

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
            Input("currently-in-bounds2", "children"),
            prevent_initial_call=True,
        )
        def update_column_dropdown_options(_):
            if not self.currently_in_bounds:
                return dash.no_update
            return self._get_column_dropdown_options(self.currently_in_bounds)

        @callback(
            Output("alert2", "children"),
            Input({"type": "filter", "index": dash.ALL}, "value"),
            Input({"type": "filter", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def filter_data(filter_functions: list[str], filter_ids: list[str]):
            if not filter_functions:
                return dash.no_update
            triggered = dash.callback_context.triggered_id

            path = triggered["index"]
            filter_function = get_index(filter_functions, filter_ids, path)
            if filter_function is None:
                # no_update only if None, not if empty string (meaning filter has been filled, then cleared)
                return dash.no_update

            filter_function = filter_function.strip()
            try:
                filter_function = eval(filter_function)
            except Exception:
                pass

            other_data = self.concatted_data.filter(
                pl.col("__file_path").str.contains(path) == False
            )
            # constructing dataset to be filtered from the full dataset, in case it has already been filtered on another query
            this_data = pl.concat(
                _get_df(path, self.loaded_data, self._paths_concatted, override=True)
            )

            out_alert = None
            if filter_function is not None and not (
                isinstance(filter_function, str) and filter_function == ""
            ):
                try:
                    if callable(filter_function):
                        filter_function = filter_function(this_data)
                    this_data = this_data.filter(filter_function)
                except Exception as e:
                    try:
                        this_data = pl.DataFrame(
                            this_data.to_pandas().loc[filter_function]
                        )
                    except Exception as e2:
                        e_name = type(e).__name__
                        e2_name = type(e2).__name__
                        e = str(e)
                        e2 = str(e2)
                        if len(e) > 1000:
                            e = e[:997] + "... "
                        if len(e2) > 1000:
                            e2 = e2[:997] + "... "
                        out_alert = dbc.Alert(
                            (
                                f"Filter function failed with polars ({e_name}: {e}) "
                                f"and pandas: ({e2_name}: {e2})"
                            ),
                            color="warning",
                            dismissable=True,
                        )

            self.concatted_data = pl.concat(
                [this_data, other_data], how="diagonal_relaxed"
            )

            return out_alert

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
            Output("data-was-concatted", "children"),
            Output("data-was-changed", "children"),
            Input("new-data-read", "children"),
            State("debounced_bounds", "value"),
            prevent_initial_call=True,
        )
        def concat_data(new_data_read, bounds):
            debug_print("concat_data", self._paths_concatted)
            t = perf_counter()
            if not new_data_read:
                return dash.no_update, 1

            dfs = [
                _get_df(
                    path,
                    loaded_data=self.loaded_data,
                    paths_concatted=self._paths_concatted,
                )
                for path in self.selected_files
            ]
            debug_print("concat_dat111111", perf_counter() - t)

            dfs = [
                df
                for sublist in dfs
                for df in sublist
                if df is not None and len(df) > 0
            ]
            if dfs:
                if self.concatted_data is not None:
                    dfs.append(self.concatted_data)

                debug_print(dfs)
                self.concatted_data = pl.concat(dfs, how="diagonal_relaxed")
                self._paths_concatted = set(self.concatted_data["__file_path"].unique())

            if DEBUG and self.concatted_data is not None:
                assert len(self.concatted_data) == len(
                    self.concatted_data["_unique_id"].unique()
                ), self.concatted_data.filter(
                    pl.col("_unique_id").is_duplicated()
                ).select(
                    "_unique_id", "__file_path"
                )

            debug_print("concat_data finished after", perf_counter() - t)

            return 1, 1

        @callback(
            Output("force-categorical", "n_clicks"),
            Input("column-dropdown", "value"),
            prevent_initial_call=True,
        )
        def reset_force_categorical(_):
            return 0

        @callback(
            Output("colorpicker-container", "children"),
            Output("bins", "children"),
            Output("is-numeric", "children"),
            Output("force-categorical", "children"),
            Output("currently-in-bounds2", "children"),
            Input("cmap-placeholder", "value"),
            Input("k", "value"),
            Input("force-categorical", "n_clicks"),
            Input("data-was-concatted", "children"),
            Input("alert2", "children"),
            Input("is_splitted", "data"),
            State("column-dropdown", "value"),
            State("debounced_bounds", "value"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "value"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "id"),
            State("bins", "children"),
        )
        def get_column_value_color_dict(
            cmap: str,
            k: int,
            force_categorical_clicks: int,
            data_was_concatted,
            alert2,
            is_splitted,
            column,
            bounds,
            colorpicker_values_list,
            colorpicker_ids,
            bins,
        ):
            triggered = dash.callback_context.triggered_id
            debug_print("\nget_column_value_color_dict", column, triggered)
            if not self.selected_files:
                self.column = None
                column = None
                return html.Div(), None, False, None, 1
            elif not column and triggered is None:
                column = self.column
            else:
                self.column = column
            if triggered is None and self.selected_files:
                self.color_dict = self._color_dict2

            debug_print(self.column, column)
            debug_print(self.color_dict)
            debug_print(self._color_dict2)
            debug_print(self.selected_files)
            if not is_splitted and self.splitted:
                colorpicker_ids, colorpicker_values_list = [], []
                self.splitted = is_splitted

            column_values = [x["column_value"] for x in colorpicker_ids]
            default_colors = list(sg.maps.map._CATEGORICAL_CMAP.values())

            if not column or (
                self.concatted_data is not None and column not in self.concatted_data
            ):
                color_dict = dict(
                    zip(column_values, colorpicker_values_list, strict=True)
                )

                if self.color_dict:
                    color_dict |= self.color_dict

                color_dict = {
                    key: color
                    for key, color in color_dict.items()
                    if any(str(key) in x for x in self.selected_files)
                }

                default_colors = [
                    x for x in default_colors if x not in set(color_dict.values())
                ]

                new_values = [
                    _get_name(value)
                    for value in self.selected_files
                    if _get_name(value) not in color_dict
                ]
                if len(color_dict) < len(default_colors):
                    default_colors = default_colors[
                        : min(len(self.selected_files), len(default_colors))
                    ]
                new_colors = (
                    default_colors
                    + [
                        _random_color()
                        for _ in range(len(new_values) - len(default_colors))
                    ]
                )[: len(new_values)]

                try:
                    color_dict = color_dict | dict(
                        zip(new_values, new_colors, strict=True)
                    )
                except ValueError as e:
                    raise ValueError(f"{e}: {new_values} - {new_colors}") from e

                self._color_dict2 = color_dict

                return (
                    get_colorpicker_container(color_dict),
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
            values_no_nans_unique = values_no_nans.unique()

            if not values_no_nans.dtype.is_numeric():
                force_categorical_button = None
            elif (force_categorical_clicks or 0) % 2 == 0:
                force_categorical_button = html.Button(
                    "Force categorical",
                    n_clicks=force_categorical_clicks,
                    style={
                        "background": "white",
                        "color": "black",
                    },
                )
            else:
                force_categorical_button = html.Button(
                    "Force categorical",
                    n_clicks=force_categorical_clicks,
                    style={
                        "background": "black",
                        "color": "white",
                    },
                )
            is_numeric = (
                force_categorical_clicks or 0
            ) % 2 == 0 and values_no_nans.dtype.is_numeric()

            if is_numeric and len(values_no_nans):
                if len(values_no_nans_unique) <= k:
                    bins = list(values_no_nans_unique)
                else:
                    bins = jenks_breaks(values_no_nans.to_numpy(), n_classes=k)

                if column_values is not None and triggered in [
                    "map",
                    "currently-in-bounds",
                    "currently-in-bounds2",
                ]:
                    color_dict = dict(
                        zip(column_values, colorpicker_values_list, strict=True)
                    )
                else:
                    cmap_ = matplotlib.colormaps.get_cmap(cmap)
                    colors_ = [
                        matplotlib.colors.to_hex(cmap_(int(i)))
                        for i in np.linspace(0, 255, num=k)
                    ]
                    rounded_bins = [round(x, 1) for x in bins]
                    color_dict = {
                        f"{round(min(values_no_nans), 1)} - {rounded_bins[0]}": colors_[
                            0
                        ],
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
                # make sure the existing color scheme is not altered
                if column_values is not None and triggered not in [
                    "column-dropdown",
                    "force-categorical",
                ]:
                    color_dict = dict(
                        zip(column_values, colorpicker_values_list, strict=True)
                    )
                else:
                    color_dict = {}

                new_values = [
                    value
                    for value in values_no_nans_unique
                    if value not in column_values
                ]
                existing_values = [
                    value for value in values_no_nans_unique if value in column_values
                ]
                default_colors = [
                    x for x in default_colors if x not in set(color_dict.values())
                ]
                colors = default_colors[
                    len(existing_values) : min(
                        len(values_no_nans_unique), len(default_colors)
                    )
                ]
                colors = colors + [
                    _random_color() for _ in range(len(new_values) - len(colors))
                ]
                color_dict = color_dict | dict(zip(new_values, colors, strict=True))
                bins = None

            debug_print("\n\ncolor_dict")
            debug_print(color_dict)
            if color_dict.get(self.nan_label, self.nan_color) != self.nan_color:
                self.nan_color = color_dict[self.nan_label]

            elif self.nan_label not in color_dict and polars_isna(values).any():
                color_dict[self.nan_label] = self.nan_color

            if self.color_dict:
                color_dict |= self.color_dict

            self._color_dict2 = color_dict
            return (
                get_colorpicker_container(color_dict),
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
            return "Finished loading. (If not all geometries are rendering, move the map bounds slightly)"

        @callback(
            Output("lc", "children"),
            Output("alert", "children"),
            Output("max_rows", "children"),
            Input("currently-in-bounds2", "children"),
            Input({"type": "colorpicker", "column_value": dash.ALL}, "value"),
            Input("is-numeric", "children"),
            Input("file-deleted", "children"),
            Input("wms-items", "children"),
            Input("wms-checklist", "value"),
            Input("new-file-added2", "children"),
            Input("max_rows_was_changed", "children"),
            Input("data-was-changed", "children"),
            Input("order-was-changed", "data"),
            Input("alpha", "value"),
            Input({"type": "checked-btn", "index": dash.ALL}, "n_clicks"),
            State("debounced_bounds", "value"),
            State("map", "zoom"),
            State("column-dropdown", "value"),
            State("bins", "children"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "id"),
            State("max_rows", "children"),
        )
        def add_data(
            currently_in_bounds2,
            colorpicker_values_list,
            is_numeric,
            file_deleted,
            wms,
            wms_checked,
            new_file_added2,
            max_rows_was_changed,
            data_was_changed,
            order_was_changed,
            alpha,
            checked_clicks,
            bounds,
            zoom,
            column,
            bins,
            colorpicker_ids,
            max_rows_component,
        ):
            debug_print("\n\nadd_data", len(self.loaded_data))
            debug_print(bounds)
            t = perf_counter()

            bounds = self._nested_bounds_to_bounds(bounds)

            column_values = [x["column_value"] for x in colorpicker_ids]
            color_dict = dict(zip(column_values, colorpicker_values_list, strict=True))
            debug_print(color_dict)

            wms_layers = []
            tiles = []
            for wms_name, wms_obj in self.wms.items():
                if wms_name not in wms_checked:
                    continue
                tiles = wms_obj._filter_tiles(shapely.box(*bounds))["name"]
                self.tile_names = list(tiles)
                for tile in tiles:
                    wms_layers.append(
                        dl.Overlay(
                            dl.WMSTileLayer(
                                url=wms_obj.url,
                                layers=tile,
                                format="image/png",
                                transparent=True,
                            ),
                            name=tile,
                            checked=wms_obj.checked,
                        )
                    )

            if is_numeric:
                color_dict = {i: color for i, color in enumerate(color_dict.values())}

            add_data_func = partial(
                _add_data_one_path,
                loaded_data=self.loaded_data,
                max_rows=self.max_rows,
                currently_in_bounds=self.currently_in_bounds,
                concatted_data=self.concatted_data,
                nan_color=self.nan_color,
                bounds=bounds,
                column=column,
                zoom=zoom,
                is_numeric=is_numeric,
                color_dict=color_dict,
                bins=bins,
                alpha=alpha,
            )
            debug_print("add_data111", perf_counter() - t)

            results = [
                add_data_func(path)
                for path, checked in self.selected_files.items()
                if checked
            ]
            debug_print("add_data2222", perf_counter() - t, len(results))

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
                len(
                    {x for x in self.concatted_data["__file_path"]}
                    if self.concatted_data is not None
                    else []
                ),
            )
            if self.splitted:
                debug_print(self.concatted_data["split_index"])

            if rows_are_not_hidden:
                max_rows_component = None
            else:
                max_rows_component = _get_max_rows_displayed_component(self.max_rows)

            return (
                dl.LayersControl(self._base_layers + wms_layers + data),
                (out_alert if out_alert else None),
                max_rows_component,
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
            return (f"Clicked features (n={len(features)}) {TABLE_TITLE_SUFFIX}",)

        @callback(
            Output("all-features-title", "children"),
            Input("all-features", "data"),
        )
        def update_all_features_title(features):
            return (f"All features (n={len(features)}) {TABLE_TITLE_SUFFIX}",)

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
            if triggered is None:
                clicked_ids = list(self.selected_features)
                clicked_features = list(self.selected_features.values())
                return clicked_features, clicked_ids, None
            if triggered == "clear-table-clicked":
                self.selected_features = {}
                return [], [], None

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
                    _used_file_paths.add(selected_path)
                    break

            feature = features[index]
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
                    .filter(pl.col("geometry").map_elements(geoms_relate))
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
            debug_print(clicked_path)
            if clicked_path is None:
                return dash.no_update
            data = self.concatted_data.filter(
                pl.col("__file_path").str.contains(clicked_path)
            )
            debug_print(data)
            for path in self.loaded_data:
                if clicked_path in path:
                    columns = set(self.loaded_data[path].columns).difference(
                        {"geometry"}
                    )
                    break
            clicked_features = data[list(columns)].to_dicts()
            return clicked_features

        @callback(
            Output("feature-table-rows", "columns"),
            Output("feature-table-rows", "data"),
            Output("feature-table-rows", "style_table"),
            Output("feature-table-rows", "hidden_columns"),
            Input("all-features", "data"),
            State("column-dropdown", "options"),
            State("feature-table-rows", "style_table"),
            # prevent_initial_call=True,
        )
        def update_table(data, column_dropdown, style_table):
            return self._update_table(data, column_dropdown, style_table)

        @callback(
            Output("feature-table-rows-clicked", "columns"),
            Output("feature-table-rows-clicked", "data"),
            Output("feature-table-rows-clicked", "style_table"),
            Output("feature-table-rows-clicked", "hidden_columns"),
            Input("clicked-features", "data"),
            State("column-dropdown", "options"),
            State("feature-table-rows-clicked", "style_table"),
            # prevent_initial_call=True,
        )
        def update_table_clicked(data, column_dropdown, style_table):
            return self._update_table(data, column_dropdown, style_table)

        @callback(
            Output("map", "bounds"),
            Output("map", "zoom"),
            Output("map", "center"),
            State("map-bounds", "data"),
            Input("map-zoom", "data"),
            State("map-center", "data"),
            prevent_initial_call=True,
        )
        def intermediate_update_bounds(bounds, zoom, center):
            """Update map bounds after short sleep because otherwise it's buggy."""
            time.sleep(0.1)
            if not zoom and not bounds and not center:
                return dash.no_update, dash.no_update, dash.no_update
            debug_print("intermediate_update_bounds", zoom, bounds, center)
            return bounds, zoom, center

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
            matches = (
                self.concatted_data.lazy()
                .filter(pl.col("_unique_id") == unique_id)
                .select("minx", "miny", "maxx", "maxy")
                .collect()
            )
            if not len(matches):
                return dash.no_update, dash.no_update, dash.no_update
            minx, miny, maxx, maxy = matches.row(0)
            center = ((miny + maxy) / 2, (minx + maxx) / 2)
            debug_print(center)
            bounds = [[miny, minx], [maxy, maxx]]
            debug_print(bounds)

            width = int(viewport["width"] * 0.7)
            height = int(viewport["height"] * 0.7)
            zoom_level = lat_lon_bounds_to_zoom(minx, miny, maxx, maxy, width, height)
            zoom_level = min(zoom_level, self.max_zoom)
            zoom_level = max(zoom_level, self.min_zoom)
            debug_print(zoom_level)
            return bounds, int(zoom_level), center

        @callback(
            Output("new-file-added2", "children"),
            Input({"type": "from-year", "index": dash.ALL}, "value"),
            Input({"type": "to-year", "index": dash.ALL}, "value"),
            Input({"type": "wms-not-contains", "index": dash.ALL}, "value"),
            Input({"type": "wms-checked", "index": dash.ALL}, "value"),
            prevent_initial_call=True,
        )
        def get_wms_object(from_year, to_year, not_contains, checked):
            triggered = dash.callback_context.triggered_id
            if triggered is None:
                for wms_name in list(self.wms):
                    if wms_name not in checked:
                        self.wms.pop(wms_name)
                return
            wms_name = triggered["index"]
            from_year = next(iter(from_year))
            to_year = next(iter(to_year))
            not_contains = next(iter(not_contains)) or None
            checked = next(iter(checked))
            if not_contains is not None:
                not_contains = [x.strip(",") for x in not_contains.split(" ")]
            try:
                self.wms[wms_name] = self._wms_constructors[wms_name](
                    years=np.arange(int(from_year), int(to_year)),
                    not_contains=not_contains,
                )
            except Exception:
                return dash.no_update
            self.wms[wms_name].checked = bool(checked)
            return 1

        @callback(
            Output("hide-wms-div", "style"),
            Input("hide-wms-button", "n_clicks"),
            prevent_initial_call=True,
        )
        def hide_wms(n_clicks):
            if not n_clicks or n_clicks % 2 == 0:
                return None
            return {"display": "none"}

        @callback(
            Output("wms-items", "children"),
            Input("wms-checklist", "value"),
            State("wms-items", "children"),
            prevent_initial_call=True,
        )
        def add_wms_panel(
            checklist_items,
            items,
        ):
            items = []
            if not checklist_items:
                return items
            for wms_name in checklist_items:
                defaults = {
                    arg: default
                    for arg, default in zip(
                        inspect.getfullargspec(self._wms_constructors[wms_name]).args[
                            1:
                        ],
                        inspect.getfullargspec(
                            self._wms_constructors[wms_name]
                        ).defaults,
                        strict=True,
                    )
                }
                from_year = int(list(defaults["years"])[0])
                to_year = int(list(defaults["years"])[-1])

                self.wms[wms_name] = self._wms_constructors[wms_name](
                    years=np.arange(int(from_year), int(to_year)),
                )
                self.wms[wms_name].checked = False

                # from_year = int(list(self.wms[wms_name].years[0]))
                # to_year = int(list(self.wms[wms_name].years[-1]))

                items.append(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Row(
                                        dcc.Input(
                                            value=from_year,
                                            id={"type": "from-year", "index": wms_name},
                                            type="number",
                                            placeholder="From year",
                                        )
                                    ),
                                    dbc.Row(
                                        dcc.Input(
                                            value=to_year,
                                            id={"type": "to-year", "index": wms_name},
                                            type="number",
                                            placeholder="To year",
                                        )
                                    ),
                                    dbc.Row(
                                        dcc.Input(
                                            id={
                                                "type": "wms-not-contains",
                                                "index": wms_name,
                                            },
                                            type="text",
                                            placeholder="Not contains",
                                        )
                                    ),
                                    dbc.Row(
                                        dcc.Checklist(
                                            options=["show/hide all"],
                                            value=[],
                                            id={
                                                "type": "wms-checked",
                                                "index": wms_name,
                                            },
                                        )
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "border": "1px solid #ccc",
                                    "borderRadius": "5px",
                                    "padding": "20px",
                                    "backgroundColor": "white",
                                    "gap": "10px",
                                    "width": "300px",
                                },
                                id=f"wms-item-{wms_name}",
                            ),
                        ],
                    )
                )

            return items

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

    def _get_column_dropdown_options(self, currently_in_bounds):
        columns = set(
            itertools.chain.from_iterable(
                set(
                    self.loaded_data.get(path, _EmptyColumnContainer).columns
                ).difference(
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
                for path in currently_in_bounds
            )
        )
        return [{"label": col, "value": col} for col in sorted(columns)]

    def _update_table(self, data, column_dropdown, style_table):
        debug_print("update_table")
        if not data:
            return None, None, style_table | {"height": "1vh"}, None
        if column_dropdown is None:
            column_dropdown = self._get_column_dropdown_options(list(self.loaded_data))
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
        debug_print("\n\n\n\ndelete_file")
        path_to_delete = get_index_if_clicks(n_clicks_list, delete_ids)
        if path_to_delete is None:
            debug_print("no path to delete\n\n\n\n\n\n\n")
            return dash.no_update, dash.no_update
        debug_print(f"path to delete: {path_to_delete}")
        for path in dict(self.selected_files):
            if path_to_delete in [path, Path(path).stem]:
                self.selected_files.pop(path)

        self._paths_concatted = {
            path for path in self._paths_concatted if path != path_to_delete
        }

        for path in list(self.loaded_data):
            if path_to_delete in path:
                del self.loaded_data[path]
                self.concatted_data = self.concatted_data.filter(
                    pl.col("__file_path").str.contains(path) == False
                )
        self.bounds_series = self.bounds_series[
            lambda x: ~x.index.str.contains(path_to_delete)
        ]
        return 1, None

    def _nested_bounds_to_bounds(
        self,
        bounds: list[list[float]],
    ) -> tuple[float, float, float, float]:
        if bounds is None:
            return (
                sg.to_gdf(reversed(self.center), 4326)
                .to_crs(3035)
                .buffer(165_000 / (self.zoom**1.25))
                .to_crs(4326)
                .total_bounds
            )
        if isinstance(bounds, str):
            bounds = json.loads(bounds)
        mins, maxs = bounds
        miny, minx = mins
        maxy, maxx = maxs
        return minx, miny, maxx, maxy

    def __str__(self) -> str:
        """String representation."""

        def maybe_to_string(x):
            if isinstance(x, str):
                return f"'{x}'"
            return x

        data = {
            key: value
            for key, value in self.__dict__.items()
            if key
            not in [
                "paths",
                "app",
                "currently_in_bounds",
                "bounds_series",
                "loaded_data",
                "bounds",
                "tile_names",
                "concatted_data",
                "splitted",
            ]
            and not key.startswith("_")
            and not (isinstance(value, (dict, list, tuple)) and not value)
        }

        if self.selected_files:
            data["data"] = list(data.pop("selected_files"))
        else:
            data.pop("selected_files")

        if self._file_browser.favorites:
            data["favorites"] = self._file_browser.favorites

        if "selected_features" in data:
            data["selected_features"] = [
                x["_unique_id"] for x in data["selected_features"].values()
            ]

        txt = ", ".join(f"{k}={maybe_to_string(v)}" for k, v in data.items())
        return f"{self.__class__.__name__}({txt})"


def get_index(values: list[Any], ids: list[Any], index: Any):
    i = [x["index"] for x in ids].index(index)
    return values[i]


def lat_lon_bounds_to_zoom(
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
    debug_print(map_width_px, map_height_px)

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
    return round(zoom, 2)


def get_index_if_clicks(n_clicks_list, ids) -> str | None:
    debug_print("get_index_if_clicks")
    debug_print(n_clicks_list)
    debug_print(ids)
    if not any(n_clicks_list):
        return None
    triggered = dash.callback_context.triggered_id
    debug_print("get_index_if_clicks", triggered)
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
                        "backgroundColor": "#2f2f2f",
                        "color": "white",
                        "fontWeight": "bold",
                    },
                    style_data={
                        "backgroundColor": OFFWHITE,
                        "color": "black",
                    },
                    style_table={
                        "overflowX": "show",
                        "overflowY": "scroll",
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
    # int_col_as_str = pl.int_range(0, len(df), eager=True).cast(pl.Utf8)

    # debug_print(
    #     df.with_columns(
    #         pl.col("__file_path").cum_count().over("__file_path").alias("cumulative_count")
    #     ).with_columns((pl.col("cumulative_count") + 1).alias("cumulative_count"))
    # )

    # df = df.with_columns(
    #     pl.col("__file_path").cum_count().over("__file_path").alias("cumulative_count")
    # ).with_columns((pl.col("cumulative_count") + 1).alias("cumulative_count"))

    return df.with_columns(
        (
            pl.col("__file_path").map_elements(_get_name, return_dtype=pl.Utf8)
            + " "
            + pl.col("__file_path").cum_count().over("__file_path").cast(pl.Utf8)
        ).alias("split_index")
    )
