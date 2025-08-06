import inspect
import itertools
import json
import math
import os
import signal
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

OFFWHITE = "#ebebeb"
DEBUG = 1

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
            style={"width": "10vh"},
        ),
    ]


def _change_order(explorer, n_clicks_list, buttons, what: str):
    if what not in ["up", "down"]:
        raise ValueError(what)
    if not any(n_clicks_list) or not buttons:
        return dash.no_update, dash.no_update
    triggered = dash.callback_context.triggered_id
    i = triggered["index"]
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
                        html.Button(
                            "❌",
                            id={
                                "type": "delete-btn",
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
    )


def _get_df(path, loaded_data, paths_concatted, override: bool = False):
    # cols_to_keep = ["_unique_id", "minx", "miny", "maxx", "maxy", "geometry"]

    t = perf_counter()
    if (
        path in loaded_data
        and not override
        and path in paths_concatted
        # (
        #     concatted_data is not None and (concatted_data["__file_path"] == path).any()
        # )
    ):
        # data already loaded and filtered
        return []
    if path in loaded_data:
        df = loaded_data[path].with_columns(__file_path=pl.lit(path))
        return [df]

    if paths_concatted:  # is not None and len(concatted_data):
        debug_print("_get_df", 222)
        # paths_loaded = {x for x in concatted_data["__file_path"].unique() if path in x}
        matches = [
            key
            for key in loaded_data
            if path in key and (override or key not in paths_concatted)
        ]
        if matches:
            matches = [
                loaded_data[key].with_columns(__file_path=pl.lit(key))
                for key in loaded_data
            ]

    else:
        debug_print("_get_df", 333)
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
    if len(df):
        df = df.to_crs(4326)
    bounds = df.geometry.bounds.astype("float32[pyarrow]")
    df[bounds.columns] = bounds
    return df


def _get_unique_id(df, i):
    """Float column of 0.0, 0.01, ..., 3.1211 etc."""
    divider = 10 ** len(str(len(df)))
    return (pd.Series(range(len(df)), index=df.index) / divider) + i


def _read_files(explorer, paths: list[str]) -> None:
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
        df["_unique_id"] = _get_unique_id(df, len(explorer.loaded_data))
        if explorer.splitted:
            df["split_index"] = [f"{_get_name(path)} {i}" for i in range(len(df))]
        df = pl.from_pandas(df.assign(geometry=df.geometry.to_wkb()))
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
        self.color_dict2 = {}
        self.wms = wms or {}
        self.file_system = file_system or LocalFileSystem()
        self.nan_color = nan_color
        self.nan_label = nan_label
        self.splitted = splitted
        self.max_rows = max_rows
        self.alpha = alpha
        self.file_system = file_system
        self.selected_features: list[str] = (
            selected_features if selected_features is not None else {}
        )
        self.bounds_series = GeoSeries()
        self.selected_files: dict[str, int] = {}
        self.paths_concatted: set[str] = set()
        self.loaded_data: dict[str, pl.DataFrame] = {}
        self._loaded_data_sizes: dict[str, int] = {}
        self.concatted_data: pl.DataFrame | None = None
        self.tile_names: list[str] = []
        self.currently_in_bounds: list[str] = []
        self.file_browser = FileBrowser(start_dir, file_system)

        self.app = Dash(
            __name__,
            suppress_callback_exceptions=True,
            external_stylesheets=[dbc.themes.SOLAR],
            requests_pathname_prefix=f"/proxy/{self.port}/" if self.port else None,
            serve_locally=True,
            assets_folder="assets",
        )

        clicked_features = []

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
                                            "width": "130vh",
                                            "height": "90vh",
                                        },
                                    ),
                                ],
                                width=8,
                            ),
                            dbc.Col(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.Button(
                                                    "Split rows",
                                                    id="splitter",
                                                    n_clicks=1 if self.splitted else 0,
                                                ),
                                                style={"display": "none"},
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Button(
                                                            "Export as code",
                                                            id="export",
                                                            style={"color": "#285cd4"},
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
                                                                        "Code to reproduce explore view."
                                                                    ),
                                                                    close_button=False,
                                                                ),
                                                                dbc.ModalBody(
                                                                    id="export-text"
                                                                ),
                                                                dbc.ModalFooter(
                                                                    dbc.Button(
                                                                        "Copy code",
                                                                        id="copy-export",
                                                                        className="ms-auto",
                                                                        n_clicks=0,
                                                                    )
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
                                            dbc.Col(id="remove-buttons"),
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
                                    "overflow": "scroll",
                                },
                                className="scroll-container",
                            ),
                        ],
                    ),
                    dbc.Row(html.Div(id="loading", style={"height": "3vh"})),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Button(
                                    "❌ Clear table",
                                    id="clear-table",
                                    style={
                                        "color": "red",
                                        "border": "none",
                                        "background": "none",
                                        "cursor": "pointer",
                                    },
                                ),
                                width=1,
                            ),
                            dbc.Col(
                                html.Div(
                                    dash_table.DataTable(
                                        id="feature-table-rows",
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
                                    id="feature-table-container",
                                ),
                                style={"width": "100%", "height": "auto"},
                                width=11,
                            ),
                        ],
                        style={
                            "height": "auto",
                            "overflow": "visible",
                        },
                    ),
                    *self.file_browser.get_file_browser_components(),
                    dcc.Store(id="is_splitted", data=False),
                    dcc.Input(
                        id="debounced_bounds",
                        value=None,
                        style={"display": "none"},
                        debounce=0.25,
                    ),
                    dcc.Store(id="viewport-container", data=None),
                    html.Div(id="currently-in-bounds", style={"display": "none"}),
                    html.Div(id="missing", style={"display": "none"}),
                    html.Div(id="currently-in-bounds2", style={"display": "none"}),
                    html.Div(id="new-file-added2", style={"display": "none"}),
                    html.Div(id="data-was-concatted", style={"display": "none"}),
                    html.Div(id="data-was-changed", style={"display": "none"}),
                    dcc.Store(id="order-was-changed", data=None),
                    html.Div(id="new-data-read", style={"display": "none"}),
                    html.Div(id="max_rows_was_changed", style={"display": "none"}),
                    html.Div(id="file-removed", style={"display": "none"}),
                    dcc.Store(id="dummy-output", data=None),
                    dcc.Store(id="dummy-output2", data=None),
                    html.Div(id="bins", style={"display": "none"}),
                    html.Div(False, id="is-numeric", style={"display": "none"}),
                    dcc.Store(id="map-bounds", data=None),
                    dcc.Store(id="map-zoom", data=None),
                    dcc.Store(id="map-center", data=None),
                    dcc.Store(id="clicked-features", data=clicked_features),
                    dcc.Store(id="clicked-ids", data=self.selected_features),
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
                for key, value in x.items():
                    if not isinstance(value, GeoDataFrame):
                        raise ValueError(error_mess)
                    key = _standardize_path(key)
                    if len(value):
                        value = value.to_crs(4326)
                    self.loaded_data[key] = value
                    self.selected_files[key] = 0
                    bounds_series_dict[key] = shapely.box(
                        *self.loaded_data[key].total_bounds
                    )
            elif isinstance(x, (str | os.PathLike | PurePath)):
                self.selected_files[_standardize_path(x)] = 0
            else:
                raise ValueError(error_mess)

        self.bounds_series = GeoSeries(bounds_series_dict)

        if self.selected_files:
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

            # dataframe dicts as input data will now be sorted first because they were added to loaded_data first.
            # now to get back original order
            loaded_data_sorted = {}
            for x in data:
                if isinstance(x, dict):
                    for key in x:
                        key = _standardize_path(key)
                        loaded_data_sorted[key] = self.loaded_data[key].assign(
                            _unique_id=lambda df: _get_unique_id(
                                df, len(loaded_data_sorted)
                            )
                        )
                else:
                    x = _standardize_path(x)
                    loaded_data_sorted[x] = self.loaded_data[x].assign(
                        _unique_id=lambda df: _get_unique_id(
                            df, len(loaded_data_sorted)
                        )
                    )

            self.loaded_data = loaded_data_sorted

            for idx in self.selected_features:
                i = int(idx[0])
                df = list(self.loaded_data.values())[i]
                row = df[lambda x: x["_unique_id"] == idx]
                assert len(row) == 1, (len(row), df)
                features = row.__geo_interface__["features"]
                assert len(features) == 1
                feature = next(iter(features))
                clicked_features.append(feature["properties"])

        self._register_callbacks()

    def run(self, debug: bool = False, jupyter_mode: str = "external") -> None:
        """Run the app."""
        try:
            self.app.run(
                debug=debug, port=self.port, jupyter_mode=jupyter_mode, threaded=False
            )
        except KeyboardInterrupt:
            os.kill(os.getpid(), signal.SIGTERM)

    def _register_callbacks(self) -> None:

        @callback(
            Output("export-text", "children"),
            Output("export-view", "is_open"),
            Input("export", "n_clicks"),
            Input("file-removed", "children"),
            State("debounced_bounds", "value"),
            State("map", "zoom"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "value"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def export(
            export_clicks,
            remove,
            bounds,
            zoom,
            colorpicker_values_list,
            colorpicker_ids,
        ):
            triggered = dash.callback_context.triggered_id
            if triggered in ["file-removed", "close-export"] or not export_clicks:
                return None, False

            centroid = shapely.box(*self._nested_bounds_to_bounds(bounds)).centroid
            center = (centroid.y, centroid.x)

            column_values = [x["column_value"] for x in colorpicker_ids]
            color_dict = dict(zip(column_values, colorpicker_values_list, strict=True))

            def to_string(x):
                if isinstance(x, str):
                    return f"'{x}'"
                return x

            defaults = {
                arg: default
                for arg, default in zip(
                    inspect.getfullargspec(self.__class__).args[1:],
                    inspect.getfullargspec(self.__class__).defaults,
                    strict=True,
                )
            }
            data = {
                k: v
                for k, v in self.__dict__.items()
                if k
                not in [
                    "paths",
                    "app",
                    "currently_in_bounds",
                    "bounds_series",
                    "loaded_data",
                    "center",
                    "bounds",
                    "zoom",
                    "tile_names",
                    "concatted_data",
                ]
                and not (k in defaults and v == defaults[k])
            } | {"zoom": zoom, "center": center, "color_dict": color_dict}
            if self.selected_files:
                data["data"] = list(data.pop("selected_files"))
            else:
                data.pop("selected_files")

            content = [
                html.Span(f"from geo_explorer import {self.__class__.__name__}"),
                html.Br(),
            ]
            if self.file_system.__module__ == "geo_explorer.fs":
                content.append(
                    html.Span(
                        f"from geo_explorer import {self.file_system.__class__.__name__}"
                    )
                )
                content.append(html.Br())

            content.append(html.Span(f"{self.__class__.__name__}("))
            content.append(html.Br())
            for k, v in data.items():
                content.append(
                    html.Span(f"{k}={to_string(v)},", style={"padding-left": "4ch"})
                )
                content.append(html.Br())
            content.append(html.Span(").run()"))
            return (html.Div(content), True)

        @callback(
            Output("remove-buttons", "children", allow_duplicate=True),
            # Input("new-file-added", "children"),
            Input("alert", "children"),
            Input("file-removed", "children"),
            State({"type": "filter", "index": dash.ALL}, "value"),
            State({"type": "filter", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def render_items(new_file_added, file_removed, filter_functions, filter_ids):
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
                                    html.Button(
                                        "🡑",
                                        id={
                                            "type": "order-button-up",
                                            "index": i,
                                        },
                                        n_clicks=0,
                                        style={"width": "1vh"},
                                    ),
                                ),
                                dbc.Row(
                                    html.Button(
                                        "🡓",
                                        id={
                                            "type": "order-button-down",
                                            "index": i,
                                        },
                                        n_clicks=0,
                                        style={"width": "1vh"},
                                    ),
                                ),
                            ],
                            style={
                                "marginRight": "10px",
                            },
                        ),
                        dbc.Col(
                            html.Button(
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
                            )
                        ),
                        dbc.Col(
                            html.Button(
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
                            )
                        ),
                        dbc.Col(html.Span(path)),
                        dbc.Col(
                            html.Button(
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
                            )
                        ),
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
                for i, path in enumerate(reversed(self.selected_files))
            ]

        @callback(
            Output("remove-buttons", "children", allow_duplicate=True),
            Output("order-was-changed", "data"),
            Input({"type": "order-button-up", "index": dash.ALL}, "n_clicks"),
            Input({"type": "order-button-down", "index": dash.ALL}, "n_clicks"),
            State("remove-buttons", "children"),
            prevent_initial_call=True,
        )
        def change_order(n_clicks_up, n_clicks_down, buttons):
            triggered = dash.callback_context.triggered_id
            if triggered and triggered["type"] == "order-button-up":
                return _change_order(self, n_clicks_up, buttons, "up")
            else:
                return _change_order(self, n_clicks_down, buttons, "down")

        @callback(
            Output("file-removed", "children", allow_duplicate=True),
            Output("alert3", "children"),
            Input({"type": "delete-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "delete-btn", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def delete_item(n_clicks_list, delete_ids):
            path_to_remove = get_index_if_clicks(n_clicks_list, delete_ids)
            if path_to_remove is None:
                return dash.no_update
            debug_print("\n\n\n\n\n\ndelete_item", path_to_remove)
            for path in dict(self.selected_files):
                if path_to_remove in [path, Path(path).stem]:
                    self.selected_files.pop(path)
            any_removed = False
            for path in list(self.loaded_data):
                if path_to_remove in path:
                    del self.loaded_data[path]
                    self.concatted_data = self.concatted_data.filter(
                        pl.col("__file_path").str.contains(path) == False
                    )
                    any_removed = True
            if not any_removed and self.column:
                debug_print(
                    "not any_removed and self.column", self.column, path_to_remove
                )
                if self.concatted_data[self.column].dtype.is_numeric():
                    return dash.no_update, dbc.Alert(
                        "Removing categories in numeric columns is not supported",
                        color="warning",
                        dismissable=True,
                    )
                if path_to_remove == self.nan_label:
                    expression = pl.col(self.column).is_not_null()
                else:
                    expression = pl.col(self.column) != path_to_remove

                debug_print(self.concatted_data[self.column].value_counts())
                debug_print(len(self.concatted_data))
                self.concatted_data = self.concatted_data.filter(expression)
                debug_print(len(self.concatted_data))
                debug_print(self.concatted_data[self.column].value_counts())
            else:
                self.bounds_series = self.bounds_series[
                    lambda x: ~x.index.str.contains(path_to_remove)
                ]
            return 1, None

        @callback(
            Output("file-removed", "children", allow_duplicate=True),
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
                    self.paths_concatted,
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
            Input("file-removed", "children"),
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
            # Input("remove-buttons", "children"),
            prevent_initial_call=True,
        )
        def is_splitted(n_clicks: int, column):
            if self.concatted_data is None:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            triggered = dash.callback_context.triggered_id
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
            self.selected_files[selected_path] = 0
            self.bounds_series = pd.concat(
                [
                    self.bounds_series,
                    more_bounds,
                ]
            )
            return 1, {"display": "none"}

        @callback(
            Output("debounced_bounds", "value", allow_duplicate=True),
            Input("map", "bounds"),
            Input("map", "zoom"),
            prevent_initial_call=True,
        )
        def update_bounds(bounds, zoom):
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
            Input("file-removed", "children"),
            Input("interval-component", "n_intervals"),
            Input("missing", "children"),
        )
        def get_files_in_bounds(bounds, file_added, file_removed, n_intervals, missing):
            t = perf_counter()
            triggered = dash.callback_context.triggered_id
            debug_print("get_files_in_bounds", triggered, len(missing or []), bounds)

            if triggered != "missing":
                box = shapely.box(*self._nested_bounds_to_bounds(bounds))
                files_in_bounds = sg.sfilter(self.bounds_series, box)
                self.currently_in_bounds = list(set(files_in_bounds.index))
                missing = list(
                    {
                        path
                        for path in files_in_bounds.index
                        if path not in self.loaded_data
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
                _get_df(path, self.loaded_data, self.paths_concatted, override=True)
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
                        out_alert = dbc.Alert(
                            (
                                f"Filter function failed with polars ({type(e).__name__}: {e!s}) "
                                f"and pandas: ({type(e2).__name__}: {e2!s})"
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
            debug_print("concat_data")
            t = perf_counter()
            if not new_data_read:
                return dash.no_update, 1

            dfs = [
                _get_df(
                    path,
                    loaded_data=self.loaded_data,
                    paths_concatted=self.paths_concatted,
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
                self.concatted_data = pl.concat(dfs, how="diagonal_relaxed")
                self.paths_concatted = set(self.concatted_data["__file_path"].unique())

            debug_print("concat_data finished after", perf_counter() - t)

            return 1, 1

        @callback(
            Output("colorpicker-container", "children"),
            Output("bins", "children"),
            Output("is-numeric", "children"),
            Output("force-categorical", "children"),
            Output("currently-in-bounds2", "children"),
            Input("column-dropdown", "value"),
            Input("cmap-placeholder", "value"),
            Input("k", "value"),
            Input("force-categorical", "n_clicks"),
            Input("data-was-concatted", "children"),
            Input("alert2", "children"),
            State("debounced_bounds", "value"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "value"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "id"),
            State("bins", "children"),
            State("is_splitted", "data"),
        )
        def get_column_value_color_dict(
            column,
            cmap: str,
            k: int,
            force_categorical_clicks: int,
            data_was_concatted,
            alert2,
            bounds,
            colorpicker_values_list,
            colorpicker_ids,
            bins,
            is_splitted,
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
                self.color_dict = self.color_dict2

            debug_print(self.column, column)
            debug_print(self.color_dict)
            debug_print(self.color_dict2)
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

                self.color_dict2 = color_dict

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

            self.color_dict2 = color_dict
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
            return "Finished loading. (Tip: change the map bounds slightly if not all geometries are loaded after a few seconds)"

        @callback(
            Output("lc", "children"),
            Output("alert", "children"),
            Output("max_rows", "children"),
            Input("currently-in-bounds2", "children"),
            Input({"type": "colorpicker", "column_value": dash.ALL}, "value"),
            Input("is-numeric", "children"),
            Input("file-removed", "children"),
            Input("wms-items", "children"),
            Input("wms-checklist", "value"),
            Input("new-file-added2", "children"),
            Input("max_rows_was_changed", "children"),
            Input("data-was-changed", "children"),
            Input("order-was-changed", "data"),
            Input("alpha", "value"),
            State("debounced_bounds", "value"),
            State("map", "zoom"),
            State("column-dropdown", "value"),
            State("bins", "children"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "id"),
            State("max_rows", "children"),
            State({"type": "delete-btn", "index": dash.ALL}, "id"),
        )
        def add_data(
            currently_in_bounds2,
            colorpicker_values_list,
            is_numeric,
            file_removed,
            wms,
            wms_checked,
            new_file_added2,
            max_rows_was_changed,
            data_was_changed,
            order_was_changed,
            alpha,
            bounds,
            zoom,
            column,
            bins,
            colorpicker_ids,
            max_rows_component,
            delete_buttons,
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

            results = [add_data_func(path) for path in self.selected_files]
            debug_print("add_data2222", perf_counter() - t)

            data = list(itertools.chain.from_iterable([x[0] for x in results if x[0]]))
            out_alert = [x[1] for x in results if x[1]]
            rows_are_not_hidden = not any(x[2] for x in results)

            debug_print(
                "add_data ferdig etter",
                perf_counter() - t,
                len(self.loaded_data),
                len(self.concatted_data if self.concatted_data is not None else []),
                len(
                    {x for x in self.concatted_data["__file_path"]}
                    if self.concatted_data is not None
                    else []
                ),
            )
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
            Output("clicked-features", "data"),
            Output("clicked-ids", "data"),
            Output("alert4", "children"),
            Input("clicked-ids", "data"),
            Input("clear-table", "n_clicks"),
            Input({"type": "geojson", "filename": dash.ALL}, "n_clicks"),
            Input({"type": "table-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "table-btn", "index": dash.ALL}, "id"),
            State({"type": "geojson", "filename": dash.ALL}, "clickData"),
            State({"type": "geojson", "filename": dash.ALL}, "id"),
            State("clicked-features", "data"),
            # prevent_initial_call=True,
        )
        def display_feature_attributes(
            clicked_ids,
            clear_table,
            geojson_n_clicks,
            table_btn_n_clicks,
            table_btn_ids,
            features,
            feature_ids,
            clicked_features,
        ):
            triggered = dash.callback_context.triggered_id
            debug_print("display_feature_attributes", triggered)
            if triggered is None:
                clicked_ids = list(self.selected_features)
                clicked_features = list(self.selected_features.values())
                return clicked_features, clicked_ids, None
            if triggered == "clear-table":
                self.selected_features = {}
                return [], [], None

            if isinstance(triggered, dict) and triggered["type"] == "table-btn":
                clicked_path = get_index_if_clicks(table_btn_n_clicks, table_btn_ids)
                if clicked_path is None:
                    return dash.no_update
                data = self.concatted_data.filter(
                    pl.col("__file_path").str.contains(clicked_path)
                )
                clicked_ids = list(data["_unique_id"])
                clicked_features = data.drop("geometry").to_dicts()
                self.selected_features = dict(
                    zip(clicked_ids, clicked_features, strict=True)
                )
                return clicked_features, [], None

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
            feature = features[index]
            props = feature["properties"]
            clicked_ids = [x["_unique_id"] for x in clicked_features]
            if props["_unique_id"] not in clicked_ids:
                clicked_features.append(props)
                clicked_ids.append(props["_unique_id"])
                # else:
                #     pop_index = next(
                #         iter(
                #             i
                #             for i, x in enumerate(clicked_features)
                #             if x["_unique_id"] == props["_unique_id"]
                #         )
                #     )
                #     clicked_features.pop(pop_index)
                #     clicked_ids.pop(pop_index)
                self.selected_features = dict(
                    zip(clicked_ids, clicked_features, strict=True)
                )
            return clicked_features, clicked_ids, None

        @callback(
            Output("feature-table-rows", "columns"),
            Output("feature-table-rows", "data"),
            Output("feature-table-rows", "style_table"),
            Output("feature-table-rows", "hidden_columns"),
            Input("clicked-features", "data"),
            State("column-dropdown", "options"),
            State("feature-table-rows", "style_table"),
            # prevent_initial_call=True,
        )
        def update_table(data, column_dropdown, style_table):
            debug_print("update_table")
            if not data:
                return None, None, style_table | {"height": "1vh"}, None
            if column_dropdown is None:
                column_dropdown = self._get_column_dropdown_options(
                    list(self.loaded_data)
                )
            all_columns = {x["label"] for x in column_dropdown}
            if not self.splitted:
                all_columns = all_columns.difference({"split_index"})
            height = min(40, len(data) * 5 + 5)
            for x in data:
                x["id"] = x.pop("_unique_id")
            columns = [{"name": k, "id": k} for k in data[0].keys() if k in all_columns]
            return (
                columns,
                data,
                style_table | {"height": f"{height}vh"},
                ["id"],
            )

        @callback(
            Output("map", "zoom"),
            Output("map", "bounds"),
            Output("map", "center"),
            Input("map-zoom", "data"),
            State("map-bounds", "data"),
            State("map-center", "data"),
            prevent_initial_call=True,
        )
        def intermediate_update_bounds(zoom, bounds, center):
            """Update map bounds after short sleep because otherwise it's buggy."""
            time.sleep(0.1)
            if not zoom:
                return dash.no_update, dash.no_update, dash.no_update
            return zoom, bounds, center

        @callback(
            Output("map-bounds", "data"),
            Output("map-zoom", "data"),
            Output("map-center", "data"),
            Input("feature-table-rows", "active_cell"),
            State("viewport-container", "data"),
            prevent_initial_call=True,
        )
        def zoom_to_feature(active: dict, viewport):
            width = int(viewport["width"] * 0.7)
            height = int(viewport["height"] * 0.7)
            if active is None:
                return dash.no_update, dash.no_update, dash.no_update
            unique_id = active["row_id"]
            debug_print("\nzoom_to_feature")
            debug_print(unique_id)
            debug_print(self.concatted_data.filter(pl.col("_unique_id") == unique_id))
            minx, miny, maxx, maxy = (
                self.concatted_data.lazy()
                .filter(pl.col("_unique_id") == unique_id)
                .select("minx", "miny", "maxx", "maxy")
                .collect()
                .row(0)
            )
            center = ((miny + maxy) / 2, (minx + maxx) / 2)
            debug_print(center)
            bounds = [[miny, minx], [maxy, maxx]]
            debug_print(bounds)
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
            assert len(from_year) == 1
            assert len(to_year) == 1
            assert len(not_contains) == 1
            assert len(checked) == 1
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
                assert isinstance(wms_name, str), type(wms_name)
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

        def to_string(x):
            if isinstance(x, str):
                return f"'{x}'"
            return x

        txt = ", ".join(
            [
                f"{k}={to_string(v)}"
                for k, v in self.__dict__.items()
                if k
                not in [
                    "paths",
                    "app",
                    "currently_in_bounds",
                    "bounds_series",
                    "data",
                ]
                and not k.startswith("_")
            ]
        )
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
    print(map_width_px, map_height_px)

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
    if not any(n_clicks_list):
        return None
    triggered = dash.callback_context.triggered_id
    if not isinstance(triggered, dict):
        return None
    triggered_index = triggered["index"]
    index = next(
        iter(i for i, id_ in enumerate(ids) if id_["index"] == triggered_index)
    )
    n_clicks = n_clicks_list[index]
    if n_clicks == 0:
        return None
    return ids[index]["index"]
