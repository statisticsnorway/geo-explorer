import inspect
import itertools
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from pathlib import PurePath
from pathlib import PurePosixPath
from typing import ClassVar

import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import matplotlib
import numpy as np
import pandas as pd
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
from dash_extensions.javascript import Namespace
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from jenkspy import jenks_breaks
from shapely.errors import GEOSException
from shapely.geometry import Polygon

from .fs import LocalFileSystem

OFFWHITE = "#ebebeb"


def _buffer_box(box: Polygon) -> Polygon:
    try:
        return sg.to_gdf(box, 4326).to_crs(3035).buffer(100).to_crs(4326).union_all()
    except GEOSException:
        return box


def get_colorpicker_container(color_dict: dict[str, str]) -> html.Div:
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


def _read_files(explorer, paths: list[str]) -> None:
    read_func = partial(sg.read_geopandas, file_system=explorer.file_system)
    with ThreadPoolExecutor() as executor:
        more_data = list(executor.map(read_func, paths))
    for path, df in zip(paths, more_data, strict=True):
        explorer.loaded_data[path] = df.to_crs(4326).assign(
            _unique_id=lambda df: [
                f"{len(explorer.loaded_data)}_{j}" for j in range(len(df))
            ]
        )
        if explorer.splitted:
            explorer.loaded_data[path]["split_index"] = [
                f"{_get_name(path)} {i}" for i in range(len(df))
            ]


def _standardize_path(path: str | PurePosixPath) -> str:
    """Make sure delimiter is '/' and path ends without '/'."""
    return (
        str(path).replace("\\", "/").replace(r"\"", "/").replace("//", "/").rstrip("/")
    )


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
    return sg.get_bounds_series(paths, file_system=file_system).to_crs(4326)


def _get_button(item, isdir: bool, file_system):
    size = 15
    is_loadable = not file_system.isdir(item) or (
        item.endswith(".parquet")
        or all(x.endswith(".parquet") for x in file_system.ls(item))
    )
    if is_loadable:
        button = html.Button(
            "Load",
            id={
                "type": "load-parquet",
                "index": item,
            },
            className="load-button",
            n_clicks=0,
        )
    else:
        button = html.Button(
            "Load",
            id={
                "type": "load-parquet",
                "index": item,
            },
            className="load-button",
            n_clicks=0,
            style={
                "color": "rgba(0, 0, 0, 0)",
                "fillColor": "rgba(0, 0, 0, 0)",
                "backgroundColor": "rgba(0, 0, 0, 0)",
            },
            disabled=True,
        )
    txt_type = html.U if isdir else str
    return html.Div(
        [
            button,
            html.Button(
                txt_type(f"[DIR] {Path(item).name}" if isdir else Path(item).name),
                id={"type": "file-item", "index": item},
                className="path-button",
                style={
                    "padding-left": f"{int(size/5)}px",
                    "backgroundColor": "rgba(0, 0, 0, 0)",
                    "fillColor": "rgba(0, 0, 0, 0)",
                    "width": "70vh",
                }
                | ({"color": OFFWHITE} if not isdir else {"color": "#78b3e7"}),
                n_clicks=0,
                disabled=False if isdir else True,
            ),
        ],
        # style={"height": f"{int(size*2)}px"},
        className="button-container",
    )


def _list_dir(path, file_system):
    paths = list(file_system.ls(path))
    paths = [
        x
        for x in paths
        if file_system.isdir(x) or any(x.endswith(txt) for txt in [".parquet"])
    ]
    paths.sort()
    isdir_list = [file_system.isdir(x) for x in paths]
    return html.Ul(
        [
            html.Li(
                [
                    _get_button(item, isdir, file_system)
                    for item, isdir in zip(paths, isdir_list, strict=True)
                ]
            )
        ]
    )


sg.NorgeIBilderWms._url = "https://wms.geonorge.no/skwms1/wms.nib-prosjekter"


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
    _map_children = [
        dl.ScaleControl(position="bottomleft"),
        dl.MeasureControl(
            position="bottomright",
            primaryLengthUnit="meters",
        ),
    ]

    def __init__(
        self,
        start_dir: str = "/buckets",
        port=8055,
        data: list[str] | None = None,
        selected_features: list[str] | None = None,
        column: str | None = None,
        wms=None,
        center: tuple[float, float] | None = None,
        zoom: int = 10,
        nan_color: str = "#969696",
        nan_label: str = "Missing",
        color_dict: dict | None = None,
        file_system=None,
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
        self.column = column
        self.color_dict = color_dict or {}
        self.wms = wms or {}
        self.file_system = file_system or LocalFileSystem()
        self.nan_color = nan_color
        self.nan_label = nan_label
        self.splitted = splitted
        self.file_system = file_system
        self.selected_features: list[str] = (
            selected_features if selected_features is not None else []
        )
        self.currently_in_bounds: set[str] = set()
        self.bounds_series = GeoSeries()
        self.loaded_data: dict[str, GeoDataFrame] = {}
        self.selected_data: dict[str, int] = {}

        self.app = Dash(
            __name__,
            suppress_callback_exceptions=True,
            external_stylesheets=[dbc.themes.SOLAR],
            requests_pathname_prefix=f"/proxy/{self.port}/" if self.port else None,
            serve_locally=True,
            assets_folder="assets",
        )

        clicked_features = []

        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    html.Div(id="alert"),
                ),
                dbc.Row(
                    html.Div(id="new-file-added"),
                ),
                dbc.Row(
                    html.Div(id="new-file-added2", style={"display": "none"}),
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dl.Map(
                                    center=self.center,
                                    zoom=self.zoom,
                                    children=self._map_children
                                    + [
                                        dl.LayersControl(self._base_layers, id="lc"),
                                    ],
                                    id="map",
                                    style={"width": "100%", "height": "90vh"},
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
                                                "Split",
                                                style={
                                                    "fillColor": "white",
                                                    "color": "black",
                                                },
                                                id="splitter",
                                                n_clicks=1 if self.splitted else 0,
                                            ),
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "Export",
                                                        id="export",
                                                        style={
                                                            "color": "blue",
                                                            # "border": "none",
                                                            # "background": "none",
                                                            # "cursor": "pointer",
                                                        },
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
                                    ]
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
                                                            "width": "100%",
                                                            "overflow": "visible",
                                                        },
                                                        maxHeight=600,
                                                        clearable=True,
                                                    ),
                                                ],
                                            ),
                                            width=10,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                id="force-categorical",
                                            ),
                                            width=2,
                                        ),
                                    ]
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
                                                    "width": "100%",
                                                    "overflow": "visible",
                                                },
                                                maxHeight=300,
                                                clearable=False,
                                            )
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
                                            )
                                        ),
                                    ]
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
                                    html.Div(id="remove-buttons"),
                                ),
                                dbc.Row(id="colorpicker-container"),
                            ],
                            style={
                                "height": "90vh",
                                "overflow": "scroll",
                            },
                        ),
                    ],
                ),
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
                            html.Div(id="feature-table-container"),
                            style={"width": "100%", "height": "auto"},
                            width=11,
                        ),
                    ],
                    style={
                        "height": "auto",
                        "overflow": "visible",
                    },
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.Br(),
                                    html.H2("File Browser"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.Button(
                                                    "⬆️ Go Up",
                                                    id="up-button",
                                                    style={"width": "10vh"},
                                                )
                                            ),
                                            dbc.Col(
                                                dcc.Input(
                                                    self.start_dir,
                                                    id="path-display",
                                                    style={
                                                        "width": "70vh",
                                                    },
                                                )
                                            ),
                                        ]
                                    ),
                                    html.Br(),
                                    html.Div(
                                        id="file-list",
                                        style={
                                            "font-size": 12,
                                            "width": "100%",
                                            "height": "70vh",
                                            "overflow": "scroll",
                                        },
                                    ),
                                ]
                            ),
                            # width=4,
                        ),
                    ],
                    style={"width": "90vh"},
                ),
                dcc.Store(id="is_splitted", data=False),
                html.Div(id="currently-in-bounds", style={"display": "none"}),
                html.Div(id="currently-in-bounds2", style={"display": "none"}),
                html.Div(id="file-removed", style={"display": "none"}),
                html.Div(id="dummy-output", style={"display": "none"}),
                html.Div(id="bins", style={"display": "none"}),
                html.Div(False, id="is-numeric", style={"display": "none"}),
                dcc.Store(id="clicked-features", data=clicked_features),
                dcc.Store(id="clicked-ids", data=self.selected_features),
                dcc.Store(id="current-path", data=self.start_dir),
            ],
            fluid=True,
        )

        error_mess = "'data' must be a list of file paths or a dict of GeoDataFrames."
        bounds_series_dict = {}
        for x in data or []:
            if isinstance(x, dict):
                for key, value in x.items():
                    if not isinstance(value, GeoDataFrame):
                        raise ValueError(error_mess)
                    key = _standardize_path(key)
                    self.loaded_data[key] = value.to_crs(4326)
                    self.selected_data[key] = 0
                    bounds_series_dict[key] = shapely.box(
                        *self.loaded_data[key].total_bounds
                    )
            elif isinstance(x, (str | os.PathLike | PurePath)):
                self.selected_data[_standardize_path(x)] = 0
            else:
                raise ValueError(error_mess)

        self.bounds_series = GeoSeries(bounds_series_dict)

        if self.selected_data:
            child_paths = _get_child_paths(
                [x for x in self.selected_data if x not in bounds_series_dict],
                self.file_system,
            )
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
                [x for x in self.selected_data if x not in bounds_series_dict],
            )

            # dataframe dicts as input data will now be sorted first because they were added to loaded_data first.
            # now to get back original order
            loaded_data_sorted = {}
            for x in data:
                if isinstance(x, dict):
                    for key in x:
                        key = _standardize_path(key)
                        loaded_data_sorted[key] = self.loaded_data[key].assign(
                            _unique_id=lambda df: [
                                f"{len(loaded_data_sorted)}_{j}" for j in range(len(df))
                            ]
                        )
                else:
                    x = _standardize_path(x)
                    loaded_data_sorted[x] = self.loaded_data[x].assign(
                        _unique_id=lambda df: [
                            f"{len(loaded_data_sorted)}_{j}" for j in range(len(df))
                        ]
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
        self.app.run(debug=debug, port=self.port, jupyter_mode=jupyter_mode)

    def _register_callbacks(self) -> None:

        @callback(
            Output("export-text", "children"),
            Output("export-view", "is_open"),
            Input("export", "n_clicks"),
            Input("file-removed", "children"),
            # Input("close-export", "n_clicks"),
            State("map", "bounds"),
            State("map", "zoom"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "value"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def export(
            export_clicks,
            remove,
            # close_exort_clicks,
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
                ]
                and not (k in defaults and v == defaults[k])
            } | {"zoom": zoom, "center": center, "color_dict": color_dict}
            if self.selected_data:
                data["data"] = data.pop("selected_data")
            else:
                data.pop("selected_data")

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
            Output("file-list", "children"),
            Input("current-path", "data"),
        )
        def update_file_list(path):
            return _list_dir(path, self.file_system)

        @callback(
            Output("current-path", "data"),
            Output("path-display", "value"),
            Input({"type": "file-item", "index": dash.ALL}, "n_clicks"),
            Input("up-button", "n_clicks"),
            Input("path-display", "value"),
            State({"type": "file-item", "index": dash.ALL}, "id"),
            State("current-path", "data"),
            prevent_initial_call=True,
        )
        def handle_click(load_parquet, up_button_clicks, path, ids, current_path):
            triggered = dash.callback_context.triggered_id
            path = _standardize_path(path)
            if triggered == "path-display":
                return path, path
            if triggered == "up-button":
                current_path = str(Path(current_path).parent)
                return current_path, current_path
            elif not any(load_parquet) or not triggered:
                return dash.no_update, dash.no_update
            selected_path = triggered["index"]
            selected_path = _standardize_path(selected_path)
            return selected_path, selected_path

        @callback(
            Output("remove-buttons", "children"),
            # Input("new-file-added", "children"),
            Input("alert", "children"),
            Input("file-removed", "children"),
        )
        def render_items(new_file_added, file_removed):
            return [
                html.Div(
                    [
                        html.Button(
                            "✓",
                            id={
                                "type": "checked-btn",
                                "index": path,
                            },
                            n_clicks=n_clicks,
                            style={
                                "color": "rgba(0, 0, 0, 0)",
                                # "fillColor": ("blue" if n_clicks % 2 == 0 else "white"),
                                "background": (
                                    "#5ca3ff" if n_clicks % 2 == 0 else OFFWHITE
                                ),
                                # "cursor": "pointer",
                            },
                        ),
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
                            },
                            className="x-button",
                        ),
                        html.Span(path),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "marginBottom": "5px",
                    },
                )
                for path, n_clicks in self.selected_data.items()
            ]

        @callback(
            Output("file-removed", "children"),
            Input({"type": "delete-btn", "index": dash.ALL}, "n_clicks"),
            State({"type": "delete-btn", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def delete_item(n_clicks_list, delete_ids):
            if not any(n_clicks_list):
                return dash.no_update
            triggered = dash.callback_context.triggered_id
            if triggered and triggered["type"] == "delete-btn":
                triggered_index = triggered["index"]
                index = next(
                    iter(
                        i
                        for i, id_ in enumerate(delete_ids)
                        if id_["index"] == triggered_index
                    )
                )
                path_to_remove = delete_ids[index]["index"]
                self.selected_data.pop(path_to_remove)
                for path in list(self.loaded_data):
                    if path_to_remove in path:
                        del self.loaded_data[path]
                self.bounds_series = self.bounds_series[
                    lambda x: ~x.index.str.contains(path_to_remove)
                ]
            return 1

        @callback(
            Output("splitter", "n_clicks"),
            Output("is_splitted", "data"),
            Output("column-dropdown", "value"),
            Input("splitter", "n_clicks"),
            Input("column-dropdown", "value"),
        )
        def is_splitted(n_clicks: int, column):
            triggered = dash.callback_context.triggered_id
            is_splitted: bool = n_clicks % 2 == 1 and not (
                triggered == "column-dropdown" and column is None
            )
            if is_splitted:
                column = "split_index"
            elif column == "split_index":
                column = None
            else:
                column = dash.no_update
            n_clicks = 1 if is_splitted else 0
            return n_clicks, is_splitted, column

        @callback(
            Output("new-file-added", "children"),
            Output("new-file-added", "style"),
            Input({"type": "load-parquet", "index": dash.ALL}, "n_clicks"),
            Input("is_splitted", "data"),
            State({"type": "file-item", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def append_path(load_parquet, is_splitted, ids):
            triggered = dash.callback_context.triggered_id
            if triggered == "is_splitted":
                if not is_splitted:
                    return dash.no_update, dash.no_update
                self.splitted = True
                for key, df in self.loaded_data.items():
                    self.loaded_data[key]["split_index"] = [
                        f"{_get_name(key)} {i}" for i in range(len(df))
                    ]
                return 1, {"display": "none"}
            if not any(load_parquet) or not triggered:
                return dash.no_update, dash.no_update
            try:
                selected_path = triggered["index"]
            except Exception as e:
                raise type(e)(f"{e}: {triggered}")
            selected_path = _standardize_path(selected_path)
            if selected_path in self.selected_data:
                return dash.no_update
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
            self.selected_data[selected_path] = 0
            self.bounds_series = pd.concat(
                [
                    self.bounds_series,
                    more_bounds,
                ]
            )
            return 1, {"display": "none"}

        @callback(
            Output("currently-in-bounds", "children"),
            Input("map", "bounds"),
            Input("new-file-added", "children"),
            Input("file-removed", "children"),
            # prevent_initial_call=True,
        )
        def get_files_in_bounds(bounds, file_added, file_removed):
            box = shapely.box(*self._nested_bounds_to_bounds(bounds))
            box = _buffer_box(box)
            files_in_bounds = sg.sfilter(self.bounds_series, box)
            currently_in_bounds = set(files_in_bounds.index)
            missing = list(
                {path for path in files_in_bounds.index if path not in self.loaded_data}
            )
            if missing:
                _read_files(self, missing)
            return list(currently_in_bounds)

        @callback(
            Output("column-dropdown", "options"),
            Input("currently-in-bounds", "children"),
            prevent_initial_call=True,
        )
        def update_column_dropdown_options(currently_in_bounds):
            return self._get_column_dropdown_options(currently_in_bounds)

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
            Input("currently-in-bounds", "children"),
            Input("file-removed", "children"),
            State("map", "bounds"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "value"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "id"),
            State("bins", "children"),
            State("is_splitted", "data"),
            prevent_initial_call=True,
        )
        def get_column_value_color_dict(
            column,
            cmap: str,
            k: int,
            force_categorical_clicks: int,
            currently_in_bounds,
            file_removed,
            bounds,
            colorpicker_values_list,
            colorpicker_ids,
            bins,
            is_splitted,
        ):
            triggered = dash.callback_context.triggered_id

            if not is_splitted and self.splitted:
                colorpicker_ids, colorpicker_values_list = [], []
                self.splitted = is_splitted

            column_values = [x["column_value"] for x in colorpicker_ids]
            default_colors = list(sg.maps.map._CATEGORICAL_CMAP.values())

            if column is None or not any(
                column in df for df in self.loaded_data.values()
            ):
                color_dict = dict(
                    zip(column_values, colorpicker_values_list, strict=True)
                )

                if self.color_dict:
                    color_dict |= self.color_dict

                color_dict = {
                    key: color
                    for key, color in color_dict.items()
                    if any(key in x for x in self.selected_data)
                }

                default_colors = [
                    x for x in default_colors if x not in set(color_dict.values())
                ]

                new_values = [
                    _get_name(value)
                    for value in self.selected_data
                    if _get_name(value) not in color_dict
                ]
                if len(color_dict) < len(default_colors):
                    default_colors = default_colors[
                        len(color_dict) : min(
                            len(self.selected_data), len(default_colors)
                        )
                    ]
                new_colors = default_colors + [
                    _random_color()
                    for _ in range(len(new_values) - len(default_colors))
                ]

                try:
                    color_dict = color_dict | dict(
                        zip(new_values, new_colors, strict=True)
                    )
                except ValueError as e:
                    raise ValueError(f"{e}: {new_values} - {new_colors}") from e

                return (
                    get_colorpicker_container(color_dict),
                    None,
                    False,
                    dash.no_update,
                    currently_in_bounds,
                )

            box = shapely.box(*self._nested_bounds_to_bounds(bounds))
            box = _buffer_box(box)
            values = pd.concat(
                [
                    sg.sfilter(df[[column, df.geometry.name]], box)[column]
                    for df in self.loaded_data.values()
                    if column in df
                ],
                ignore_index=True,
            ).dropna()

            if not pd.api.types.is_numeric_dtype(values):
                force_categorical_button = None
            elif (force_categorical_clicks or 0) % 2 == 0:
                force_categorical_button = html.Button(
                    "Force categorical",
                    n_clicks=force_categorical_clicks,
                    style={
                        "fillColor": "white",
                        "color": "black",
                    },
                )
            else:
                force_categorical_button = html.Button(
                    "Force categorical",
                    n_clicks=force_categorical_clicks,
                    style={
                        "fillColor": "black",
                        "color": "white",
                    },
                )
            is_numeric = (
                force_categorical_clicks or 0
            ) % 2 == 0 and pd.api.types.is_numeric_dtype(values)
            if is_numeric:
                series = pd.concat(
                    [
                        df[column]
                        for path, df in self.loaded_data.items()
                        if any(x in path for x in self.selected_data) and column in df
                    ]
                ).dropna()
                if len(series.dropna().unique()) <= k:
                    bins = list(series.dropna().unique())
                else:
                    bins = jenks_breaks(series, n_classes=k)

                if column_values is not None and triggered in [
                    "map",
                    "currently-in-bounds",
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
                        f"{round(min(series), 1)} - {rounded_bins[0]}": colors_[0],
                        **{
                            f"{start} - {stop}": colors_[i + 1]
                            for i, (start, stop) in enumerate(
                                itertools.pairwise(rounded_bins[1:-1])
                            )
                        },
                        f"{rounded_bins[-1]} - {round(max(series), 1)}": colors_[-1],
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

                unique_values = values.unique()
                new_values = [
                    value for value in unique_values if value not in column_values
                ]
                existing_values = [
                    value for value in unique_values if value in column_values
                ]
                default_colors = [
                    x for x in default_colors if x not in set(color_dict.values())
                ]
                colors = default_colors[
                    len(existing_values) : min(len(unique_values), len(default_colors))
                ]
                colors = colors + [
                    _random_color() for _ in range(len(new_values) - len(colors))
                ]
                color_dict = color_dict | dict(
                    zip(
                        new_values,
                        colors,
                        strict=True,
                    )
                )
                bins = None

            if color_dict.get(self.nan_label, self.nan_color) != self.nan_color:
                self.nan_color = color_dict[self.nan_label]

            elif self.nan_label not in color_dict and (
                values.isna().any()
                or any(column not in df for df in self.loaded_data.values())
            ):
                color_dict[self.nan_label] = self.nan_color

            if self.color_dict:
                color_dict |= self.color_dict

            return (
                get_colorpicker_container(color_dict),
                bins,
                is_numeric,
                force_categorical_button,
                currently_in_bounds,
            )

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
            Output({"type": "checked-btn", "index": dash.ALL}, "style"),
            Input({"type": "checked-btn", "index": dash.ALL}, "n_clicks"),
            Input({"type": "checked-btn", "index": dash.ALL}, "id"),
        )
        def update_clicks(n_clicks_list, ids):
            for n_clicks, id_ in zip(n_clicks_list, ids, strict=True):
                path = id_["index"]
                self.selected_data[path] = n_clicks
            return [
                {
                    # "display": "none",
                    "color": (
                        "rgba(0, 0, 0, 0)"
                    ),  # if n_clicks % 2 == 0 else OFFWHITE),
                    "background": ("#5ca3ff" if n_clicks % 2 == 0 else OFFWHITE),
                }
                for n_clicks in n_clicks_list
            ]

        @callback(
            Output("lc", "children"),
            Output("alert", "children"),
            Input("currently-in-bounds2", "children"),
            Input({"type": "colorpicker", "column_value": dash.ALL}, "value"),
            Input("is-numeric", "children"),
            Input("file-removed", "children"),
            Input("clear-table", "n_clicks"),
            Input("wms-items", "children"),
            Input("wms-checklist", "value"),
            Input("new-file-added2", "children"),
            Input({"type": "checked-btn", "index": dash.ALL}, "style"),
            State("map", "bounds"),
            State("map", "zoom"),
            State({"type": "geojson-overlay", "filename": dash.ALL}, "checked"),
            State("column-dropdown", "value"),
            State("bins", "children"),
            State("clicked-ids", "data"),
            State({"type": "colorpicker", "column_value": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def add_data(
            currently_in_bounds,
            colorpicker_values_list,
            is_numeric,
            file_removed,
            clear_table,
            wms,
            wms_checked,
            new_file_added2,
            checked_buttons,
            bounds,
            zoom,
            is_checked,
            column,
            bins,
            clicked_ids,
            colorpicker_ids,
        ):
            column_values = [x["column_value"] for x in colorpicker_ids]
            color_dict = dict(zip(column_values, colorpicker_values_list, strict=True))

            box = shapely.box(*self._nested_bounds_to_bounds(bounds))
            box = _buffer_box(box)
            data = []
            filenames = []

            wms_layers = []
            for wms_name, wms_obj in self.wms.items():
                if wms_name not in wms_checked:
                    continue
                tiles = wms_obj._filter_tiles(box)["name"]
                for tile in tiles:
                    wms_layers.append(
                        dl.Overlay(
                            dl.WMSTileLayer(
                                url=wms_obj._url,
                                layers=tile,
                                format="image/png",
                                transparent=True,
                            ),
                            name=tile,
                            checked=wms_obj.checked,
                        )
                    )

            ns = Namespace("onEachFeatureToggleHighlight", "default")

            if is_numeric:
                color_dict = {i: color for i, color in enumerate(color_dict.values())}

            out_alert = []
            for path, n_clicks in self.selected_data.items():
                if path in self.loaded_data:
                    df = self.loaded_data[path]
                else:
                    matches = [
                        df for key, df in self.loaded_data.items() if path in key
                    ]
                    if not matches:
                        continue
                    df = pd.concat(matches)

                df = sg.sfilter(df, box)
                if n_clicks:
                    checked = n_clicks % 2 == 0
                else:
                    checked: bool = len(df) < 35_000  #  and not n_clicks
                    if not checked:
                        self.selected_data[path] = 1
                if not checked and not n_clicks:
                    out_alert.append(
                        dbc.Alert(
                            html.Div(
                                [
                                    html.Span(
                                        f"Layer '{Path(path).name}' was set as unchecked because it has too many ({len(df)}) rows in the current map bounds."
                                    ),
                                    html.Br(),
                                    html.Span(
                                        "You can view the data by checking the checkbox in the top right of the map. "
                                    ),
                                    html.Br(),
                                    html.Span(
                                        "But preferably zoom in first to avoid crashing the page."
                                    ),
                                    html.Br(),
                                ]
                            ),
                            color="warning",
                            dismissable=True,
                        )
                    )

                if column is not None and column in df and not is_numeric:
                    df["_color"] = df[column].map(color_dict)
                elif column is not None and column in df:
                    notnas = df[df[column].notna()]
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
                    assert len(conditions) == len(color_dict), (
                        (conditions),
                        (color_dict),
                        (bins),
                    )
                    choices = np.arange(len(conditions)) if bins is not None else None
                    try:
                        notnas["_color"] = [
                            color_dict[x] for x in np.select(conditions, choices)
                        ]
                    except KeyError as e:
                        raise KeyError(e, color_dict, conditions, choices, bins) from e
                    df = pd.concat([notnas, df[df[column].isna()]])
                if not any(path in x for x in currently_in_bounds):
                    filenames.append(path)
                    data.append(
                        dl.Overlay(
                            dl.GeoJSON(id={"type": "geojson", "filename": path}),
                            name=_get_name(path),
                            checked=checked,
                            id={"type": "geojson-overlay", "filename": path},
                        )
                    )
                    continue
                if column is not None and column not in df:
                    filenames.append(path)
                    data.append(
                        dl.Overlay(
                            dl.GeoJSON(
                                data=df.__geo_interface__,
                                style={
                                    "color": self.nan_color,
                                    "fillColor": self.nan_color,
                                    "weight": 2,
                                    "fillOpacity": 0.7,
                                },
                                onEachFeature=ns("yellowIfHighlighted"),
                                pointToLayer=ns("pointToLayerCircle"),
                                hideout=dict(
                                    circleOptions=dict(
                                        fillOpacity=1, stroke=False, radius=5
                                    ),
                                ),
                                id={"type": "geojson", "filename": path},
                            ),
                            name=_get_name(path),
                            checked=checked,
                            id={"type": "geojson-overlay", "filename": path},
                        )
                    )
                elif column is not None:
                    for color_ in df["_color"].dropna().unique():
                        filenames.append(path + color_)
                    filenames.append(path + "_nan")
                    data.append(
                        dl.Overlay(
                            dl.LayerGroup(
                                [
                                    dl.GeoJSON(
                                        data=(
                                            df[df["_color"] == color_]
                                        ).__geo_interface__,
                                        style={
                                            "color": color_,
                                            "fillColor": color_,
                                            "weight": 2,
                                            "fillOpacity": 0.7,
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
                                            "color": self.nan_color,
                                            "fillColor": self.nan_color,
                                            "weight": 2,
                                            "fillOpacity": 0.7,
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
                            checked=checked,
                            id={"type": "geojson-overlay", "filename": path},
                        )
                    )
                else:
                    # no column
                    filenames.append(path)
                    color = color_dict[_get_name(path)]
                    data.append(
                        dl.Overlay(
                            dl.GeoJSON(
                                data=df.__geo_interface__,
                                style={
                                    "color": color,
                                    "fillColor": color,
                                    "weight": 2,
                                    "fillOpacity": 0.7,
                                },
                                onEachFeature=ns("yellowIfHighlighted"),
                                pointToLayer=ns("pointToLayerCircle"),
                                hideout=dict(
                                    circleOptions=dict(
                                        fillOpacity=1, stroke=False, radius=5
                                    ),
                                ),
                                id={"type": "geojson", "filename": path},
                            ),
                            name=_get_name(path),
                            checked=checked,
                            id={"type": "geojson-overlay", "filename": path},
                        )
                    )

            return (
                (self._base_layers + wms_layers + data),
                (out_alert if out_alert else None),
            )

        @callback(
            Output("clicked-features", "data"),
            Output("clicked-ids", "data"),
            Input("clicked-ids", "data"),
            Input("clear-table", "n_clicks"),
            Input({"type": "geojson", "filename": dash.ALL}, "n_clicks"),
            State({"type": "geojson", "filename": dash.ALL}, "clickData"),
            State({"type": "geojson", "filename": dash.ALL}, "id"),
            State("clicked-features", "data"),
            prevent_initial_call=True,
        )
        def display_feature_attributes(
            clicked_ids,
            clear_table,
            n_clicks,
            features,
            feature_ids,
            clicked_features,
        ):
            triggered = dash.callback_context.triggered_id
            if triggered == "clear-table":
                return [], []
            if not features or not any(features):
                return dash.no_update, dash.no_update

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
            else:
                pop_index = next(
                    iter(
                        i
                        for i, x in enumerate(clicked_features)
                        if x["_unique_id"] == props["_unique_id"]
                    )
                )
                clicked_features.pop(pop_index)
                clicked_ids.pop(pop_index)
            return clicked_features, clicked_ids

        @callback(
            Output("feature-table-container", "children"),
            Input("clicked-features", "data"),
            State("column-dropdown", "options"),
            State("currently-in-bounds", "children"),
        )
        def update_table(data, column_dropdown, currently_in_bounds):
            if not data:
                return "No features clicked."
            if column_dropdown is None:
                column_dropdown = self._get_column_dropdown_options(
                    list(self.loaded_data)
                )
            all_columns = {x["label"] for x in column_dropdown}
            if not self.splitted:
                all_columns = all_columns.difference({"split_index"})
            columns = [{"name": k, "id": k} for k in data[0].keys() if k in all_columns]
            return html.Div(
                [
                    dash_table.DataTable(
                        columns=columns,
                        data=data,
                        style_header={
                            "backgroundColor": "#2f2f2f",
                            "color": "white",
                            "fontWeight": "bold",
                        },
                        style_data={
                            "backgroundColor": OFFWHITE,
                            "color": "black",
                        },
                        style_table={"overflowX": "auto"},
                    ),
                ]
            )

        self.app.clientside_callback(
            """
            function(ids) {
                window.selectedFeatureIds = ids || [];
                // Reset all highlighted features if list is empty
                if (window.selectedFeatureIds.length === 0 && window.leafletMap) {
                    window.leafletMap.eachLayer(function(layer) {
                        if (layer.setStyle && layer._originalStyle) {
                            layer.setStyle(layer._originalStyle);
                        }
                    });
                }
                return null;
            }
            """,
            Output("dummy-output", "children"),
            Input("clicked-ids", "data"),
        )

    def _get_column_dropdown_options(self, currently_in_bounds):
        columns = set(
            itertools.chain.from_iterable(
                set(
                    self.loaded_data[path].columns.difference(
                        {self.loaded_data[path].geometry.name, "_unique_id"}
                    )
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
                .buffer(165_000 / (self.zoom**1.5))
                .to_crs(4326)
                .total_bounds
            )
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
            ]
        )
        return f"{self.__class__.__name__}({txt})"
