import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input
from dash import Output
from dash import State
from dash import callback
from dash import dcc
from dash import html
from dash.development.base_component import Component
from fsspec.spec import AbstractFileSystem

from .utils import _clicked_button_style
from .utils import _standardize_path
from .utils import _unclicked_button_style
from .utils import get_button_with_tooltip


class FileBrowser:
    def __init__(
        self,
        start_dir: str,
        favorites: list[str] | None = None,
        file_system: AbstractFileSystem | None = None,
    ) -> None:
        self.start_dir = _standardize_path(start_dir)
        self.file_system = file_system
        self.favorites = (
            [_standardize_path(x) for x in favorites] if favorites is not None else []
        )
        self._history = [self.start_dir]
        self._register_callbacks()

    def get_file_browser_components(self, width: str = "140vh") -> list[Component]:
        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H2("File Browser"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Button(
                                                    "🡑 Go Up",
                                                    id="up-button",
                                                    style=_unclicked_button_style()
                                                    | {"width": "10vh"},
                                                ),
                                                *get_button_with_tooltip(
                                                    "Recursive",
                                                    id="recursive",
                                                    n_clicks=0,
                                                    tooltip_text="Note that recursive file search might be extremely slow if there are many files and subfolders",
                                                ),
                                                dcc.Dropdown(
                                                    id="favorites-dropdown",
                                                    placeholder="Favorites",
                                                    options=[
                                                        _standardize_path(path)
                                                        for path in self.favorites
                                                    ],
                                                    clearable=False,
                                                    className="expandable-dropdown-left-aligned",
                                                ),
                                                dcc.Dropdown(
                                                    id="history-dropdown",
                                                    placeholder="History",
                                                    options=self._history,
                                                    clearable=False,
                                                    className="expandable-dropdown-left-aligned",
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "gap": "0.2rem",
                                                "alignItems": "center",
                                            },
                                        ),
                                        dbc.Col(
                                            get_button_with_tooltip(
                                                "Case sensitive",
                                                id="case-sensitive",
                                                n_clicks=1,
                                                tooltip_text="Whether lower/upper case matters in subtext search",
                                            ),
                                            className="align-right",
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Input(
                                                self.start_dir,
                                                id="current-path",
                                                debounce=1,
                                                className="expandable-input-left-aligned",
                                            ),
                                        ),
                                        dbc.Col(
                                            dcc.Input(
                                                placeholder="Search for files by substring (use '|' as OR and ',' as AND)...",
                                                id="filename-filter",
                                                debounce=0.5,
                                                className="expandable-input-right-aligned",
                                            ),
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    html.Div(id="file-list-alert"),
                                ),
                                dbc.Row(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                className="align-right",
                                            ),
                                        ]
                                    ),
                                ),
                                html.Br(),
                                dbc.Row(
                                    html.Div(
                                        html.Table(
                                            [
                                                html.Thead(
                                                    html.Tr(
                                                        [
                                                            html.Th("Load"),
                                                            html.Th(
                                                                html.Button(
                                                                    "File Name 🡑🡓",
                                                                    id={
                                                                        "type": "sort_by",
                                                                        "key": "name",
                                                                    },
                                                                    n_clicks=0,
                                                                )
                                                            ),
                                                            html.Th(
                                                                html.Button(
                                                                    "Timestamp 🡑🡓",
                                                                    id={
                                                                        "type": "sort_by",
                                                                        "key": "updated",
                                                                    },
                                                                    n_clicks=0,
                                                                )
                                                            ),
                                                            html.Th(
                                                                html.Button(
                                                                    "Size (MB) 🡑🡓",
                                                                    id={
                                                                        "type": "sort_by",
                                                                        "key": "size",
                                                                    },
                                                                    n_clicks=0,
                                                                )
                                                            ),
                                                        ]
                                                    )
                                                ),
                                                html.Tbody(
                                                    id="file-list",
                                                ),
                                            ]
                                        )
                                    ),
                                    style={
                                        "font-size": 12,
                                        "width": "100%",
                                        "height": "70vh",
                                        "overflow": "scroll",
                                    },
                                    className="scroll-container",
                                ),
                            ]
                        ),
                    ),
                ],
                style={
                    "width": width,
                    "border": "1px solid #ccc",
                    "margin-bottom": "7px",
                    "margin-top": "7px",
                    "margin-left": "0px",
                    "margin-right": "0px",
                    "borderRadius": "3px",
                },
                className="scroll-container",
            ),
            dcc.Store(id="file-data-dict", data=None),
            html.Div(),
        ]

    def _register_callbacks(self):

        @callback(
            Output("favorites-dropdown", "value"),
            Output("current-path", "value", allow_duplicate=True),
            Input("favorites-dropdown", "value"),
            State("current-path", "value"),
            prevent_initial_call=True,
        )
        def go_to_favorite(clicked_favorite: str | None, current_path: str):
            if clicked_favorite and clicked_favorite == current_path:
                return None, dash.no_update
            elif clicked_favorite:
                return None, clicked_favorite
            else:
                return None, self.start_dir

        @callback(
            Output("history-dropdown", "value"),
            Output("current-path", "value", allow_duplicate=True),
            Input("history-dropdown", "value"),
            State("current-path", "value"),
            prevent_initial_call=True,
        )
        def go_to_clicked_from_history(clicked: str | None, current_path: str):
            if clicked and clicked == current_path:
                return None, dash.no_update
            elif clicked:
                return None, clicked
            else:
                return None, self.start_dir

        @callback(
            Output("case-sensitive", "style"),
            Input("case-sensitive", "n_clicks"),
        )
        def update_case_button_style(n_clicks):
            if (n_clicks or 0) % 2 == 1:
                return _clicked_button_style()
            else:
                return _unclicked_button_style()

        @callback(
            Output("recursive", "style"),
            Input("recursive", "n_clicks"),
        )
        def update_recursive_button(n_clicks):
            if (n_clicks or 0) % 2 == 1:
                return _clicked_button_style()
            else:
                return _unclicked_button_style()

        @callback(
            Output("current-path", "value", allow_duplicate=True),
            Input({"type": "file-path", "index": dash.ALL}, "n_clicks"),
            Input("up-button", "n_clicks"),
            Input("current-path", "value"),
            State({"type": "file-path", "index": dash.ALL}, "id"),
            prevent_initial_call=True,
        )
        def handle_click(load_parquet, up_button_clicks, current_path, ids):
            triggered = dash.callback_context.triggered_id
            current_path = _standardize_path(current_path)
            if triggered == "current-path" and not current_path:
                time.sleep(1)
                return self.start_dir
            if triggered == "current-path":
                return current_path
            if triggered == "up-button":
                return str(Path(current_path).parent)
            elif not any(load_parquet) or not triggered:
                return dash.no_update
            selected_path = triggered["index"]
            selected_path = _standardize_path(selected_path)
            return selected_path

        @callback(
            Output("file-data-dict", "data"),
            Output("file-list", "children"),
            Output("file-list-alert", "children"),
            Output({"type": "sort_by", "key": dash.ALL}, "n_clicks"),
            Output("history-dropdown", "options"),
            Input("current-path", "value"),
            Input("filename-filter", "value"),
            Input("case-sensitive", "n_clicks"),
            Input("recursive", "n_clicks"),
            Input({"type": "sort_by", "key": dash.ALL}, "n_clicks"),
            Input({"type": "sort_by", "key": dash.ALL}, "id"),
            State("file-list", "children"),
            State("file-data-dict", "data"),
        )
        def update_file_list(
            path,
            search_word,
            case_sensitive,
            recursive,
            sort_by_clicks,
            sort_by_ids,
            file_list,
            file_data_dict,
        ):
            triggered = dash.callback_context.triggered_id
            path = _standardize_path(path)

            if isinstance(triggered, dict) and triggered["type"] == "sort_by":
                sort_by_key = triggered["key"]
                sort_by_clicks = [
                    clicks if x["key"] == sort_by_key else 0
                    for x, clicks in zip(sort_by_ids, sort_by_clicks, strict=True)
                ]
                alert = None
            else:
                file_data_dict, file_list, alert = _list_dir(
                    path, search_word, case_sensitive, recursive, self.file_system
                )
                if sum(sort_by_clicks):
                    sort_by_key = next(
                        iter(
                            x["key"]
                            for x, clicks in zip(
                                sort_by_ids, sort_by_clicks, strict=True
                            )
                            if clicks
                        )
                    )
            if sum(sort_by_clicks):

                def try_to_sort(x):
                    try:
                        return x[0][sort_by_key]
                    except KeyError:
                        return "sort this string last" * 100

                sorted_pairs = sorted(
                    zip(file_data_dict, file_list, strict=False),
                    key=try_to_sort,
                )
                if sum(sort_by_clicks) % 2 == 0:
                    sorted_pairs = list(reversed(sorted_pairs))
                file_data_dict = [x[0] for x in sorted_pairs]
                file_list = [x[1] for x in sorted_pairs]

            self._history = list(dict.fromkeys([path, *self._history]))

            return (file_data_dict, file_list, alert, sort_by_clicks, self._history[1:])


def _list_dir(
    path: str, containing: str, case_sensitive: bool, recursive: bool, file_system
):
    containing = containing or ""
    containing = [txt.strip() for txt in containing.split(",") if txt.strip()]
    if (case_sensitive or 0) % 2 == 0:

        def _contains(path):
            if not containing:
                return True
            return all(
                any(
                    txt.strip().lower() in path.lower()
                    for txt in x.split("|")
                    if txt.strip()
                )
                for x in containing
            )

    else:

        def _contains(path):
            if not containing:
                return True
            return all(
                any(txt.strip() in path for txt in x.split("|") if txt.strip())
                for x in containing
            )

    if (recursive or 0) % 2 == 0:

        def _ls(path):
            return file_system.ls(path, detail=True)

    else:

        def _ls(path):
            path = str(Path(path) / "**")
            return _try_glob(path, file_system)

    try:
        paths = _ls(path)
    except Exception as e:
        try:
            paths = _try_glob(path, file_system)
        except Exception:
            return (
                [],
                [],
                dbc.Alert(
                    f"Couldn't list files in {path}. {type(e)}: {e}",
                    color="warning",
                    dismissable=True,
                ),
            )

    if not paths:
        paths = _try_glob(path, file_system)

    if isinstance(paths, dict):
        paths = list(paths.values())

    def is_dir_or_is_partitioned_parquet(x) -> bool:
        return x["type"] == "directory" or any(
            x["name"].endswith(txt) for txt in [".parquet"]
        )

    paths = [
        x
        for x in paths
        if isinstance(x, dict)
        and _contains(x["name"])
        and is_dir_or_is_partitioned_parquet(x)
        and Path(path).parts != Path(x["name"]).parts
    ]

    paths.sort(key=lambda x: x["name"])
    isdir_list = [x["type"] == "directory" for x in paths]

    partitioned = {
        i: x
        for i, x in enumerate(paths)
        if x["type"] == "directory"
        and any(
            str(x).endswith(".parquet") for x in (x["name"], *Path(x["name"]).parents)
        )
    }

    def get_summed_size_and_latest_timestamp_in_subdirs(
        x,
    ) -> tuple[float, datetime.datetime]:
        file_info = _try_glob(str(Path(x["name"]) / "**/*.parquet"), file_system)

        if isinstance(file_info, dict):
            file_info = list(file_info.values())

        file_info = [
            x for x in file_info if isinstance(x, dict) and x["type"] != "directory"
        ]
        if not file_info:
            return 0, str(datetime.datetime.fromtimestamp(0))
        return sum(x["size"] for x in file_info), max(x["updated"] for x in file_info)

    with ThreadPoolExecutor() as executor:
        summed_size_ant_time = list(
            executor.map(
                get_summed_size_and_latest_timestamp_in_subdirs, partitioned.values()
            )
        )
        for i, (size, timestamp) in zip(partitioned, summed_size_ant_time, strict=True):
            paths[i]["size"] = size
            paths[i]["updated"] = timestamp

    return (
        paths,
        [
            _get_file_list_row(
                x["name"], x.get("updated", None), x["size"], isdir, path, file_system
            )
            for x, isdir in zip(paths, isdir_list, strict=True)
            if isinstance(x, dict)
        ],
        None,
    )


def _get_file_list_row(path, timestamp, size, isdir: bool, current_path, file_system):
    path = _standardize_path(path)
    timestamp = str(timestamp)[:19]
    mb = str(round(size / 1_000_000, 2))
    is_loadable = not isdir or (
        path.endswith(".parquet")
        or all(
            x.endswith(".parquet") or _standardize_path(x) == path
            for x in file_system.ls(path)
        )
    )
    if is_loadable:
        button = html.Button(
            "Load",
            id={"type": "load-parquet", "index": path},
            className="load-button",
            n_clicks=0,
        )
    else:
        button = html.Button(
            "Load",
            id={"type": "load-parquet", "index": path},
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
    path_name = _standardize_path(path).replace(current_path, "").lstrip("/")
    return html.Tr(
        [
            html.Td(button),
            html.Td(
                html.Button(
                    txt_type(f"[DIR] {path_name}" if isdir else path_name),
                    id={"type": "file-path", "index": path},
                    className="path-button",
                    style={
                        "paddingLeft": "3px",
                        "backgroundColor": "rgba(0, 0, 0, 0)",
                        "fillColor": "rgba(0, 0, 0, 0)",
                        "width": "80vh",
                    }
                    | ({"color": "white"} if not isdir else {"color": "#78b3e7"}),
                    n_clicks=0,
                    disabled=False if isdir else True,
                )
            ),
            html.Td(
                timestamp,
                style={
                    "paddingLeft": "10px",
                },
            ),
            html.Td(
                mb,
                style={
                    "paddingLeft": "10px",
                },
            ),
        ]
    )


def _try_glob(path, file_system):
    try:
        return file_system.glob(path, detail=True, recursive=True)
    except Exception:
        return file_system.glob(path, detail=True)
