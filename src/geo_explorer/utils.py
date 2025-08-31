from pathlib import Path
import time
from functools import wraps
from typing import Callable

import dash_bootstrap_components as dbc
from dash import html


def _standardize_path(path: str | Path) -> str:
    """Make sure delimiter is '/' and path ends without '/'."""
    return str(path).replace("\\", "/").replace(r"\"", "/")


def _clicked_button_style():
    return {
        "color": "#e4e4e4",
        "background": "#2F3034",
    }


def _unclicked_button_style():
    return {
        "background": "#e4e4e4",
        "color": "black",
    }


def get_button_with_tooltip(
    button_text, id, tooltip_text: str, n_clicks=0, **button_kwargs
) -> list[html.Button, dbc.Tooltip]:
    return [
        html.Button(
            button_text,
            id=id,
            n_clicks=n_clicks,
            **button_kwargs,
        ),
        dbc.Tooltip(
            tooltip_text,
            target=id,
            delay={"show": 500, "hide": 100},
        ),
    ]


def time_method_call(method_dict) -> Callable:
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = method(self, *args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            method_name = method.__name__
            n_calls, prev_time = method_dict.get(method_name, (0, 0))
            method_dict[method_name] = (n_calls + 1, prev_time + elapsed)
            return result

        return wrapper

    return decorator
