from pathlib import Path

import dash_bootstrap_components as dbc
from dash import html


def _standardize_path(path: str | Path) -> str:
    """Make sure delimiter is '/' and path ends without '/'."""
    return str(path).replace("\\", "/").replace(r"\"", "/").replace("//", "/")


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
