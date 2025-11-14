import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import html

_PROFILE_DICT = {}


DEBUG: bool = False


def debug_print(*args):
    print(
        *(
            f"{type(arg).__name__}: {arg}" if isinstance(arg, Exception) else arg
            for arg in args
        )
    )


def time_method_call(method_dict) -> Callable:
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            start_time = time.perf_counter()
            method_name = method.__name__
            print(method_name)
            result = method(self, *args, **kwargs)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            print(method_name, elapsed)
            n_calls, prev_time = method_dict.get(method_name, (0, 0))
            method_dict[method_name] = (n_calls + 1, prev_time + elapsed)
            return result

        return wrapper

    return decorator


def time_function_call(method_dict):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            func_name = func.__name__
            print(func_name)
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            print(func_name, elapsed)
            n_calls, prev_time = method_dict.get(func_name, (0, 0))
            method_dict[func_name] = (n_calls + 1, prev_time + elapsed)
            return result

        return wrapper

    return decorator


if not DEBUG:

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


def get_xarray_bounds(ds) -> tuple[float, float, float, float]:
    return (
        float(ds["x"].min().values),
        float(ds["y"].min().values),
        float(ds["x"].max().values),
        float(ds["y"].max().values),
    )


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
