import abc
from abc import abstractmethod
from dataclasses import dataclass
from multiprocessing import cpu_count

import joblib
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
from affine import Affine
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from rasterio import features
from shapely.geometry import shape

try:
    from xarray import DataArray
    from xarray import Dataset
except ImportError:

    class Dataset:
        """Placeholder."""

    class DataArray:
        """Placeholder."""


from .utils import _PROFILE_DICT
from .utils import time_function_call
from .utils import time_method_call


@dataclass
class NetCDFConfig(abc.ABC):
    """Sets the configuration for reading NetCDF files and getting crs and bounds.

    Args:
        code_block: String of Python code that takes an xarray.Dataset (ds) and returns an xarray.DataArray (xarr).
            Note that the input Dataset must be references as 'ds' and the output must be assigned to 'xarr'.
    """

    code_block: str | None = None
    time_dtype: str = "datetime64[D]"

    @abstractmethod
    def crs_getter(self, ds: Dataset):
        attrs = [x for x in ds.attrs if "projection" in x.lower() or "crs" in x.lower()]
        return pyproj.CRS(ds[attrs[0]])

    @abstractmethod
    def bounds_getter(self, ds: Dataset):
        attrs = [
            x
            for x in ds.attrs
            if ("bounds" in x.lower() or "bbox" in x.lower())
            and not ("projection" in x.lower() or "crs" in x.lower())
        ]
        return pyproj.CRS(ds[attrs[0]])

    @time_method_call(_PROFILE_DICT)
    def to_numpy(
        self,
        ds,
        bounds,
        code_block: str | None,
    ) -> GeoDataFrame | None:

        if code_block is None:
            return

        crs = self.crs_getter(ds)
        ds_bounds = self.bounds_getter(ds)

        bbox_correct_crs = (
            GeoSeries([shapely.box(*bounds)], crs=4326).to_crs(crs).union_all()
        )
        clipped_bbox = bbox_correct_crs.intersection(shapely.box(*ds_bounds))
        minx, miny, maxx, maxy = clipped_bbox.bounds

        ds = ds.sel(
            x=slice(minx, maxx),
            y=slice(maxy, miny),
        )

        if self.dataset_is_empty(ds):
            return

        ds["time"] = ds["time"].astype(self.time_dtype)

        try:
            xarr = eval(code_block)
            if callable(xarr):
                xarr = xarr(ds)
        except SyntaxError:
            loc = {}
            exec(code_block, globals=globals() | {"ds": ds}, locals=loc)
            xarr = loc["xarr"]

        if isinstance(xarr, Dataset):
            if "time" in set(xarr.dims) and (
                not hasattr(xarr.time.values, "__len__") or len(xarr.time.values) > 1
            ):
                xarr = xarr.mean(dim="time")
            return np.array([xarr.B4.values, xarr.B3.values, xarr.B2.values])

        return xarr.values

    @time_method_call(_PROFILE_DICT)
    def to_geopandas(
        self,
        ds,
        bounds,
        code_block: str | None,
    ) -> GeoDataFrame | None:

        if code_block is None:
            return

        crs = self.crs_getter(ds)
        ds_bounds = self.bounds_getter(ds)

        bbox_correct_crs = (
            GeoSeries([shapely.box(*bounds)], crs=4326).to_crs(crs).union_all()
        )
        clipped_bbox = bbox_correct_crs.intersection(shapely.box(*ds_bounds))
        minx, miny, maxx, maxy = clipped_bbox.bounds

        ds = ds.sel(
            x=slice(minx, maxx),
            y=slice(maxy, miny),
        )

        if self.dataset_is_empty(ds):
            return

        ds["time"] = ds["time"].astype(self.time_dtype)

        try:
            xarr = eval(code_block)
            if callable(xarr):
                xarr = xarr(ds)
        except SyntaxError:
            loc = {}
            exec(code_block, globals=globals() | {"ds": ds}, locals=loc)
            xarr = loc["xarr"]

        if not isinstance(xarr, (DataArray | Dataset)):
            raise ValueError(
                f"code block must return xarray DataArray or Dataset. Got {type(xarr)} from {code_block}"
            )

        df: GeoDataFrame = xarry_to_geopandas(
            xarr, "value", bounds=clipped_bbox.bounds, crs=crs
        )
        print(df)
        return df

    @staticmethod
    def dataset_is_empty(ds):
        for attr in ds.coords:
            if not any(getattr(ds, attr).shape):
                print("empty", attr, getattr(ds, attr).shape)
                return True
        return False


class NBSNetCDFConfig(NetCDFConfig):
    def crs_getter(self, ds):
        return pyproj.CRS(ds.UTM_projection.epsg_code)

    def bounds_getter(self, ds):
        return (
            GeoSeries(
                [
                    shapely.box(
                        *[
                            getattr(ds, f"geospatial_{x}")
                            for x in ["lon_min", "lat_min", "lon_max", "lat_max"]
                        ]
                    )
                ],
                crs=ds.geospatial_bounds_crs,
            )
            .to_crs(pyproj.CRS(ds.UTM_projection.epsg_code))
            .total_bounds
        )


class Sentinel2NBSNetCDFConfig(NBSNetCDFConfig):
    rgb_bands = ["B4", "B3", "B2"]


@time_function_call(_PROFILE_DICT)
def xarry_to_geopandas(xarr, name: str, bounds, crs):
    if isinstance(xarr, Dataset):

        def scale_to_255(arr):
            max_val = np.max(arr)
            arr = np.clip(arr, 0, max_val)
            return (arr / max_val * 255).astype(np.uint8)

        def rgb_to_hex(r, g, b):
            r = scale_to_255(r)
            g = scale_to_255(g)
            b = scale_to_255(b)
            return np.vectorize(lambda r, g, b: f"#{r:02x}{g:02x}{b:02x}")(r, g, b)

        rgb = rgb_to_hex(xarr.B4.values, xarr.B3.values, xarr.B2.values)
        height, width = rgb.shape
        transform = rasterio.transform.from_bounds(*bounds, width, height)
        int_to_hex_mapper = dict(enumerate(np.unique(rgb)))

        rgb_as_int = np.zeros_like(rgb)
        for key, value in int_to_hex_mapper.items():
            rgb_as_int[rgb == value] = key

        df = GeoDataFrame(
            pd.DataFrame(
                _array_to_geojson(rgb_as_int, transform, processes=cpu_count()),
                columns=[name, "geometry"],
            ),
            geometry="geometry",
            crs=crs,
        )[lambda df: (df[name].notna())]
        df[name] = df[name].map(int_to_hex_mapper)
        df["_color"] = df[name]
        return df

    if len(xarr.shape) == 2:
        height, width = xarr.shape
    elif len(xarr.shape) == 3:
        if xarr.shape[0] != 1:
            raise ValueError("0th dimension/axis must have 1 level")
        height, width = xarr.shape[1:]
    else:
        raise ValueError(xarr.shape)
    transform = rasterio.transform.from_bounds(*bounds, width, height)
    return GeoDataFrame(
        pd.DataFrame(
            _array_to_geojson(xarr.values, transform, processes=cpu_count()),
            columns=[name, "geometry"],
        ),
        geometry="geometry",
        crs=crs,
    )[lambda df: (df[name].notna())]


@time_function_call(_PROFILE_DICT)
def _array_to_geojson(
    array: np.ndarray, transform: Affine, processes: int
) -> list[tuple]:
    if hasattr(array, "mask"):
        if isinstance(array.mask, np.ndarray):
            mask = array.mask == False
        else:
            mask = None
        array = array.data
    else:
        mask = None

    try:
        return _array_to_geojson_loop(array, transform, mask, processes)
    except ValueError:
        try:
            array = array.astype(np.float32)
            return _array_to_geojson_loop(array, transform, mask, processes)

        except Exception as err:
            raise err.__class__(f"{array.shape}: {err}") from err


@time_function_call(_PROFILE_DICT)
def _array_to_geojson_loop(array, transform, mask, processes):
    if processes == 1:
        return [
            (value, shape(geom))
            for geom, value in features.shapes(array, transform=transform, mask=mask)
        ]
    else:
        with joblib.Parallel(n_jobs=processes, backend="threading") as parallel:
            return parallel(
                joblib.delayed(_value_geom_pair)(value, geom)
                for geom, value in features.shapes(
                    array, transform=transform, mask=mask
                )
            )


def _value_geom_pair(value, geom):
    return (value, shape(geom))
