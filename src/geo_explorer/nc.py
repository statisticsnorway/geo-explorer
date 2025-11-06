import abc
from abc import abstractmethod
from typing import ClassVar
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


class NetCDFConfig:
    """Sets the configuration for reading NetCDF files and getting crs and bounds.

    Args:
        code_block: String of Python code that takes an xarray.Dataset (ds) and returns an xarray.DataArray (xarr).
            Note that the input Dataset must be references as 'ds' and the output must be assigned to 'xarr'.
    """

    rgb_bands: ClassVar[list[str] | None] = None

    def __init__(
        self,
        code_block: str | None = None,
        time_dtype: str = "datetime64[D]",
    ) -> None:
        self.code_block = code_block
        self.time_dtype = time_dtype

    def get_crs(self, ds: Dataset) -> pyproj.CRS:
        attrs = [x for x in ds.attrs if "projection" in x.lower() or "crs" in x.lower()]
        return pyproj.CRS(ds[attrs[0]])

    def get_bounds(self, ds: Dataset) -> tuple[float, float, float, float]:
        try:
            return as_bounds(ds["bounds"])
        except Exception:
            pass
        attrs = [
            x
            for x in ds.attrs
            if ("bounds" in x.lower() or "bbox" in x.lower())
            and not ("projection" in x.lower() or "crs" in x.lower())
        ]
        if not attrs:
            raise ValueError(f"Could not find bounds attribute in dataset: {ds}")
        elif len(attrs) == 1:
            bounds = next(iter(attrs))
            return as_bounds(bounds)
        try:
            minx = next(
                iter(
                    x
                    for x in attrs
                    if "west" in x.lower()
                    or ("min" in x.lower() and ("lon" in x.lower() or "x" in x.lower()))
                )
            )
            miny = next(
                iter(
                    x
                    for x in attrs
                    if "south" in x.lower()
                    or ("min" in x.lower() and ("lat" in x.lower() or "y" in x.lower()))
                )
            )
            maxx = next(
                iter(
                    x
                    for x in attrs
                    if "east" in x.lower()
                    or ("max" in x.lower() and ("lon" in x.lower() or "x" in x.lower()))
                )
            )
            maxy = next(
                iter(
                    x
                    for x in attrs
                    if "north" in x.lower()
                    or ("max" in x.lower() and ("lat" in x.lower() or "y" in x.lower()))
                )
            )
            return minx, miny, maxx, maxy
        except StopIteration as e:
            raise ValueError(f"Could not find bounds attribute in dataset: {ds}") from e

    @time_method_call(_PROFILE_DICT)
    def to_numpy(
        self,
        ds,
        bounds,
        code_block: str | None,
    ) -> GeoDataFrame | None:

        if code_block is None:
            return

        crs = self.get_crs(ds)
        ds_bounds = self.get_bounds(ds)

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
                xarr = xarr[self.rgb_bands].mean(dim="time")
            return np.array([getattr(xarr, band).values for band in self.rgb_bands])
        if isinstance(xarr, np.ndarray):
            return xarr

        return xarr.values

    @staticmethod
    def dataset_is_empty(ds):
        for attr in ds.coords:
            if not any(getattr(ds, attr).shape):
                print("empty", attr, getattr(ds, attr).shape)
                return True
        return False


def as_bounds(bounds):
    if isinstance(bounds, str):
        return shapely.wkt.loads(bounds).bounds
    elif isinstance(bounds, bytes):
        return shapely.wkb.loads(bounds).bounds
    elif isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        return tuple(bounds)


class NBSNetCDFConfig(NetCDFConfig):
    def get_crs(self, ds: Dataset) -> pyproj.CRS:
        return pyproj.CRS(ds.UTM_projection.epsg_code)

    def get_bounds(self, ds: Dataset) -> tuple[float, float, float, float]:
        return tuple(
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
    rgb_bands: ClassVar[list[str]] = ["B4", "B3", "B2"]
