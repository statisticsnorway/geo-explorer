from typing import ClassVar

import numpy as np
import pyproj
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
import pandas as pd

try:
    from xarray import DataArray
    from xarray import Dataset
except ImportError:

    class DataArray:
        """Placeholder."""

    class Dataset:
        """Placeholder."""


from .utils import _PROFILE_DICT
from .utils import get_xarray_bounds
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
        self._code_block = code_block
        self.code_block  # trigger property
        self.time_dtype = time_dtype

    @property
    def code_block(self) -> str | None:
        return self._code_block

    @code_block.setter
    def code_block(self, value: str | None):
        if (
            value
            and "xarr=" not in value.replace(" ", "")
            or "=ds" not in value.replace(" ", "")
        ):
            raise ValueError(
                "'code_block' must be a piece of code that takes the xarray object 'ds' and defines the object 'xarr'"
            )
        self._code_block = value

    def get_crs(self, ds: Dataset) -> pyproj.CRS:
        attrs = [x for x in ds.attrs if "projection" in x.lower() or "crs" in x.lower()]
        if not attrs:
            raise ValueError(f"Could not find CRS attribute in dataset: {ds}")
        for i, attr in enumerate(attrs):
            try:
                return pyproj.CRS(ds.attrs[attr])
            except Exception as e:
                if i == len(attrs) - 1:
                    attrs_dict = {attr: ds.attrs[attr] for attr in attrs}
                    raise ValueError(
                        f"No valid CRS attribute found among {attrs_dict}"
                    ) from e

    @time_method_call(_PROFILE_DICT)
    def to_numpy(
        self,
        ds: Dataset,
        bounds: tuple[float, float, float, float],
        code_block: str | None,
    ) -> GeoDataFrame | None:
        crs = self.get_crs(ds)
        ds_bounds = get_xarray_bounds(ds)

        bbox_correct_crs = (
            GeoSeries([shapely.box(*bounds)], crs=4326).to_crs(crs).union_all()
        )
        clipped_bbox = bbox_correct_crs.intersection(shapely.box(*ds_bounds))
        minx, miny, maxx, maxy = clipped_bbox.bounds

        ds = ds.sel(
            x=slice(minx, maxx),
            y=slice(maxy, miny),
        )

        if "time" in set(ds.dims).union(set(ds.coords)):
            ds["time"] = ds["time"].astype(self.time_dtype)

        if not code_block and isinstance(ds, DataArray):
            return ds.values
        elif not code_block and isinstance(ds, Dataset):
            # if max(len(ds[x]) for x in set(ds.coords).difference({"x", "y"})) in [0, 1]:
            #     return ds
            raise ValueError(
                "code_block cannot be None for nc files with more than one dimension."
            )

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


class NBSNetCDFConfig(NetCDFConfig):
    def get_crs(self, ds: Dataset) -> pyproj.CRS:
        return pyproj.CRS(ds.UTM_projection.epsg_code)


class Sentinel2NBSNetCDFConfig(NBSNetCDFConfig):
    rgb_bands: ClassVar[list[str]] = ["B4", "B3", "B2"]


def _pd():
    """Function that makes sure 'pd' is not removed by 'ruff' fixes. Because pd is useful in code_block."""
    pd
