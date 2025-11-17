import abc
from typing import Any
from typing import ClassVar

import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from shapely.geometry import Polygon

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


class AbstractImageConfig(abc.ABC):
    rgb_bands: ClassVar[list[str] | None] = None

    def __init__(self, code_block: str | None = None) -> None:
        self.code_block = code_block
        self.validate_code_block()

    @abc.abstractmethod
    def get_crs(self, ds: Dataset, path: str) -> pyproj.CRS:
        pass

    @abc.abstractmethod
    def get_bounds(self, ds: Dataset, path: str) -> tuple[float, float, float, float]:
        pass

    @time_method_call(_PROFILE_DICT)
    def filter_ds(
        self,
        ds: Dataset,
        path: str,
        bounds: tuple[float, float, float, float],
    ) -> GeoDataFrame | None:
        crs = self.get_crs(ds, None)
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

        return self._run_code_block(ds)

    @time_method_call(_PROFILE_DICT)
    def to_numpy(self, xarr: Dataset | DataArray) -> GeoDataFrame | None:
        if isinstance(xarr, Dataset) and len(xarr.data_vars) == 1:
            xarr = xarr[next(iter(xarr.data_vars))]
        elif isinstance(xarr, Dataset):
            try:
                xarr = xarr[self.rgb_bands]
            except Exception:
                pass

        if "time" in set(xarr.dims) and (
            not hasattr(xarr["time"].values, "__len__") or len(xarr["time"].values) > 1
        ):
            print("Note: selecting first DataArray in time dimension.")
            xarr = xarr.isel(time=0)

        if isinstance(xarr, Dataset) and self.rgb_bands:
            return np.array([xarr[band].values for band in self.rgb_bands])
        elif isinstance(xarr, Dataset):
            return np.array([xarr[var].values for var in xarr.data_vars])

        if isinstance(xarr, np.ndarray):
            return xarr

        return xarr.values

    def validate_code_block(self) -> None:
        if self.code_block is None:
            return
        try:
            assert "ds" in self.code_block
            eval(self.code_block)
        except NameError:
            # allow valid single line python
            return
        if "xarr=" not in self.code_block.replace(" ", "") or not any(
            txt in self.code_block.replace(" ", "") for txt in ("=ds", "(ds")
        ):
            raise ValueError(
                "'code_block' must be a piece of code that takes the xarray object 'ds' and defines the object 'xarr'. "
                f"Got '{self.code_block}'"
            )

    def _run_code_block(self, ds: DataArray | Dataset) -> Dataset | DataArray:
        if not self.code_block:
            return ds

        try:
            xarr = eval(self.code_block)
            if callable(xarr):
                xarr = xarr(ds)
        except SyntaxError:
            loc = {}
            exec(self.code_block, globals() | {"ds": ds}, loc)
            xarr = loc["xarr"]

        if isinstance(xarr, np.ndarray) and isinstance(ds, DataArray):
            ds.values = xarr
            return ds
        elif isinstance(xarr, np.ndarray) and isinstance(ds, Dataset):
            raise ValueError(
                "Cannot return np.ndarray from 'code_block' if ds is xarray.Dataset."
            )
        return xarr

    def __str__(self) -> str:
        code_block = f"'{self.code_block}'" if self.code_block else None
        return f"{self.__class__.__name__}({code_block})"

    def __repr__(self) -> str:
        return str(self)


class NetCDFConfig(AbstractImageConfig):
    """Sets the configuration for reading NetCDF files and getting crs and bounds.

    Args:
        code_block: String of Python code that takes an xarray.Dataset (ds) and returns an xarray.DataArray (xarr).
            Note that the input Dataset must be references as 'ds' and the output must be assigned to 'xarr'.
    """

    rgb_bands: ClassVar[list[str]] = ["B4", "B3", "B2"]

    def get_bounds(self, ds, path) -> tuple[float, float, float, float]:
        return get_xarray_bounds(ds)

    def get_crs(self, ds: Dataset, path: str) -> pyproj.CRS:
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


class NBSNetCDFConfig(NetCDFConfig):
    def get_crs(self, ds: Dataset, path: str) -> pyproj.CRS:
        return pyproj.CRS(ds.UTM_projection.epsg_code)


class Sentinel2NBSNetCDFConfig(NBSNetCDFConfig):
    rgb_bands: ClassVar[list[str]] = ["B4", "B3", "B2"]


class GeoTIFFConfig(AbstractImageConfig):
    def get_crs(self, ds, path):
        with rasterio.open(path) as src:
            return src.crs

    def get_bounds(self, ds, path) -> tuple[float, float, float, float]:
        with rasterio.open(path) as src:
            return tuple(
                GeoSeries([shapely.box(*src.bounds)], crs=src.crs)
                .to_crs(4326)
                .total_bounds
            )

    def filter_ds(
        self,
        ds: None,
        path: str,
        bounds: tuple[float, float, float, float],
    ) -> DataArray | Dataset:
        from rioxarray.rioxarray import _generate_spatial_coords

        bbox = shapely.box(*bounds)
        arr, crs, descriptions = rasterio_to_numpy(
            path, bbox, return_attrs=["crs", "descriptions"]
        )
        bounds_in_img_crs = GeoSeries([bbox], crs=4326).to_crs(crs).total_bounds

        if not all(arr.shape):
            return DataArray(
                arr,
                dims=["y", "x"],
                attrs={"crs": crs},
            )

        if len(arr.shape) == 2:
            height, width = arr.shape
        elif len(arr.shape) == 3 and arr.shape[0] == 1:
            arr = arr[0]
            height, width = arr.shape
        elif len(arr.shape) == 3:
            height, width = arr.shape[1:]
        else:
            raise ValueError(arr.shape)

        transform = rasterio.transform.from_bounds(*bounds_in_img_crs, width, height)
        coords = _generate_spatial_coords(transform, width, height)

        if len(arr.shape) == 2:
            ds = DataArray(
                arr,
                coords=coords,
                dims=["y", "x"],
                attrs={"crs": crs},
            )
        else:
            if len(descriptions) != arr.shape[0]:
                descriptions = range(arr.shape[0])
            ds = Dataset(
                {
                    desc: DataArray(
                        arr[i],
                        coords=coords,
                        dims=["y", "x"],
                        attrs={"crs": crs},
                        name=desc,
                    )
                    for i, desc in enumerate(descriptions)
                }
            )
        return self._run_code_block(ds)


def rasterio_to_numpy(
    img_path: str, bbox: Polygon, return_attrs: list[str] | None = None
) -> np.ndarray | tuple[Any]:
    with rasterio.open(img_path) as src:
        bounds_in_img_crs = GeoSeries([bbox], crs=4326).to_crs(src.crs).total_bounds
        window = rasterio.windows.from_bounds(
            *bounds_in_img_crs, transform=src.transform
        )
        arr = src.read(window=window, boundless=False, masked=False)
        if not return_attrs:
            return arr
        return (arr, *[getattr(src, attr) for attr in return_attrs])


def _pd():
    """Function that makes sure 'pd' is not removed by 'ruff' fixes. Because pd is useful in code_block."""
    pd
