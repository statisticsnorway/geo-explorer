import abc
from abc import abstractmethod
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
import xarray as xr
from affine import Affine
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from rasterio import features
from shapely.geometry import shape


@dataclass
class NetCDFConfig(abc.ABC):
    """Args:
    code_block: String of Python code that takes an xarray.Dataset (ds) and returns an xarray.DataArray (xarr).
        Note that the input Dataset must be references as 'ds' and the output must be assigned to 'xarr'.
    """

    code_block: str | None = None

    @abstractmethod
    def crs_getter(self, ds: xr.Dataset):
        pass

    @abstractmethod
    def bounds_getter(self, ds: xr.Dataset):
        pass

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

        filtered = ds.sel(
            x=slice(minx, maxx),
            y=slice(maxy, miny),
        )

        if self.dataset_is_empty(filtered):
            return

        ds = filtered

        loc = {}

        try:
            xarr = eval(code_block)
            if callable(xarr):
                xarr = xarr(ds)
        except SyntaxError:
            exec(code_block, globals=globals() | {"ds": ds}, locals=loc)
            xarr = loc["xarr"]

        if not isinstance(xarr, (xr.DataArray | xr.Dataset)):
            raise ValueError(
                f"code block must return xarray DataArray or Dataset. Got {type(xarr)} from {code_block}"
            )

        df: GeoDataFrame = xarry_to_geopandas(
            xarr, "value", bounds=clipped_bbox.bounds, crs=crs
        )
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


def xarry_to_geopandas(xarr, name: str, bounds, crs):
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
            _array_to_geojson(xarr.values, transform, 1),
            columns=[name, "geometry"],
        ),
        geometry="geometry",
        crs=crs,
    )[lambda df: (df[name].notna())]


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
