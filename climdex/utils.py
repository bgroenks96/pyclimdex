import xarray as xr
import numpy as np
from xarray.core.resample import DataArrayResample
from typing import Union, Callable

NS_PER_DAY = 86400000000000

def data_array_or_dataset_var(X: Union[xr.DataArray, xr.Dataset], var=None) -> xr.DataArray:
    """
    If X is a Dataset, selects variable 'var' from X and returns the corresponding
    DataArray. If X is already a DataArray, returns X unchanged.
    """
    if isinstance(X, xr.Dataset):
        assert var is not None, 'var name must be supplied for Dataset input'
        return X[var]
    elif isinstance(X, xr.DataArray):
        return X
    else:
        raise Exception('unrecognized data type: {}'.format(type(X)))
        
def resample_daily(X: xr.DataArray,
                   resample_op: Callable[[DataArrayResample], xr.DataArray],
                   time_dim='time') -> xr.DataArray:
    """
    Resamples subdaily time data according to the given resampling op.
    If X already has daily time resolution, returns X unchanged.
    If X has lower than daily time resolution (e.g. monthly), raises an error.
    """
    assert time_dim in X.coords, f'dimension {time_dim} not found'
    assert X.coords[time_dim].size > 1, 'time dimension should have length > 1'
    interval = (X.coords[time_dim][1] - X.coords[time_dim][0]).astype(np.int64)
    if interval == NS_PER_DAY:
        return X
    if interval > NS_PER_DAY:
        raise ValueError('resample_daily expects input data with daily resolution or higher')
    # resample X to daily
    X_resample = X.resample({time_dim: '1D'})
    return resample_op(X_resample)
    