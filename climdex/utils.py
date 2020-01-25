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

def max_consecutive_count(x):
    assert hasattr(x, 'dtype') and x.dtype == np.bool, 'x should be a boolean ndarray or DataArray'
    def _ffill(arr, axis):
        """
        Generic implementation of ffill borrowed from xarray.core.missing.ffill.
        Requires bottleneck to work.
        """
        import bottleneck as bn
        from xarray.core.computation import apply_ufunc
        # work around for bottleneck 178
        _limit = arr.shape[axis]
        return apply_ufunc(
            bn.push,
            arr,
            dask="allowed",
            keep_attrs=True,
            output_dtypes=[arr.dtype],
            kwargs=dict(n=_limit, axis=axis),
        )
    # pad x with extra, opposite value to trigger switching logic for final value
    x_ = np.concatenate([x, ~x[-1:]])
    # [x[0], [1 : X[i] != X[i-1]]]; i.e. initial element + all elements where pattern changes
    chunk_cond = np.concatenate([x_[:1], x_[1:] != x_[:-1]])
    # create block of time indices, uniform across extra/spatial dimensions
    all_indices = np.cumsum(np.ones(chunk_cond.shape), axis=0)
    # select indices at change points
    chunk_indices = np.where(chunk_cond, all_indices, np.empty(chunk_cond.shape)*np.nan)
    # forward fill NaNs with previous index and replace any remaining nan values
    filled_indices = np.nan_to_num(_ffill(chunk_indices, axis=0))
    # take sequential difference between indices
    # this gives us the distance between pattern changes, or chunk sizes;
    # we multiply by the original boolean array x to filter out chunks of False values
    true_counts = np.diff(filled_indices, axis=0)*x
    # finally, take the max over all of the True chunk sizes
    return true_counts.max(axis=0)
    
    