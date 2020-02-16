import xarray as xr
import numpy as np
import climdex.utils as utils
from typing import Union

def indices(time_dim='time', convert_units_fn=lambda x: x):
    return TemperatureIndices(time_dim=time_dim, convert_units_fn=convert_units_fn)

class TemperatureIndices:
    def __init__(self, time_dim='time', convert_units_fn=lambda x: x):
        self.time_dim = time_dim
        self.convert_units_fn = convert_units_fn

    def annual_frost_days(self, X: Union[xr.DataArray, xr.Dataset], varname='MINT'):
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.min(), time_dim=self.time_dim)
        return (X_arr < self.convert_units_fn(0.0)).astype(X_arr.dtype).groupby(f'{self.time_dim}.year').sum()

    def annual_icing_days(self, X: Union[xr.DataArray, xr.Dataset], varname='MAXT'):
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.max(), time_dim=self.time_dim)
        return (X_arr < self.convert_units_fn(0.0)).astype(X_arr.dtype).groupby(f'{self.time_dim}.year').sum()

    def annual_summer_days(self, X: Union[xr.DataArray, xr.Dataset], varname='MAXT'):
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.max(), time_dim=self.time_dim)
        return (X_arr > self.convert_units_fn(25.0)).astype(X_arr.dtype).groupby(f'{self.time_dim}.year').sum()

    def annual_tropical_nights(self, X: Union[xr.DataArray, xr.Dataset], varname='MINT'):
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.min(), time_dim=self.time_dim)
        return (X_arr > self.convert_units_fn(20.0)).astype(X_arr.dtype).groupby(f'{self.time_dim}.year').sum()
    
    def annual_growing_season_length(self, X: Union[xr.DataArray, xr.Dataset], varname='MEANT'):
        raise NotImplementedError()
        
    def monthly_txx(self, X: Union[xr.DataArray, xr.Dataset], varname=None):
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.max(), time_dim=self.time_dim)
        return X.resample({self.time_dim: '1M'}).max()
    
    def monthly_txn(self, X: Union[xr.DataArray, xr.Dataset], varname=None):
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.max(), time_dim=self.time_dim)
        return X.resample({self.time_dim: '1M'}).min()
    
    def monthly_tnx(self, X: Union[xr.DataArray, xr.Dataset], varname=None):
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.min(), time_dim=self.time_dim)
        return X.resample({self.time_dim: '1M'}).max()
    
    def monthly_tnn(self, X: Union[xr.DataArray, xr.Dataset], varname=None):
        X_arr = utils.data_array_or_dataset_var(X, var=varname)
        X_arr = utils.resample_daily(X_arr, lambda x: x.min(), time_dim=self.time_dim)
        return X.resample({self.time_dim: '1M'}).min()
    
    def daily_temperature_range(self,
                                X1: Union[xr.DataArray, xr.Dataset],
                                X2: Union[xr.DataArray, xr.Dataset]=None,
                                min_varname='MINT',
                                max_varname='MAXT'):
        X1_arr = utils.data_array_or_dataset_var(X1, var=min_varname)
        X2_arr = utils.data_array_or_dataset_var(X2, var=max_varname)
        X_min_arr = utils.resample_daily(X1_arr, lambda x: x.min(), time_dim=self.time_dim)
        X_max_arr = utils.resample_daily(X2_arr, lambda x: x.max(), time_dim=self.time_dim)
        return X_max_arr - X_min_arr
    