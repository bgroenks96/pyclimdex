# pyclimdex
Implementation of Climdex indices in Python/xarray/dask

### Usage

1. Install from pypi:

```
pip install pyclimdex
```

or from GitHub:

```
pip install git+https://github.com/bgroenks96/pyclimdex.git
```

2. Import either temperature or precipitation indices

```python
import climdex.precipitation as pdex
```

3. Initialize indices and compute them on your xarray DataArray or Dataset

```python
   indices = pdex.indices(time_dim='time')
   # compute total monthly precipitation;
   # your data should be daily or sub-daily time scale
   ptot = indices.prcptot(data, period='1M')
```

That's it! You can find more info on the Climdex indices [here](https://climdex.org).

`pyclimdex` currently supports the following indices for temperature and precipitation respectively:

**Temperature**

- Annual frost days
- Annual tropical nights
- Annual icing days
- Annual summer days
- Monthly max daily max temp (TXx)
- Monthly min daily max temp (TXn)
- Monthly max daily min temp (TNx)
- Monthly min daily min temp (TNx)
- Daily temperature range (DTR)

**Precipitation**

- Monthly 1-day precip (Rx1day)
- Monthly 5-day precip (Rx5day)
- Annual 10mm precip days
- Annual 20mm precip days
- Annual n mm precip days
- Total precipitation (variable time period)
- Simple intensity index (SDII)
- Consecutive dry days (CDD)
- Consecutive wet days (CWD)

Indices which rely on historical data are not currently supported. Contributions are welcome!
