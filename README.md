# ameriflux-postprocessing
Reformat AmeriFlux data files into NetCDF and add quality flags for outliers and low-turbulence conditions

# Running
Download [AmeriFlux BASE data files](https://ameriflux.lbl.gov/data/flux-data-products/) into one directory, then run
```bash
python ameriflux_base_to_netcdf.py /path/to/ameriflux/base/data \
    /path/to/variable/units/file.csv /path/to/measurement/platform/metadata/file.csv
```
where the file with units for all the variables is available from https://ameriflux.lbl.gov/data/aboutdata/data-variables/ 
and the measurement location metadata file is available from https://ameriflux.lbl.gov/data/measurement-height/

# Dependencies
- Quality flags provided by [hesseflux](https://pypi.org/project/hesseflux/)
- Backup solar radiation provided by [pvlib](https://pypi.org/project/pvlib/)
