#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Read in and plot the AmeriFlux data.
"""
from __future__ import division, print_function

import argparse
import calendar
import datetime
import glob
import os.path
import re
import zipfile

import cycler
import matplotlib as mpl
# mpl.interactive(True)
# mpl.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from pytz import UTC
import seaborn as sns
import xarray

MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY
MINUTES_PER_DAY = MINUTES_PER_HOUR * HOURS_PER_DAY
MONTHS_PER_YEAR = 12
OBJECT_DTYPE = np.dtype("O")
MONTH_NAMES = calendar.month_name

HALF_HOUR_FREQ_STR = "30T"
HOUR_FREQ_STR = "H"

PARSER = argparse.ArgumentParser(
    description=__doc__,
)

PARSER.add_argument(
    "ameriflux_root",
    help="Directory containing site directories with data.",
)
PARSER.add_argument(
    "casa_path",
    help="Directory containing downscaled CASA data.",
)

CENTRAL_TIME = pytz.timezone("US/Central")
# README_AmeriFlux_BASE.txt says missing value is -9999
MISSING_VALUES = ["-{nines:s}{dot:s}".format(nines="9"*i, dot=dot)
                  for i in range(4, 5) for dot in (".", "")]
HALF_HOUR = datetime.timedelta(minutes=30)
HOUR = datetime.timedelta(hours=1)

CASA_START = "2003-01-01T00:00+00:00"
NOW_TIME = datetime.datetime.now(UTC).replace(microsecond=0, second=0, minute=0)
NOW = NOW_TIME.isoformat()

AMERIFLUX_METADATA_TO_INCLUDE = [
    "IGBP",
    "LOCATION_LAT",
    "LOCATION_LONG",
    "STATE",
    "URL_AMERIFLUX",
    "UTC_OFFSET",
    # Reference information not always available
    "ACKNOWLEDGEMENT",
    "ACKNOWLEDGEMENT_COMMENT",
    "CLIMATE_KOEPPEN",
    "COUNTRY",
    "DOI",
    "DOI_CITATION",
    "FLUX_MEASUREMENTS_METHOD",
    "SITE_NAME",
    "IGBP_COMMENT",
    "LOCATION_ELEV",
    "TERRAIN",
    "ASPECT",
    "SITE_DESC",
    "SITE_FUNDING",
    "UTC_OFFSET_COMMENT"
]
AMERIFLUX_DATA_VARS = [
    "FC", "FC_1_1_1", "FC_1_2_1", "FC_1_3_1",
    "NEE_PI", "NEE_PI_1_1_1", "NEE_PI_1_2_1", "NEE_PI_1_3_1",
    # "NEE_PI_F"
]
AMERIFLUX_LONG_NAMES = dict(
    FC="Carbon Dioxide (CO2) flux",
    FCH4="Methane (CH4) flux",
    NEE="Net Ecosystem Exchange",
    NEE_PI="Net Ecosystem Exchange from tower team",
    FC_1_1_1="Carbon Dioxide (CO2) flux", # position 1, height 1, replicate 1
    FC_1_2_1="Carbon Dioxide (CO2) flux", # position 1, height 2, replicate 1
    FC_1_3_1="Carbon Dioxide (CO2) flux", # position 1, height 3, replicate 1
    NEE_PI_1_1_1="Net Ecosystem Exchange from tower team", # position 1, height 1, replicate 1
    NEE_PI_1_2_1="Net Ecosystem Exchange from tower team", # position 1, height 1, replicate 1
    NEE_PI_1_3_1="Net Ecosystem Exchange from tower team", # position 1, height 1, replicate 1)
)

# def parse_file(ameriflux_file):
#     """Pull NEE-related data from AmeriFlux file into DataFrame.

#     Parameters
#     ----------
#     ameriflux_file : str

#     Returns
#     -------
#     pd.DataFrame
#     """
#     site_name = os.path.basename(os.path.dirname(ameriflux_file))
#     site_id = os.path.basename(ameriflux_file)[:5]
#     if "-" not in site_id:
#         site_id = "{country:2s}-{site:3s}".format(
#             country=site_id[:2], site=site_id[2:]
#         )
#     year_match = re.search(r"\d{4}_", ameriflux_file)
#     year = ameriflux_file[year_match.start():year_match.end() - 1]
#     year_start = np.datetime64("{year:s}-01-01T00:00-06:00".format(year=year))
#     ds = pd.read_csv(
#         ameriflux_file, index_col=["time"],
#         parse_dates=dict(time=["DoY"]),
#         date_parser=lambda doy: (
#             year_start + np.array(
#                 np.round(float(doy) * MINUTES_PER_DAY),
#                 dtype="m8[m]"
#             )
#         ),
#         na_values=[
#             "-{nines:s}{dot:s}".format(nines="9" * n_nines, dot=dot)
#             for n_nines in (3, 4, 5, 6)
#             for dot in (".", "")
#         ]
#     )
#     nee_ds = ds[[col for col in ds.columns if "NEE" in col]]
#     nee_ds.columns = pd.MultiIndex.from_product([[site_id], nee_ds.columns])
#     return nee_ds


def parse_ameriflux(file_name):
    """Parse AmeriFlux data from file_name.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    flux_data : pd.DataFrame
    """
    dir_name = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    print(base_name, base_name.split("_"))
    network, site_id, fmt, ver = base_name.split("_")
    metadata_file = "AMF_{site:s}_BIF_LATEST.xlsx".format(site=site_id)
    metadata_df = pd.read_excel(
        os.path.join(dir_name, metadata_file)
    ).set_index(
        ["VARIABLE_GROUP", "VARIABLE"]
    )
    # Find timezone offset from UTC
    utc_offset = float(metadata_df.loc[("GRP_UTC_OFFSET", "UTC_OFFSET"), "DATAVALUE"].values[0])
    utc_offset_val = datetime.timedelta(hours=utc_offset)
    site_data_dir = zipfile.ZipFile(file_name)
    site_data_names = [info.filename for info in site_data_dir.filelist
                       if info.filename.endswith(".csv")]
    site_data_names_sel = [name for name in site_data_names
                           if name.split("_")[3] == "HH"]
    if not site_data_names_sel:
        site_data_names_sel = site_data_names
    site_data = pd.read_csv(
        site_data_dir.open(site_data_names_sel[0]),
        skiprows=2,
        # parse_dates=["TIMESTAMP_START", "TIMESTAMP_END"],
        infer_datetime_format=True,
        na_values=MISSING_VALUES
    )
    # Convert times to UTC
    for time_name in ("TIMESTAMP_START", "TIMESTAMP_END"):
        site_data[time_name] = (
            pd.to_datetime(site_data[time_name], format="%Y%m%d%H%M") -
            utc_offset_val
        )
    # site_data.index = pd.IntervalIndex.from_arrays(
    #     site_data["TIMESTAMP_START"],
    #     site_data["TIMESTAMP_END"]
    # )
    first_span = site_data["TIMESTAMP_END"].iloc[0] - site_data["TIMESTAMP_START"].iloc[0]
    assert (site_data["TIMESTAMP_END"] - site_data["TIMESTAMP_START"] == first_span).all()
    if first_span == HALF_HOUR:
        site_data.index = pd.DatetimeIndex(site_data["TIMESTAMP_START"]).to_period(
            HALF_HOUR_FREQ_STR
        )
    elif first_span == HOUR:
        site_data.index = pd.DatetimeIndex(site_data["TIMESTAMP_START"]).to_period(
            HOUR_FREQ_STR
        )
    nee_data = site_data[[col_name for col_name in site_data.columns
                          if "FC" in col_name or "NEE" in col_name]]
    # nee_data.columns = pd.MultiIndex.from_product([[site_id], nee_data.columns])
    nee_data = nee_data.loc[CASA_START:NOW, :]
    has_data = ~nee_data.isnull().all(axis=1)
    # return nee_data.loc[has_data, :]
    nee_ds = xarray.Dataset.from_dataframe(nee_data.loc[has_data, :])
    for name, var in nee_ds.items():
        var.attrs["standard_name"] = "surface_upward_mole_flux_of_carbon_dioxide"
        if name == "FCH4":
            var.attrs["units"] = "nmol/m^2/s"
        else:
            var.attrs["units"] = "umol/m^2/s"
    nee_ds.coords["site"] = np.array(site_id, dtype="U6")
    nee_ds.coords["site"].attrs.update(dict(
        cf_role="timeseries_id",
        standard_name="platform_id",
    ))
    nee_ds.attrs["feature_type"] = "timeSeries"
    for name in AMERIFLUX_METADATA_TO_INCLUDE:
        try:
            coord_val = metadata_df.loc[(slice(None), name), "DATAVALUE"].values[0]
        except KeyError:
            coord_val = ""
        if name.endswith("LAT") or name.endswith("LONG") or name.endswith("OFFSET"):
            try:
                coord_val = float(coord_val)
            except ValueError:
                coord_val = -99
        nee_ds.coords[name] = np.array(coord_val)
        nee_ds.coords[name].attrs.update(dict(
            long_name=name,
            coverage_content_type="referenceInformation",
        ))
    return nee_ds


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    TOWER_DATA = [
        parse_ameriflux(name)
        for name in sorted(glob.glob(os.path.join(
            ARGS.ameriflux_root,
            "AMF_US-*_BASE-BADM_*-*.zip"
        )))
        if os.path.exists(name.rsplit("BASE-BADM")[0] + "BIF_LATEST.xlsx")
    ]
    print("Done reading in")

    HALF_HOUR_DATA = [
        df for df in TOWER_DATA
        if df.indexes["TIMESTAMP_START"].freqstr == HALF_HOUR_FREQ_STR
    ]
    HOUR_DATA = [
        df for df in TOWER_DATA
        if df.indexes["TIMESTAMP_START"].freqstr == HOUR_FREQ_STR
    ]

    HOUR_DATASETS = [
        xarray.concat(
            [ds[amf_var] for ds in HOUR_DATA if amf_var in ds],
            dim="site", coords=AMERIFLUX_METADATA_TO_INCLUDE, compat="override"
        )
        for amf_var in AMERIFLUX_DATA_VARS
        if any(amf_var in ds for ds in HOUR_DATA)
    ]
    HALF_HOUR_DATASETS = [
        xarray.concat(
            [ds[amf_var] for ds in HALF_HOUR_DATA if amf_var in ds],
            dim="site", coords=AMERIFLUX_METADATA_TO_INCLUDE, compat="override"
        )
        for amf_var in AMERIFLUX_DATA_VARS
        if any(amf_var in ds for ds in HALF_HOUR_DATA)
    ]

    for resolution, datasets in zip(("half_hour", "hour"),
                                    (HALF_HOUR_DATASETS, HOUR_DATASETS)):
        sites_included = set()
        to_concat = []
        # ["FC", "NEE_PI", "FC_1_1_1", "NEE_PI_1_1_1", "FC_1_2_1", "NEE_PI_1_2_1"]
        for i in [0, 3, 1, 4, 2, 5]:
            new_sites = set(datasets[i].coords["site"].values) - sites_included
            to_concat.append(datasets[i].sel(site=sorted(new_sites)))
            sites_included.update(new_sites)
        for da in to_concat:
            da.name = "ameriflux_carbon_dioxide_flux_estimate"
        big_da = xarray.concat(to_concat, "site", coords=AMERIFLUX_METADATA_TO_INCLUDE,
                               compat="override", join="outer")
        big_da.attrs.update(dict(
            coverage_content_type="physicalMeasurement",
            long_name="ameriflux_eddy_covariance_carbon_dioxide_flux_estimate",
            cell_methods="site: point TIMESTAMP_START: mean",
            ancillary_variables=" ".join(
                name for name in AMERIFLUX_METADATA_TO_INCLUDE
                if not name.endswith("LAT") and not name.endswith("LONG")
                and not name.endswith("ELEV")
            )
        ))
        big_da.coords["LOCATION_LAT"].attrs.update(dict(
            standard_name="latitude",
            long_name="tower_latitude",
            units="degrees_north",
            axis="Y",
            coverage_content_type="coordinate",
        ))
        big_da.coords["LOCATION_LONG"].attrs.update(dict(
            standard_name="longitude",
            long_name="tower_longitude",
            units="degrees_east",
            axis="X",
            coverage_content_type="coordinate",
        ))
        big_da.coords["UTC_OFFSET"].attrs.update(dict(
            long_name="time zone offset from UTC",
            units="hours",
            source="AmeriFlux",
            coverage_content_type="auxiliaryInformation",
        ))
        big_ds = big_da.to_dataset()
        big_ds.attrs.update(dict(
            feature_type="timeSeries",
            featureType="timeSeries",
            cdm_data_type="Station",
            ncei_template_version="NCEI_NetCDF_TimeSeries_Orthogonal_Template_v2.0",
            title="AmeriFlux CO2 flux data",
            summary="AmeriFlux CO2 flux data as provided by the website.",
            history="{now}: Convert from collection of CSV files to NetCDF".format(now=NOW),
            source="AmeriFlux network (ameriflux.ornl.gov)",
            date_created=NOW,
            institution="AmeriFlux",
            project="AmeriFlux",
            program="AmeriFlux",
            date_modified=NOW,
            date_metadata_modified=NOW,
            comment="""I pulled one flux value for each tower,
preferring "FC" to "NEE_PI" to "FC_1_1_1" to "NEE_PI_1_1_1" to "FC_1_2_1" to "NEE_PI_1_2_1"
where multiple values were present.""",
            Conventions="CF-1.6,ACDD-1.3",
            geospatial_lat_min=min(big_ds.coords["LOCATION_LAT"].values),
            geospatial_lat_max=max(big_ds.coords["LOCATION_LAT"].values),
            geospatial_lon_min=min(big_ds.coords["LOCATION_LONG"].values),
            geospatial_lon_max=max(big_ds.coords["LOCATION_LONG"].values),
            standard_name_vocabulary="CF Standard Name Table v69",
        ))
        period_index = big_ds.indexes["TIMESTAMP_START"]
        big_ds.coords["time_bnds"] = xarray.DataArray(
            np.column_stack([period_index.start_time, period_index.end_time]),
            dims=("TIMESTAMP_START", "bnds2"),
            attrs=dict(standard_name="time", coverage_content_type="coordinate"),
        )
        big_ds.coords["TIMESTAMP_START"] = xarray.DataArray(
            period_index.start_time,
            dims="TIMESTAMP_START",
            attrs=dict(
                standard_name="time", axis="T",
                bounds="time_bnds",
                long_name="start_of_observation_period",
                coverage_content_type="coordinate",
            ),
        )
        if resolution == "half_hour":
            big_ds.attrs["time_coverage_resolution"] = "P0000-00-00T00:30:00"
        else:
            big_ds.attrs["time_coverage_resolution"] = "P0000-00-00T01:00:00"
        big_ds.attrs["time_coverage_start"] = str(min(big_ds.coords["TIMESTAMP_START"].values))
        big_ds.attrs["time_coverage_end"] = str(max(big_ds.coords["TIMESTAMP_START"].values))
        big_ds.coords["time_written"] = NOW_TIME.time().isoformat()
        big_ds.coords["date_written"] = NOW_TIME.date().isoformat()
        encoding = {name: {"_FillValue": -99} for name in big_ds.data_vars}
        encoding.update({name: {"_FillValue": None} for name in big_ds.coords})
        big_ds.to_netcdf(
            "/abl/s0/Continent/dfw5129/ameriflux_netcdf/AmeriFlux_single_value_per_tower_{res:s}_data.nc4".format(res=resolution),
            encoding=encoding, format="NETCDF4_CLASSIC", mode="w"
        )
