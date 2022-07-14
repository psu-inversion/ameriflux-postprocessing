#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Read in and plot the AmeriFlux data.
"""
from __future__ import division, print_function

import argparse
import calendar
import datetime
import glob
import itertools
import logging
import operator
import os.path
import re
import warnings
import zipfile
from typing import List

import hesseflux
import numpy as np
import pandas as pd
import pint_xarray  # noqa: F401  # pylint: disable=unused-import
import pvlib.location
import pytz
import xarray
from pytz import UTC

_LOGGER = logging.getLogger(__name__)

MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY
MINUTES_PER_DAY = MINUTES_PER_HOUR * HOURS_PER_DAY
MONTHS_PER_YEAR = 12
OBJECT_DTYPE = np.dtype("O")
MONTH_NAMES = calendar.month_name

HAVE_ZIP_NOT_NETCDF = False
"""Whether to read the data from ZIP or skip to netCDF

True: have ZIP files but not netCDF
False: have netCDF
"""

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
    metavar="flux_unit_file",
    type=pd.read_csv,
    dest="flux_unit_df",
)
PARSER.add_argument(
    "measurement_height_file",
    type=str,
)

CENTRAL_TIME = pytz.timezone("US/Central")
# README_AmeriFlux_BASE.txt says missing value is -9999
MISSING_VALUES = [
    "-{nines:s}{dot:s}".format(nines="9" * i, dot=dot)
    for i in range(4, 5)
    for dot in (".", "")
]
HALF_HOUR = datetime.timedelta(minutes=30)
HOUR = datetime.timedelta(hours=1)

N_BOOTSTRAP_SAMPLES = 5
"""Number of bootstrap samples to use for ustar filtering.

1 is fastest, five seems reasonable, 20 is slow.
"""

CASA_START = "2003-01-01T00:00+00:00"
CASA_END = "2019-12-31T23:59:59.999999999+00:00"
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
    "UTC_OFFSET_COMMENT",
]
MEASUREMENT_METADATA_TO_INCLUDE = [
    # Data Quality Flags
    "FC_SSITC_TEST",
    "NEE_SSITC_TEST",
    "NEE_PI_SSITC_TEST",
]

AMF_BASE_VAR_NAME_REGEX = re.compile(
    r"^(?P<physical_name>\w+?)(?P<quality_flag1>_SSITC_TEST)?(?:_PI)?(?:_QC)?"
    r"(?:_F)?(?:_IU)?(?P<loc_rep_agg>_\d+_\d+_(?:\d+|A)|_\d+)?(?:_(?:SD|N))?$"
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


def variable_sort_key(variable_name):
    # type: (str) -> int
    """Return a sort key for the variable name.

    Puts the NEE/storage-flux-adjusted variables before the
    FC/non-storage-flux-adjusted variables, then puts the site-level
    variables (without the location suffix _1_1_1 or similar) before
    the individual measurements, then sorts the individual
    measurements using the location suffix as an integer (so _1_1_1
    becomes 111).

    Parameters
    ----------
    variable_name : str

    Returns
    -------
    sort_key : int

    """
    result = variable_name.startswith("NEE") * 10000
    if variable_name.count("_") > 3:
        result += int("".join(variable_name.rsplit("_", 4)[-3:]))
    return result


def parse_ameriflux(file_name, unit_df, tower_height_da):
    # type: (str, pd.DataFrame, xarray.DataArray) -> xarray.Dataset
    """Parse AmeriFlux data from file_name.

    Parameters
    ----------
    file_name : str
    unit_df : pd.DataFrame
    tower_height_da : xarray.DataArray

    Returns
    -------
    flux_data : xarray.Dataset
    """
    # dir_name = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    _LOGGER.info(
        "Parsing file %s, decomposed as %s", base_name, base_name[:-4].split("_")
    )
    _network, site_id, _fmt, ver = base_name[:-4].split("_")
    with zipfile.ZipFile(file_name) as site_data_dir:
        site_data_names = [
            info.filename
            for info in site_data_dir.filelist
            if info.filename.endswith(".csv")
        ]
        site_data_names_sel = [
            name for name in site_data_names if name.split("_")[3] == "HH"
        ]
        if not site_data_names_sel:
            site_data_names_sel = site_data_names

        # Read site metadata
        metadata_names = [
            info.filename
            for info in site_data_dir.filelist
            if info.filename.endswith(".xlsx")
        ]
        # metadata_file = "AMF_{site:s}_BIF_LATEST.xlsx".format(site=site_id)
        metadata_file = metadata_names[0]
        with site_data_dir.open(metadata_file) as meta_file:
            metadata_df = (
                pd.read_excel(
                    meta_file
                    # os.path.join(dir_name, metadata_file)
                )
                .set_index(["VARIABLE_GROUP", "VARIABLE"])
                .sort_index()
            )

        # Find timezone offset from UTC
        utc_offset = float(
            metadata_df.loc[("GRP_UTC_OFFSET", "UTC_OFFSET"), "DATAVALUE"].values[0]
        )
        utc_offset_val = datetime.timedelta(hours=utc_offset)

        # Read site data
        with site_data_dir.open(site_data_names_sel[0]) as data_file:
            site_data = pd.read_csv(
                data_file,
                skiprows=2,
                # parse_dates=["TIMESTAMP_START", "TIMESTAMP_END"],
                infer_datetime_format=True,
                na_values=MISSING_VALUES,
            ).sort_index(1)
        # Convert times to UTC
        for time_name in ("TIMESTAMP_START", "TIMESTAMP_END"):
            site_data[time_name] = (
                pd.to_datetime(site_data[time_name], format="%Y%m%d%H%M")
                - utc_offset_val
            )
        # site_data.index = pd.IntervalIndex.from_arrays(
        #     site_data["TIMESTAMP_START"],
        #     site_data["TIMESTAMP_END"]
        # )
        first_span = (
            site_data["TIMESTAMP_END"].iloc[0] - site_data["TIMESTAMP_START"].iloc[0]
        )
        assert (
            site_data["TIMESTAMP_END"] - site_data["TIMESTAMP_START"] == first_span
        ).all()
        if first_span == HALF_HOUR:
            site_data.index = pd.DatetimeIndex(site_data["TIMESTAMP_START"]).to_period(
                HALF_HOUR_FREQ_STR
            )
        elif first_span == HOUR:
            site_data.index = pd.DatetimeIndex(site_data["TIMESTAMP_START"]).to_period(
                HOUR_FREQ_STR
            )
        nee_data = site_data[
            [
                col_name
                for col_name in site_data.columns
                # Methane flux is interesting, but not what I'm after here
                if ("FC" in col_name and "FCH4" not in col_name)
                or "NEE" in col_name
                or "SW_IN" in col_name
                or "USTAR" in col_name
                or "TA" in col_name
            ]
        ]
        # nee_data.columns = pd.MultiIndex.from_product([[site_id], nee_data.columns])
        freq = "30min" if first_span == HALF_HOUR else "1H"
        nee_data = nee_data.resample(freq).mean().loc[CASA_START:CASA_END, :]
        has_data = ~nee_data.isnull().all(axis=1)
        # return nee_data.loc[has_data, :]
        nee_data = nee_data.loc[has_data, :]
        start_date = nee_data.index[0].start_time.replace(
            month=1, day=1, hour=0, minute=0, second=0
        )
        end_date = nee_data.index[-1].end_time.replace(
            month=12, day=31, hour=23, minute=59, second=59
        )
        nee_data = nee_data.reindex(
            pd.period_range(start_date, end_date, freq=freq, name="TIMESTAMP_START")
        )
        nee_ds = xarray.Dataset.from_dataframe(nee_data).expand_dims("site", 0)

    ##############################
    # Set site metadata
    nee_ds.coords["site"] = np.array([site_id], dtype="U6")
    nee_ds.coords["site"].attrs.update(
        dict(
            cf_role="timeseries_id",
            standard_name="platform_id",
        )
    )
    nee_ds.attrs["feature_type"] = "timeSeries"
    for metadata_name in AMERIFLUX_METADATA_TO_INCLUDE:
        try:
            coord_val = metadata_df.loc[
                (slice(None), metadata_name), "DATAVALUE"
            ].values[0]
        except KeyError:
            coord_val = ""
        if (
            metadata_name.endswith("LAT")
            or metadata_name.endswith("LONG")
            or metadata_name.endswith("OFFSET")
            or metadata_name.endswith("ELEV")
        ):
            try:
                coord_val = float(coord_val)
            except ValueError:
                coord_val = -99
            else:
                assert np.isfinite(coord_val)
        nee_ds.coords[metadata_name] = (
            ("site",),
            np.array([coord_val]),
            dict(
                long_name=metadata_name,
                coverage_content_type="referenceInformation",
            ),
        )

    nee_ds.coords["data_version"] = (("site",), [ver], {"long_name": "data version"})

    # Add a fallback radiation variable
    pv_location = pvlib.location.Location(
        nee_ds.coords["LOCATION_LAT"].values[0],
        nee_ds.coords["LOCATION_LONG"].values[0],
        name=nee_ds.coords["site"].values[0],
        altitude=(
            nee_ds.coords["LOCATION_ELEV"].values[0]
            if "LOCATION_ELEV" in nee_ds.coords
            else 0.0
        ),
    )
    _LOGGER.debug(pv_location)
    nee_ds["SW_IN_CLEARSKY"] = (
        ("site", "TIMESTAMP_START"),
        pv_location.get_clearsky(nee_ds.indexes["TIMESTAMP_START"].to_timestamp())[
            "ghi"
        ].values[np.newaxis, :],
        # From https://pvlib-python.readthedocs.io/en/stable/reference/generated/
        #   pvlib.irradiance.dni.html
        # GHI = Global Horizontal Irradiance
        # DHI = Diffuse Horizontal Irradiance
        # DNI = Direct Normal Irradiance
        # Looks like DNI > GHI > DHI; DNI is derived from GHI and DHI
        # From https://pvlib-python.readthedocs.io/en/stable/gallery/
        #   irradiance-transposition/plot_ghi_transposition.html
        # GHI is close to photovoltaic input for a flat solar array
        {
            "standard_name": (
                "surface_downwelling_shortwave_flux_in_air_assuming_clear_sky"
            ),
            "units": "W/m^2",
            "long_name": (
                "downwelling shortwave irradiance at the surface assuming clear sky"
                " from pvlib-python"
            ),
            "description": (
                "downwelling shortwave irradiance near the surface assuming clear sky."
                "\nCalculated using PVLib"
            ),
        },
    )
    del pv_location

    # Update dataset CF metadata
    for name, var in sorted(
        nee_ds.items(),
        key=lambda name_var: (
            "{pre:s}_{name:s}".format(
                pre="z" if ("FC" in name_var[0] or "NEE" in name_var[0]) else "a",
                name=name_var[0],
            )
        ),
    ):
        physical_name = re.sub(AMF_BASE_VAR_NAME_REGEX, r"\g<physical_name>", name)

        if name != "SW_IN_CLEARSKY":
            # SW_IN_CLEARSKY is my variable, not AmeriFlux's
            var.attrs["description"] = unit_df.loc[physical_name, "Description"]
            var.attrs["units"] = unit_df.loc[physical_name, "Units"]

        if name == "FC" or name.startswith("FC_") or name.startswith("NEE"):
            var.attrs["standard_name"] = "surface_upward_mole_flux_of_carbon_dioxide"
            if "QC" in name or "SSITC_TEST" in name:
                var.attrs["standard_name"] += " status_flag"
                var.attrs.update(
                    {
                        "coverage_content_type": "qualityInformation",
                    }
                )
            else:
                var.attrs.update(
                    dict(
                        coverage_content_type="physicalMeasurement",
                        cell_methods="site: point TIMESTAMP_START: mean",
                        long_name="surface_upward_mole_flux_of_carbon_dioxide",
                        valid_range=[
                            -90.0,
                            +90.0,
                        ],  # I see -40 to 20 or so, with fuzzy spike around them
                    )
                )

                ds_for_flagging = nee_ds[[name]]
                for met_var in ["USTAR", "TA", "SW_IN"]:
                    trial_name = re.sub(
                        AMF_BASE_VAR_NAME_REGEX,
                        r"{met_var:s}\<loc_rep_agg>".format(met_var=met_var),
                        name,
                    )
                    if trial_name in nee_ds:
                        # Try to get at same level
                        found_name = trial_name
                    elif met_var in nee_ds:
                        # Try to get site-level
                        found_name = met_var
                    else:
                        # Get whatever I can find
                        found_names = [
                            trial_name
                            for trial_name in nee_ds.data_vars
                            if trial_name.startswith(met_var + "_")
                        ]
                        if found_names:
                            found_name = found_names[0]
                        else:
                            continue
                    ds_for_flagging[found_name] = nee_ds[found_name]

                    NEEDED_UNITS = {"TA": "degC", "USTAR": "meter/second"}
                    if met_var in NEEDED_UNITS:
                        try:
                            # TODO: Convert units earlier
                            ds_for_flagging[found_name] = (
                                ds_for_flagging[found_name]
                                .pint.quantify()
                                .pint.to(NEEDED_UNITS[met_var])
                                .pint.dequantify()
                            )
                        except (TypeError, ValueError):
                            if ds_for_flagging[found_name].attrs["units"].replace(
                                " ", ""
                            ) == NEEDED_UNITS[met_var].replace(" ", ""):
                                # Already converted
                                ds_for_flagging[found_name].attrs[
                                    "units"
                                ] = NEEDED_UNITS[met_var]
                            elif "-" in ds_for_flagging[found_name].attrs["units"]:
                                ds_for_flagging[found_name].attrs["units"] = (
                                    ds_for_flagging[found_name]
                                    .attrs["units"]
                                    .replace("-", "^-")
                                )

                                ds_for_flagging[found_name] = (
                                    ds_for_flagging[found_name]
                                    .pint.quantify()
                                    .pint.to(NEEDED_UNITS[met_var])
                                    .pint.dequantify()
                                )
                            else:
                                warnings.warn(
                                    "Have units {:s}, need units {:s}".format(
                                        ds_for_flagging[found_name].attrs["units"],
                                        NEEDED_UNITS[met_var],
                                    ),
                                )

                # Include only the time-series coords for time-series flagging
                for coord_name in ds_for_flagging.coords:
                    if "TIMESTAMP_START" not in ds_for_flagging.coords[coord_name].dims:
                        del ds_for_flagging.coords[coord_name]
                _LOGGER.debug(ds_for_flagging)
                df_for_flagging = ds_for_flagging.isel(site=0).to_dataframe()
                df_for_flagging = df_for_flagging.loc[
                    :, df_for_flagging.dtypes != object
                ]
                df_for_flagging.index = df_for_flagging.index.to_timestamp()
                mask_for_flagging = df_for_flagging.isna()
                df_for_flagging = df_for_flagging.fillna(-9999)
                _LOGGER.debug(df_for_flagging.dtypes)

                _LOGGER.info("Starting MAD spike flagging")
                outlier_flags = hesseflux.madspikes(
                    df_for_flagging,
                    mask_for_flagging,
                )
                nee_ds.coords["{name:s}_outlier_flag".format(name=name)] = (
                    ("site", "TIMESTAMP_START"),
                    outlier_flags[name]
                    .to_numpy(np.int8, na_value=-99)
                    .reshape(1, -1),
                    {
                        "standard_name": "spike_test_quality_flag",
                        "flag_values": np.array([0, 2], np.int8),
                        "flag_meanings": "good outlier",
                        "valid_range": np.array([0, 2], np.int8),
                        "comments": "Uses hesseflux implementation",
                        "references": """
Papale, D., M. Reichstein, M. Aubinet, E. Canfora, C. Bernhofer,
W. Kutsch, B. Longdoz, et al. 2006. “Towards a Standardized Processing
of Net Ecosystem Exchange Measured with Eddy Covariance Technique:
Algorithms and Uncertainty Estimation.” Biogeosciences 3 (4):
571–83. https://doi.org/10.5194/bg-3-571-2006.
""".strip(),
                    },
                )
                nee_ds[name].attrs["ancillary_variables"] = (
                    nee_ds[name].attrs.get("ancillary_variables", "") + " {name:s}_outlier_flag".format(name=name)
                ).strip()

                _LOGGER.info("Starting u-star flagging")
                ustar_thresh, ustar_flags = hesseflux.ustarfilter(
                    df_for_flagging,
                    mask_for_flagging,
                    nboot=N_BOOTSTRAP_SAMPLES,
                )
                _LOGGER.info("Creating u-star flag variables")
                nee_ds.coords["{name:s}_ustar_flag".format(name=name)] = (
                    ("site", "TIMESTAMP_START"),
                    ustar_flags.to_numpy(np.int8, na_value=-99).reshape(1, -1),
                    {
                        "standard_name": "multi_variate_test_quality_flag",
                        "flag_values": [0, 2],
                        "flag_meanings": "good ustar_below_threshold",
                        "comments": "Uses hesseflux implementation",
                        "references": """
Papale, D., M. Reichstein, M. Aubinet, E. Canfora, C. Bernhofer,
W. Kutsch, B. Longdoz, et al. 2006. “Towards a Standardized Processing
of Net Ecosystem Exchange Measured with Eddy Covariance Technique:
Algorithms and Uncertainty Estimation.” Biogeosciences 3 (4):
571–83. https://doi.org/10.5194/bg-3-571-2006.
""".strip(),
                    },
                )
                nee_ds[name].attrs["ancillary_variables"] += " {name:s}_ustar_flag".format(name=name)
                nee_ds.coords["{name:s}_ustar_thresholds".format(name=name)] = (
                    # Change "year" to "season" if seasonout=True
                    ("site", "quantile", "year"),
                    ustar_thresh.reshape(1, 3, -1),
                    {
                        "long_name": "ustar_threshold_and_confidence_interval",
                        "comments": "Uses hesseflux implementation",
                        "references": """
Papale, D., M. Reichstein, M. Aubinet, E. Canfora, C. Bernhofer,
W. Kutsch, B. Longdoz, et al. 2006. “Towards a Standardized Processing
of Net Ecosystem Exchange Measured with Eddy Covariance Technique:
Algorithms and Uncertainty Estimation.” Biogeosciences 3 (4):
571–83. https://doi.org/10.5194/bg-3-571-2006.
""".strip(),
                    },
                )
                nee_ds[name].attrs["ancillary_variables"] += " {name:s}_ustar_thresholds".format(name=name)
                nee_ds.coords["quantile"] = (
                    ("quantile",),
                    np.array([5, 50, 95], np.float32),
                    {
                        "long_name": "quantile_values",
                        "units": "percent",
                        "n_bootstrap_samples": N_BOOTSTRAP_SAMPLES,
                    },
                )
                nee_ds.coords["year"] = (
                    ("year",),
                    np.arange(
                        df_for_flagging.index.min().year,
                        df_for_flagging.index.max().year + 1,
                        dtype=np.int16,
                    ),
                    {
                        "standard_name": "time",
                        "axis": "T",
                        "units": "years since 0000-01-01T00:00:00Z",
                    },
                )

                if name.startswith("NEE"):
                    var.attrs.update(
                        {
                            "description": (
                                "Estimate of upward mole flux of carbon dioxide"
                                " at the surface"
                            ),
                        }
                    )
                else:
                    # name.startswith("FC")
                    var.attrs.update(
                        {
                            "description": (
                                "Eddy-covariance measurement of upward mole "
                                "flux of carbon dioxide at sensor level"
                            ),
                        }
                    )

                nee_ds.coords["raw_{:s}".format(name)] = (
                    var.dims,
                    var.data
                )
                # df_for_flagging.index = nee_ds.indexes["TIMESTAMP_START"]
                nee_ds[name] = (  # var.copy(False, df_for_flagging[name].values[np.newaxis, :])
                    # var.dims,
                    # var[outlier_flags + ustar_flags == 0]
                    var.where(
                        (
                            outlier_flags[name].to_numpy(np.int8, na_value=-99).reshape(1, -1)
                            + ustar_flags.to_numpy(np.int8, na_value=-99).reshape(1, -1)
                        ) == 0
                    )
                )
                nee_ds[name].attrs["ancillary_variables"] = "{} raw_{:s}".format(
                    nee_ds[name].attrs.get("ancillary_variables", ""),
                    name
                )
                # xarray.DataArray.from_series(df_for_flagging[name])

        # if "FCH4" in name:
        #     var.attrs["units"] = "nmol/m^2/s"
        # else:
        #     var.attrs["standard_name"] = "surface_upward_mole_flux_of_carbon_dioxide"
        #     var.attrs["units"] = "umol/m^2/s"

    qc_flag_names = [name for name in nee_ds if "_QC" in name or "_SSITC_TEST" in name]
    nee_ds = nee_ds.set_coords(qc_flag_names)
    for name in qc_flag_names:
        described_name = re.sub(
            AMF_BASE_VAR_NAME_REGEX, r"\g<physical_name>\g<loc_rep_agg>", name
        )
        described_var = nee_ds[described_name]
        var = nee_ds[name]
        described_var_ancillaries = described_var.attrs.get(
            "ancillary_variables", ""
        ).split()
        if name not in described_var_ancillaries:
            described_var_ancillaries.append(name)
            described_var.attrs["ancillary_variables"] = " ".join(described_var_ancillaries)
        var.attrs["standard_name"] = "quality_flag"
        if "_SSITC_TEST" in name:
            nee_ds[name] = var = var.astype(np.int8)
            nee_ds[described_name] = nee_ds[described_name].where(var < 2)
            var.attrs.update(
                {
                    "valid_range": np.array([0, 2], dtype=np.int8),
                    "flag_values": np.array([0, 1, 2], dtype=np.int8),
                    "flag_meanings": " ".join(
                        ["data_qc_good", "data_qc_fair", "data_qc_poor"]
                    ),
                    "description": """
Results of the quality flagging for {described_name:s} according to Foken et al
(2004), based on a combination of Steady State and Integral Turbulence
Characteristics tests by Foken and Wichura (1996) (i.e., 0, 1, 2)
""".format(
                        described_name=described_name
                    ).strip(),
                    "references": """
Foken, Th., and B. Wichura. 1996. “Tools for Quality Assessment of
Surface-Based Flux Measurements.” Agricultural and Forest Meteorology
78 (1–2): 83–105. https://doi.org/10.1016/0168-1923(95)02248-1.

Foken, Thomas, Mathias Göockede, Matthias Mauder, Larry Mahrt, Brian
Amiro, and William Munger. 2004. “Post-Field Data Quality Control.” In
Handbook of Micrometeorology: A Guide for Surface Flux Measurement and
Analysis, edited by Xuhui Lee, William Massman, and Beverly Law, 1st
ed., 181–208. Atmospheric and Oceanographic Sciences Library
29. Dordrecht: Springer
Netherlands. https://doi.org/10.1007/1-4020-2265-4_9.
""".strip(),
                }
            )

    for name in nee_ds.data_vars.keys():
        if (
            site_id in tower_height_da.indexes["Site_ID"]
            and name in tower_height_da.indexes["Variable"]
        ):
            tower_height_var = tower_height_da.sel(
                Site_ID=site_id,
                Variable=name,
                Start_Date=var.indexes["TIMESTAMP_START"].to_timestamp(),
            ).ffill("Start_Date")
            nee_ds.coords["height"] = (
                ("site", "TIMESTAMP_START"),
                tower_height_var.data.reshape(1, -1),
                {
                    "standard_name": "height",
                    "long_name": "sensor_height_above_ground",
                    "units": "meters",
                    "positive": "up",
                    "axis": "Z",
                },
            )

    optimum_name = min(nee_ds.data_vars.keys(), key=variable_sort_key)
    nee_ds["ameriflux_single_flux_variable_per_tower"] = (
        nee_ds[optimum_name]
    )
    nee_ds["single_variable_per_tower_name"] = (
        ("site",),
        np.array([optimum_name], dtype="S"),
    )

    return nee_ds


def save_dataset_netcdf(dataset, filename):
    # type: (xarray.Dataset, str) -> None
    """Save the dataset as a netcdf.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to save
    filename : str
        Name to save it under
    """
    # Avoid propagating changes to caller.  I could probably achieve a
    # similar effect with dataset.set_coords().
    dataset = dataset.copy()

    period_index = dataset.indexes["TIMESTAMP_START"]
    dataset.coords["time_bnds"] = xarray.DataArray(
        np.column_stack([period_index.start_time, period_index.end_time]),
        dims=("TIMESTAMP_START", "bnds2"),
        attrs=dict(standard_name="time", coverage_content_type="coordinate"),
    )
    dataset.coords["TIMESTAMP_START"] = xarray.DataArray(
        period_index.start_time,
        dims="TIMESTAMP_START",
        attrs=dict(
            standard_name="time",
            axis="T",
            bounds="time_bnds",
            long_name="start_of_observation_period",
            coverage_content_type="coordinate",
            freq=period_index.freqstr,
        ),
    )
    # if resolution == "half_hour":
    #     dataset.attrs["time_coverage_resolution"] = "P0000-00-00T00:30:00"
    # else:
    #     dataset.attrs["time_coverage_resolution"] = "P0000-00-00T01:00:00"
    dataset.attrs["time_coverage_resolution"] = period_index.freq.delta.isoformat()
    dataset.attrs["time_coverage_start"] = str(
        min(dataset.coords["TIMESTAMP_START"].values)
    )
    dataset.attrs["time_coverage_end"] = str(
        max(dataset.coords["TIMESTAMP_START"].values)
    )
    dataset.attrs["time_coverage_duration"] = operator.sub(
        *dataset.indexes["TIMESTAMP_START"][[-1, 0]]
    ).isoformat()
    dataset.coords["time_written"] = NOW_TIME.time().isoformat()
    dataset.coords["date_written"] = NOW_TIME.date().isoformat()

    encoding = {name: {"_FillValue": -9999, "zlib": True} for name in dataset.data_vars}
    encoding.update({name: {"_FillValue": None} for name in dataset.coords})

    # Set units for time variables with bounds
    for coord_name in dataset.coords:
        if "bounds" not in dataset.coords[coord_name].attrs:
            continue

        if "M8" in dataset.coords[coord_name].dtype.str:
            start_time = dataset.coords[coord_name].values[0]
            encoding[coord_name]["units"] = "minutes since {:s}".format(str(start_time))
            encoding[dataset.coords[coord_name].attrs["bounds"]]["units"] = encoding[
                coord_name
            ]["units"]

    dataset.to_netcdf(
        filename,
        encoding=encoding,
        format="NETCDF4",
        mode="w",
    )


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    logging.basicConfig(level=logging.INFO)
    BASE_FILE_NAMES = sorted(
        glob.glob(os.path.join(ARGS.ameriflux_root, "AMF_US-*_BASE-BADM_*-*.zip"))
    )
    SITE_NAMES = [os.path.basename(name).split("_")[1] for name in BASE_FILE_NAMES]

    # Find height of measurements
    MEASUREMENT_HEIGHT_DF = pd.read_csv(
        "BASE_MeasurementHeight_20211109.csv", dtype={"Start_Date": pd.Int64Dtype()}
    )
    US_NEE_INDEX = MEASUREMENT_HEIGHT_DF["Site_ID"].str.startswith(
        "US"
    ) & np.bitwise_or(
        *[
            MEASUREMENT_HEIGHT_DF["Variable"].str.startswith(var)
            for var in ["FC", "NEE"]
        ]
    )
    MEASUREMENT_HEIGHT_DF["Start_Date"] = pd.DatetimeIndex(
        [
            datetime.datetime.strptime(str(val), "%Y%m%d%H%M"[: len(str(val)) - 2])
            if "NA" not in str(val)
            else datetime.datetime(1970, 1, 1)
            for val in MEASUREMENT_HEIGHT_DF["Start_Date"]
        ]
    )
    MEASUREMENT_HEIGHT_DS = xarray.Dataset.from_dataframe(
        MEASUREMENT_HEIGHT_DF.loc[US_NEE_INDEX, :]
        .groupby(["Site_ID", "Variable", "Start_Date"])
        .first()
    )
    # It would be nice to also get instrument types out of this
    MEASUREMENT_HEIGHT_DA = (
        MEASUREMENT_HEIGHT_DS["Height"]
        .sel(
            # Start_Date=slice(CASA_START.split("+")[0], NOW.split("+")[0]),
            Variable=sorted(
                [
                    name
                    for name in MEASUREMENT_HEIGHT_DS.coords["Variable"].values
                    if name.startswith("FC") or name.startswith("NEE")
                ]
            ),
            Site_ID=sorted(
                [
                    name
                    for name in MEASUREMENT_HEIGHT_DS.coords["Site_ID"].values
                    if name.startswith("US-") and name in SITE_NAMES
                ]
            ),
        )
        .ffill("Start_Date")
        .resample(Start_Date="30min")
        .ffill()
        .sel(
            Start_Date=slice(
                CASA_START.split("+", maxsplit=1)[0], CASA_END.split("+", maxsplit=1)[0]
            )
        )
    ).sortby(["Variable", "Site_ID"])

    FLUX_UNIT_DF = (
        ARGS.flux_unit_df.set_index("Variable").sort_index().sort_index(axis=1)
    )

    # Read in data
    if HAVE_ZIP_NOT_NETCDF:
        # Only have ZIP files
        # TOWER_DATA = [
        #     parse_ameriflux(name, FLUX_UNIT_DF, MEASUREMENT_HEIGHT_DA)
        #     for name in BASE_FILE_NAMES
        #     # if os.path.exists(name.rsplit("BASE-BADM")[0] + "BIF_LATEST.xlsx")
        # ]
        # _LOGGER.info("Done reading in")
        # # Save the data I just read in, since it takes six hours to
        # # process
        # for ds, zip_name in zip(TOWER_DATA, BASE_FILE_NAMES):
        #     save_dataset_netcdf(ds, zip_name.replace(".zip", ".nc"))
        TOWER_DATA = []
        for zip_name in BASE_FILE_NAMES:
            # if not os.path.exists(name.rsplit("BASE-BADM")[0] + "BIF_LATEST.xlsx"):
            #     continue
            _LOGGER.info("Reading file %s", zip_name)
            tower_ds = parse_ameriflux(zip_name, FLUX_UNIT_DF, MEASUREMENT_HEIGHT_DA)
            save_dataset_netcdf(tower_ds, zip_name.replace(".zip", ".nc"))
    else:
        # Have converted ZIP files into netCDF files
        TOWER_DATA = []
        for zip_name in BASE_FILE_NAMES:
            dataset = xarray.open_dataset(
                zip_name.replace(".zip", ".nc"), decode_times=False
            )
            interval = np.diff(
                xarray.conventions.decode_cf_variable(
                    "time_bnds", dataset.coords["time_bnds"]
                ).values,
                axis=1,
            ).mean()
            if interval.astype("m8[m]") <= 30:
                ds_freq = "30min"
            else:
                ds_freq = "1H"

            period_index = pd.PeriodIndex(
                xarray.conventions.decode_cf_variable(
                    "TIMESTAMP_START", dataset.coords["TIMESTAMP_START"]
                ).values,
                freq=ds_freq,
            )
            dataset.coords["TIMESTAMP_START"] = (
                ("TIMESTAMP_START",),
                period_index,
                dataset.coords["TIMESTAMP_START"].attrs,
                dataset.coords["TIMESTAMP_START"].encoding,
            )
            del dataset.coords["time_bnds"]
            TOWER_DATA.append(dataset)

    _LOGGER.info("Done writing out preprocessed data")

    AMERIFLUX_DATA_VARS = sorted(
        {
            name
            for ds in TOWER_DATA
            for name in ds.data_vars
            if "_F" not in name and (name.startswith("FC") or name.startswith("NEE"))
        }
    )
    _LOGGER.info("Found flux variables [\n\t%s\n]", "\n\t".join(AMERIFLUX_DATA_VARS))

    HALF_HOUR_DATA = [
        df
        for df in TOWER_DATA
        if df.indexes["TIMESTAMP_START"].freqstr == HALF_HOUR_FREQ_STR
    ]
    HOUR_DATA = [
        df
        for df in TOWER_DATA
        if df.indexes["TIMESTAMP_START"].freqstr == HOUR_FREQ_STR
    ]

    def take_var_from_ds(source_dataset, flux_var, apply_qc=False):
        # type: (xarray.Dataset, str) -> xarray.DataArray
        """Take var from the dataset, dropping irrelevant aux vars.

        Parameters
        ----------
        source_dataset : xarray.Dataset
            The dataset from which to take the variable
        flux_var : str
            The flux variable to extract
        apply_qc : bool
            Drop data with bad QC? (>=2)

        Returns
        -------
        xarray.DataArray
        """
        result = source_dataset[flux_var]
        to_drop = [
            name
            for name in source_dataset.coords
            if re.sub(
                AMF_BASE_VAR_NAME_REGEX, r"\g<physical_name>\g<loc_rep_agg>", name
            )
            not in [flux_var, name]
        ]
        result = result.drop_vars(to_drop)
        if apply_qc:
            result = result.where("SSITC_TEST < 2")
        return result

    def harmonize_coords(da_lst):
        # type: (List[xarray.DataArray]) -> List[xarray.DataArray]
        """Make sure the datasets have the same coords."""
        coord_dims = {name: da.coords[name].dims for da in da_lst for name in da.coords}
        ancillary_variables = {
            name: dict(zip(da.coords[name].dims, da.coords[name].shape))
            for da in da_lst
            for name in da.attrs.get("ancillary_variables", "").split()
            if name in da.coords
        }
        result = [da.copy() for da in da_lst if da.size > 0]
        for da in result:
            for coord_name, coord_dim_list in coord_dims.items():
                if coord_name not in da.coords and all(
                    dim in da.dims for dim in coord_dim_list
                ):
                    da.coords[coord_name] = (coord_dim_list, np.full_like(da, np.nan))
            for aux_name, aux_dims in ancillary_variables.items():
                if aux_name not in da.coords:
                    da.coords[aux_name] = (aux_dims, np.full_like(da, np.nan))
                elif "site" not in da.coords[aux_name].dims:
                    da.coords[aux_name] = da.coords[aux_name].expand_dims(aux_dims)
            da.attrs["ancillary_variables"] = " ".join(ancillary_variables.keys())
        return result

    HOUR_DATAARRAYS = [
        xarray.concat(
            harmonize_coords(
                [take_var_from_ds(ds, amf_var) for ds in HOUR_DATA if amf_var in ds]
            ),
            dim="site",
            data_vars="all",
            coords="all",
            compat="no_conflicts",
            join="outer",
            combine_attrs="no_conflicts",
        )
        for amf_var in AMERIFLUX_DATA_VARS
        if any(amf_var in ds.data_vars for ds in HOUR_DATA)
    ]
    HOUR_DATASET = xarray.merge(HOUR_DATAARRAYS)
    HALF_HOUR_DATAARRAYS = [
        xarray.concat(
            harmonize_coords(
                [
                    take_var_from_ds(ds, amf_var)
                    for ds in HALF_HOUR_DATA
                    if amf_var in ds
                ]
            ),
            dim="site",
            data_vars="all",
            coords="all",
            compat="no_conflicts",
            join="outer",
            combine_attrs="no_conflicts",
        )
        for amf_var in AMERIFLUX_DATA_VARS
        if any(amf_var in ds for ds in HALF_HOUR_DATA)
    ]
    HALF_HOUR_DATASET = xarray.merge(HALF_HOUR_DATAARRAYS, compat="no_conflicts")

    SITE_DATA_LIST = []
    for resolution, dataset in zip(
        ("half_hour", "hour"), (HALF_HOUR_DATASET, HOUR_DATASET)
    ):
        _LOGGER.info("Adding metadata to %s-resolution data", resolution)
        dataset.coords["LOCATION_LAT"].attrs.update(
            dict(
                standard_name="latitude",
                long_name="tower_latitude",
                units="degrees_north",
                axis="Y",
                coverage_content_type="coordinate",
            )
        )
        dataset.coords["LOCATION_LONG"].attrs.update(
            dict(
                standard_name="longitude",
                long_name="tower_longitude",
                units="degrees_east",
                axis="X",
                coverage_content_type="coordinate",
            )
        )
        dataset.coords["LOCATION_ELEV"].attrs.update(
            {
                "standard_name": "altitude",
                "long_name": "elevation of site above sea level",
                "units": "meters",
                "axis": "Z",
                "coverage_content_type": "coordinate",
            }
        )
        dataset.coords["UTC_OFFSET"].attrs.update(
            dict(
                long_name="time zone offset from UTC",
                units="hours",
                source="AmeriFlux",
                coverage_content_type="auxiliaryInformation",
            )
        )
        dataset.coords["SITE_NAME"].attrs.update(
            {
                "standard_name": "platform_name",
                "long_name": "Name for AmeriFlux tower site",
                "description": "Human-readable site name",
            }
        )

        # Make sure everything's listed in the proper ancillary variable channels
        qc_flag_names = [
            name for name in dataset if "_QC" in name or "_SSITC_TEST" in name
        ]
        for name in qc_flag_names:
            described_name = re.sub(
                AMF_BASE_VAR_NAME_REGEX, r"\g<physical_name>\g<loc_rep_agg>", name
            )
            described_var = dataset[described_name]
            var = dataset[name]
            described_var_ancillaries = described_var.attrs.get(
                "ancillary_variables", ""
            ).split()
            if name not in described_var_ancillaries:
                described_var_ancillaries.append(name)
                described_var.attrs["ancillary_variables"] = " ".join(
                    described_var_ancillaries
                )

        # Update the global dataset attributes
        dataset.attrs.update(
            dict(
                feature_type="timeSeries",
                featureType="timeSeries",
                cdm_data_type="Station",
                ncei_template_version="NCEI_NetCDF_TimeSeries_Orthogonal_Template_v2.0",
                title="AmeriFlux CO2 flux data",
                summary="AmeriFlux CO2 flux data as provided by the website.",
                history="{now}: Convert from collection of CSV files to NetCDF".format(
                    now=NOW
                ),
                source="AmeriFlux network (ameriflux.ornl.gov)",
                # source="tower eddy covariance flux measurements",
                date_created=NOW,
                institution="AmeriFlux",
                project="AmeriFlux",
                program="AmeriFlux",
                date_modified=NOW,
                date_metadata_modified=NOW,
                comment="""I pulled one flux value for each tower,
preferring "FC" to "NEE_PI" to "FC_1_1_1" to "NEE_PI_1_1_1"
to "FC_1_2_1" to "NEE_PI_1_2_1" where multiple values were present.""",
                Conventions="CF-1.6 ACDD-1.3",
                geospatial_lat_min=min(dataset.coords["LOCATION_LAT"].values),
                geospatial_lat_max=max(dataset.coords["LOCATION_LAT"].values),
                geospatial_lon_min=min(dataset.coords["LOCATION_LONG"].values),
                geospatial_lon_max=max(dataset.coords["LOCATION_LONG"].values),
                standard_name_vocabulary="CF Standard Name Table v69",
                acknowledgment="""
AmeriFlux data were made available through the data portal
(https://ameriflux.lbl.gov) and processing maintained by the AmeriFlux
Management Project, supported by the U.S. Department of Energy Office
of Science, Office of Biological and Environmental Research, under
contract number DE-AC02-05CH11231.
""".strip(),
                license="""
AmeriFlux Legacy Data Policy

When you start in-depth analysis that may result in a publication,
contact the data contributors directly, so that they have the
opportunity to contribute substantively and become a co-author.

Data shared under the AmeriFlux Legacy Data Policy follow these
attribution guidelines:

    For each AmeriFlux site used: Provide a citation to the site’s
    data product that includes the data-product DOI and/or recommended
    publication.

    Acknowledge funding for site support if it was provided in the
    data download information.

    Acknowledge the AmeriFlux data resource: Funding for the AmeriFlux
    data portal was provided by the U.S. Department of Energy Office
    of Science.

Note:
The required citation with DOI and an email list of AmeriFlux PIs for
the downloaded sites is provided to the data user with the data
download. This information is also accessible via each site’s Site
Info page (See Site Search).

Inform all data providers when publications are about to be published.
""".strip(),
                keywords="carbon dioxide,eddy covariance,carbon dioxide flux",
                creator_name="Daniel Wesloh",
                creator_email="dfw5129@psu.edu",
                references="""
Novick, K. A., J. A. Biederman, A. R. Desai, M. E. Litvak,
D. J. P. Moore, R. L. Scott, and M. S. Torn. 2018. “The AmeriFlux
Network: A Coalition of the Willing.” Agricultural and Forest
Meteorology 249 (February): 444–56.
https://doi.org/10.1016/j.agrformet.2017.10.009.
""".strip(),
            )
        )
        save_dataset_netcdf(
            dataset,
            (
                "/abl/s0/Continent/dfw5129/ameriflux_netcdf/ameriflux_base_data"
                "/output/AmeriFlux_all_CO2_flux_values_{res:s}_data.nc4"
            ).format(res=resolution),
        )

        # Create single time series estimate
        sites_included = set()
        to_concat = []
        # ["FC", "NEE_PI", "FC_1_1_1", "NEE_PI_1_1_1", "FC_1_2_1", "NEE_PI_1_2_1"]
        # for i in [0, 3, 1, 4, 2, 5]:
        #     new_sites = set(datasets[i].coords["site"].values) - sites_included
        #     to_concat.append(datasets[i].sel(site=sorted(new_sites)))
        #     sites_included.update(new_sites)
        # for da in to_concat:
        #     da.name = "ameriflux_carbon_dioxide_flux_estimate"
        # big_da = xarray.concat(
        #     to_concat,
        #     "site",
        #     coords=AMERIFLUX_METADATA_TO_INCLUDE,
        #     compat="override",
        #     join="outer",
        # )

        big_da = dataset["NEE_PI"]
        for name in sorted(dataset.data_vars.keys(), key=variable_sort_key):
            if "FC" not in name and "NEE" not in name:
                continue
            if "_F" in name:
                continue
            big_da = big_da.combine_first(dataset[name])
        big_da.attrs.update(
            dict(
                coverage_content_type="physicalMeasurement",
                long_name="ameriflux_eddy_covariance_carbon_dioxide_flux_estimate",
                cell_methods="site: point TIMESTAMP_START: mean",
                ancillary_variables=" ".join(
                    name
                    for name in itertools.chain(
                        AMERIFLUX_METADATA_TO_INCLUDE, MEASUREMENT_METADATA_TO_INCLUDE
                    )
                    if not name.endswith("LAT")
                    and not name.endswith("LONG")
                    and not name.endswith("ELEV")
                ),
            )
        )
        dataset["ameriflux_carbon_dioxide_flux_estimate"] = big_da
        save_dataset_netcdf(
            dataset,
            (
                "/abl/s0/Continent/dfw5129/ameriflux_netcdf"
                "/ameriflux_base_data/output/"
                "AmeriFlux_all_CO2_fluxes_with_single_estimate_per_tower_"
                "{res:s}_data.nc4"
            ).format(res=resolution),
        )
        SITE_DATA_LIST.append(
            dataset.coords["site"]
            .to_dataset(name="SiteFID")
            .drop_vars(
                ["FC_SSITC_TEST", "time_written", "date_written"], errors="ignore"
            )
        )
    site_ds = xarray.concat(SITE_DATA_LIST, dim="site")
    site_df = site_ds.to_dataframe()
    site_df.to_csv(
        "/abl/s0/Continent/dfw5129/ameriflux_netcdf"
        "/ameriflux_base_data/output/"
        "AmeriFlux_site_data.csv"
    )
