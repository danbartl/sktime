# -*- coding: utf-8 -*-
"""Functions to load datasets included in sktime."""

__all__ = [
    "load_airline",
    "load_arrow_head",
    "load_gunpoint",
    "load_basic_motions",
    "load_osuleaf",
    "load_italy_power_demand",
    "load_japanese_vowels",
    "load_longley",
    "load_lynx",
    "load_shampoo_sales",
    "load_UCR_UEA_dataset",
    "load_unit_test",
    "load_uschange",
    "load_PBS_dataset",
    "load_japanese_vowels",
    "load_gun_point_segmentation",
    "load_electric_devices_segmentation",
    "load_acsf1",
    "load_macroeconomic",
    "load_from_tsfile",
    "load_from_tsfile_to_dataframe",
    "TsFileParseException",
]
from sktime.datasets._data_io import (
    TsFileParseException,
    load_acsf1,
    load_airline,
    load_arrow_head,
    load_basic_motions,
    load_electric_devices_segmentation,
    load_from_tsfile,
    load_from_tsfile_to_dataframe,
    load_gun_point_segmentation,
    load_gunpoint,
    load_italy_power_demand,
    load_japanese_vowels,
    load_longley,
    load_lynx,
    load_macroeconomic,
    load_osuleaf,
    load_PBS_dataset,
    load_shampoo_sales,
    load_UCR_UEA_dataset,
    load_unit_test,
    load_uschange,
)
