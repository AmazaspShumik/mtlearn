#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# marital_status education and income_50k are also categorical features but we
# use them as targets, so they are not included in either of groups 
# (we follow the paper here as well).
CATEGORICAL_COLUMNS = ('class_worker',
                       'det_ind_code',
                       'det_occ_code',
                       'hs_college',
                       'major_ind_code',
                       'major_occ_code',
                       'race',
                       'hisp_origin',
                       'sex',
                       'union_member',
                       'unemp_reason',
                       'full_or_part_emp',
                       'tax_filer_stat',
                       'region_prev_res',
                       'state_prev_res',
                       'det_hh_fam_stat',
                       'det_hh_summ',
                       'mig_chg_msa', 
                       'mig_chg_reg',
                       'mig_move_reg',
                       'mig_same',
                       'mig_prev_sunbelt',
                       'fam_under_18',
                       'country_father',
                       'country_mother', 
                       'country_self',
                       'citizenship',
                       'vet_question')


def encode_cat_features(*data,
                        categorical_columns: Tuple[str]=CATEGORICAL_COLUMNS
                        ) -> Tuple[pd.DataFrame, LabelEncoder]:
  """
  Encode categorical features, represent them as integers and save the mapping
  from category to integer.

  Parameters
  ----------
  *data: instances of pd.DataFrame 
    Data frames with training / validation and test datasets

  categorical_columns: tuple of str
    Tuple of categorical columns

  Returns
  -------
  data: list of pd.DataFrame instances
    Data frames corresponding to train / validation / test after encoding of
    categorical features

  feature_encoders: dict
    Dictionary for which key is feature name and value is an instance of 
    LabelEncoder that holds mapping from category to integer.
  """
  feature_encoders = {}
  for col in categorical_columns:
    feature = np.concatenate([df[col].values for df in data])
    le = LabelEncoder().fit(feature)
    for df in data:
      df[col] = le.transform(df[col].values)
    feature_encoders[col] = le
  return data, feature_encoders


def data_group_one(*data,
                   categorical_columns: Tuple[str]=CATEGORICAL_COLUMNS
                   ) -> Tuple[List[pd.DataFrame],
                              np.array,
                              np.array,
                              Dict[str, LabelEncoder]
                             ]:
  """
  Get group one data as described in section 6.3 of the paper. First group of 
  data contains two binary target variables:
   1] Marital Status (1 if "Never married", 0 otherwise)
   2] Income level (1 if above 50K, 0 otherwise)
  explanatory variables in the first group are all the columns of the original
  dataset with the exception of marital status, income, education level and 
  instance weight.

  Parameters
  ----------
  *data: instances of pd.DataFrame 
    Data frames with training / validation and test datasets

  categorical_columns: tuple of str
    Tuple of categorical columns

  Returns
  -------
  transformed_data: list of pd.DataFrame instances
    Data frames corresponding to train / validation / test after encoding of
    categorical features

  y_marital: np.array
    Binary target variable identifying whether the person was ever married.

  y_income: np.array
    Binary target variable identifying whether the persone has income above or
    below 50k.

  feature_encoders: dict
    Dictionary for which key is feature name and value is an instance of 
    LabelEncoder that holds mapping from category to integer.
  """
  # extract two target variables: marital status and income above/below 50K
  y_marital, y_income, data_frames = [], [], []
  for df in data:
    y_marital.append( 1*(df.marital_stat.values == " Never married") )
    y_income.append( 1*(df.income_50k.values == ' - 50000.'))
    data_frames.append( df.drop(["marital_stat","income_50k","education"], 
                                axis=1, 
                                inplace=False))  
  # encode all categorical features   
  transformed_data, feature_encoders = encode_cat_features(*data_frames)
  return transformed_data, y_marital, y_income, feature_encoders


def data_group_two(*data,
                   categorical_columns: List[str]=CATEGORICAL_COLUMNS
                   ) -> Tuple[List[pd.DataFrame],
                              np.array,
                              np.array,
                              Dict[str, LabelEncoder]
                             ]:
  """
  Get group one data as described in section 6.3 of the paper. Second group of 
  data contains two binary target variables:
   1] Marital Status (1 if "Never married", 0 otherwise)
   2] Education level (1 if college or higher, 0 otherwise)
  explanatory variables in the first group are all the columns of the original
  dataset with the exception of marital status, income, education level and 
  instance weight.

  Parameters
  ----------
  *data: instances of pd.DataFrame 
    Data frames with training / validation and test datasets

  categorical_columns: tuple of str
    Tuple of categorical columns

  Returns
  -------
  transformed_data: list of pd.DataFrame instances
    Data frames corresponding to train / validation / test after encoding of
    categorical features

  y_marital: np.array
    Binary target variable identifying whether the person was ever married.

  y_income: np.array
    Binary target variable identifying whether the persone has income above or
    below 50k.

  feature_encoders: dict
    Dictionary for which key is feature name and value is an instance of 
    LabelEncoder that holds mapping from category to integer.
  """
  # extract two target variables: marital status and education
  y_marital, y_education, data_frames = [], [], []
  college_educated = [' Bachelors degree(BA AB BS)',
                      ' Masters degree(MA MS MEng MEd MSW MBA)',
                      ' Doctorate degree(PhD EdD)',
                      ' Prof school degree (MD DDS DVM LLB JD)'
                      ]
  for df in data:
    y_marital.append( 1*(df.marital_stat.values == "Never married") )
    college_educated = [1*(e in college_educated) for e in df.education.values]
    y_education.append( np.asarray(college_educated) )
    data_frames.append( df.drop(["marital_stat","income_50k","education"], 
                                axis=1, 
                                inplace=False))  
  # encode all categorical features   
  transformed_data, feature_encoders = encode_cat_features(*data_frames)
  return transformed_data, y_marital, y_education, feature_encoders   
        