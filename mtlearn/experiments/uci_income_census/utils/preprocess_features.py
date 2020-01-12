#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocess_features(df_train: pd.DataFrame,
                        df_val: pd.DataFrame, 
                        df_test: pd.DataFrame,
                        categorical_columns: Tuple[str],
                       ) -> Tuple[pd.DataFrame, LabelEncoder]:
  """
  Encode categorical features, represent them as dense set of integers and save 
  the mapping from c. Normalize all of numeric feature using min-max scaling,
  min and max are found from training data, scaling is applied to 
  training/val/test.

  Parameters
  ----------
  df_train: pd.DataFrame 
    Data frame with training data

  df_val: pd.DataFrame 
    Data frame with validation data
    
  df_test: pd.DataFrame
    Data frame with test data

  categorical_columns: tuple of str
    Tuple of categorical columns

  Returns
  -------
  all_data: list of pd.DataFrame instances
    Data frames corresponding to train / validation / test after encoding of
    categorical features and normalization of numerical ones.

  cat_feature_dims: dict
    Dictionary for which key is feature name and value is an number of 
    categories for that feature.
  """
  cat_feature_dims = {}
  all_columns = df_train.columns
  all_data = (df_train, df_val, df_test)
  for col in all_columns:
    if col in categorical_columns:
      feature = np.concatenate([df[col].values for df in all_data])
      le = LabelEncoder().fit(feature)
      for df in all_data:
        df[col] = le.transform(df[col].values)
      cat_feature_dims[col] = len(le.classes_)
    else:
      train_feature = df_train[col].values
      min_val, max_val = np.min(train_feature), np.max(train_feature)
      scaler = lambda min_x, max_x, x: (x-min_x)/(max_x-min_x)
      for df in all_data:
        feature = scaler(min_val, max_val, df[col].values)
        df[col] = np.array(feature, dtype=np.float32)
  return all_data, cat_feature_dims