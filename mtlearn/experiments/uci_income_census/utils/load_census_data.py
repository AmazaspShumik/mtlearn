#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

def load_census_data_uci(local_path: str,
                         write_data_locally: bool,
                         columns: Tuple[str],
                         uci_train_link_path: str,
                         uci_test_link_path: str,
                         seed: int=0
                         ) -> Tuple[pd.DataFrame]:
  """ 
  Checks whether data are in path directory or in current directory if path is 
  not specified, loads data from the uci and splits test set into 
  new test and validation sets (in the proportion 1:1) as described in the 
  paper. 
  
  Parameters
  ----------
  local_path: str, optional (default=None)
    Local path where all the data are stored, if not provided current directory
    will be used.

  uci_train_link_path: str, optional (default=UCI_CENSUS_TRAIN_DATA)
    Path to training data (in UCI Machine Learning Datasets)

  uci_test_link_path: str, optional (default=UCI_CENSUS_TEST_DATA)
    Path to original test data (in UCI Machine Learning Datasets). Note that 
    in the paper original test data are split into validation and test with ratio
    1:1.

  write_data_locally: bool, optional (default=True)
    If True will write train and original test datasets to local path if they 
    are not there already.

  columns: list of str
    Names of the columns in dataset

  seed: int
    Seed for the random train test split

  Returns
  -------
  : tuple of pandas.DataFrame 
    Tuple of data frames (train, validation, test)

  References
  ----------
  [1] Modeling Task Relationships in Multi-task Learning with 
      Multi-gate Mixture-of-Experts, Jiaqi Ma1 et al, 2018 KDD
  """
  train_file, test_file = "census-income.data", "census-income.test"
  if not local_path:
    local_path = os.getcwd()
  train_file_path = os.path.join(local_path, train_file)
  test_file_path = os.path.join(local_path, test_file)
  
  # load training data 
  if train_file in os.listdir(local_path):
    data_train = pd.read_csv(train_file_path, 
                             index_col=0)
  else:
    data_train = pd.read_csv(uci_train_link_path, 
                             header=None, 
                             index_col=False, 
                             names=columns)
    
  # load test data 
  if test_file in os.listdir(local_path):
     data_test_orig = pd.read_csv(test_file_path, 
                                  index_col=0)
  else:
     data_test_orig = pd.read_csv(uci_test_link_path, 
                                  header=None, 
                                  index_col=False, 
                                  names=columns)
  assert list(data_train.columns) == list(data_test_orig.columns)


  # write loaded data to local path if they are not there already
  if write_data_locally:
    if train_file not in os.listdir(local_path):
      data_train.to_csv(train_file_path)
    if test_file not in os.listdir(local_path):
      data_test_orig.to_csv(test_file_path)

  # drop instance weight ( in section 6.3 of the paper there is no mention
  # of using instance_weight as sample weights for the loss functions)
  data_train.drop("instance_weight", axis=1, inplace=True)
  data_test_orig.drop("instance_weight", axis=1, inplace=True)

  # split original test data into training and validation as mentioned 
  # in the paper
  data_val, data_test = train_test_split(data_test_orig, test_size=0.5, random_state=seed)
  return data_train, data_val, data_test
