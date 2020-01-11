#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

from experiments.uci_income_census.utils.load_census_data import load_census_data_uci
from experiments.uci_income_census.utils.preprocess_features import preprocess_features

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ('class_worker', 'det_ind_code', 'det_occ_code',
                       'hs_college', 'major_ind_code', 'major_occ_code',
                       'race', 'hisp_origin', 'sex', 'union_member',
                       'unemp_reason', 'full_or_part_emp', 'tax_filer_stat',
                       'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                       'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg',
                       'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                       'fam_under_18', 'country_father', 'country_mother',
                       'country_self', 'citizenship', 'vet_question')

COLUMNS = ('age', 'class_worker', 'det_ind_code', 'det_occ_code',
           'education', 'wage_per_hour', 'hs_college', 'marital_stat',
           'major_ind_code', 'major_occ_code', 'race', 'hisp_origin',
           'sex', 'union_member', 'unemp_reason', 'full_or_part_emp',
           'capital_gains', 'capital_losses', 'stock_dividends', 'tax_filer_stat',
           'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
           'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg',
           'mig_same', 'mig_prev_sunbelt', 'num_emp', 'fam_under_18',
           'country_father', 'country_mother', 'country_self', 'citizenship',
           'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked',
           'year', 'income_50k')

UCI_CENSUS_TRAIN_DATA = ("https://archive.ics.uci.edu/ml/machine-learning-"
                         "databases/census-income-mld/census-income.data.gz")

UCI_CENSUS_TEST_DATA = ("https://archive.ics.uci.edu/ml/machine-learning-"
                        "databases/census-income-mld/census-income.test.gz")


class DataLoaderPreprocessorCensusUCI:
    """
    Loads and preprocesses data the way it is decribed in section 6.3 of
    the paper.

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

    References
    ----------
    [1] Modeling Task Relationships in Multi-task Learning with
        Multi-gate Mixture-of-Experts, Jiaqi Ma1 et al, 2018 KDD
    """

    def __init__(self,
                 local_path: str = None,
                 write_data_locally: bool = True,
                 columns: Tuple[str] = COLUMNS,
                 uci_train_link_path: str = UCI_CENSUS_TRAIN_DATA,
                 uci_test_link_path: str = UCI_CENSUS_TEST_DATA,
                 categorical_columns: Tuple[str] = CATEGORICAL_COLUMNS,
                 seed: int = 0
                 ):
        # load data
        self.train, self.val, self.test = load_census_data_uci(local_path,
                                                               write_data_locally,
                                                               columns,
                                                               uci_train_link_path,
                                                               uci_test_link_path,
                                                               seed
                                                               )

        # split features and targets, remove education, income_50K and marital_stat
        # from the features as described in section 6.3
        (train_features,
         val_features,
         test_features,
         self.education,
         self.marital_stat,
         self.income) = self._split_features_targets(self.train, self.val, self.test)

        # transform categorical features into dense indices and memorize
        # their dimensionality, normalize numeric features to 0-1 scale
        (self.features,
         self.cat_features_dim) = preprocess_features(train_features,
                                                      val_features,
                                                      test_features,
                                                      categorical_columns)

    def _split_features_targets(self,
                                df_train: pd.DataFrame,
                                df_val: pd.DataFrame,
                                df_test: pd.DataFrame):
        """
        Split feature from targets, remove education, income_50K and
        marital_stat from the features as described in section 6.3
        """
        data = (df_train, df_val, df_test)
        education, marital_stat, income = [], [], []
        for df in data:
            education.append(df.education.values)
            marital_stat.append(df.marital_stat.values)
            income.append(df.income_50k.values)
            df.drop(["marital_stat", "income_50k", "education"], axis=1, inplace=True)
        return (*data, education, marital_stat, income)

    def get_education_marital_stat_tasks(self):
        """
        Get group two data as described in section 6.3 of the paper. Second group of
        data contains two binary target variables:
        1] Marital Status (1 if "Never married", 0 otherwise)
        2] Education level (1 if college or higher, 0 otherwise)
        explanatory variables are all the columns of the original dataset with the
        exception of marital status, income, education level and instance weight.
        """
        college_educated = [' Bachelors degree(BA AB BS)',
                            ' Masters degree(MA MS MEng MEd MSW MBA)',
                            ' Doctorate degree(PhD EdD)',
                            ' Prof school degree (MD DDS DVM LLB JD)'
                            ]
        y_marital, y_education = [], []
        for ms, edu in zip(self.marital_stat, self.education):
            y_marital.append(np.array(1 * (ms == "Never married"), dtype=np.float32))
            educated = [1 * (e in college_educated) for e in edu]
            y_education.append(np.array(educated, dtype=np.float32))
        return self.features, y_marital, y_education, self.cat_features_dim

    def get_income_marital_stat_tasks(self):
        """
        Get group one data as described in section 6.3 of the paper. First group of
        data contains two binary target variables:
        1] Marital Status (1 if "Never married", 0 otherwise)
        2] Income level (1 if above 50K, 0 otherwise)
        explanatory variables in the first group are all the columns of the original
        dataset with the exception of marital status, income, education level and
        instance weight.
        """
        y_marital, y_income = [], []
        for ms, income in zip(self.marital_stat, self.income):
            y_marital.append(np.array(1 * (ms == " Never married"), dtype=np.float32))
            y_income.append(np.array(1 * (income == ' 50000+.'), dtype=np.float32))
        return self.features, y_marital, y_income, self.cat_features_dim

    def get_raw_train_val_test(self):
        """ Get raw data """
        return self.train, self.val, self.test


dl = DataLoaderPreprocessorCensusUCI()
