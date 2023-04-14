#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import statsmodels.api as sm
import numpy as np
import pandas as pd

data_dir = "data/"

csv_filename = data_dir+ "manual_annotation_five_comments_per_author.csv"

df = pd.read_csv(csv_filename, engine="python")

col_names = list(df.columns)

helpfulness_comment = list(df['comment_helpfulness'])

feature_names = col_names[-11:-1]
print(feature_names)

def get_one_feature(feature_name):
    feature = list(df[feature_name])
    feature = [i if i == 1 else 0 for i in feature]
    return feature

def get_all_features(feature_names):
    all_features = [get_one_feature(feature_name) for feature_name in feature_names]
    return np.array(all_features).T

all_features = get_all_features(feature_names)

y = helpfulness_comment
X = all_features
X2 = sm.add_constant(X)
est = sm.Logit(y, X2)
est2 = est.fit()
print(est2.summary())


