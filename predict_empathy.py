#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def main(classifier_name, data_dir, seed):

    with open(data_dir+"helpful_comments.json", "r") as read_file:
        helpful_comments = json.load(read_file)

    with open(data_dir+"unhelpful_comments.json", "r") as read_file:
        unhelpful_comments = json.load(read_file)

    all_comments = helpful_comments + unhelpful_comments
    len_all_comments =[len(i.split(' ')) for i in all_comments]

    print("mean : ", np.mean(len_all_comments))
    print("std : ", np.std(len_all_comments))
    print("len: ", len(len_all_comments))

    positive_examples = helpful_comments
    negative_examples = unhelpful_comments

    X = positive_examples + negative_examples
    #X_original = copy.copy(X)

    y = [1]*len(positive_examples) + [0]*len(negative_examples)
    vect = CountVectorizer(min_df=10)

    X = vect.fit_transform(X)
    
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=seed)
    
    if classifier_name == "nb":
        clf = MultinomialNB()
    elif classifier_name == "lr":
        clf = LogisticRegression()
    elif classifier_name == "svc":
        clf = LinearSVC()
    elif classifier_name == "rf":
        clf = RandomForestClassifier(n_estimators=100)

    clf.fit(X_train, y_train)
    
    y_pred_val = clf.predict(X_val)
    print("val: ",metrics.classification_report(y_val,y_pred_val, output_dict=True)['weighted avg']['f1-score']) #

    y_pred = clf.predict(X_test)
    print("test: ", metrics.classification_report(y_test,y_pred, output_dict=True)['weighted avg']['f1-score']) #

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run classification')
    parser.add_argument("classifier",
                        choices=['rf', 'nb','svc','lr'],
                        help='set classifier rf (random forest), nb (naive bayes), svc (support vector classifier), lr (logistic regression), ',
                        default='lr')
    parser.add_argument("--seed", default="42")
    args = parser.parse_args()
    data_dir = "data/"
    main(args.classifier, data_dir, int(args.seed))
