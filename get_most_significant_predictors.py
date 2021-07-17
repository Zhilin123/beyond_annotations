from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import copy
from scipy import stats
import json
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator, get_single_color_func

import matplotlib.pyplot as plt

def get_se_and_wald(X,y):
    logit = LogisticRegression()
    resLogit = logit.fit(X, y)
    predProbs = resLogit.predict_proba(X)
    V = np.diagflat(np.product(predProbs, axis=1))
    covLogit = np.linalg.inv(X.T @ V @ X)
    std_erros = np.sqrt(np.diag(covLogit))
    wald_stats = (resLogit.coef_ / np.sqrt(np.diag(covLogit))) ** 2
    return std_erros, wald_stats, resLogit.coef_

def get_p_value(z_score,two_side=False):
    if z_score > 0:
        return (1-stats.norm.cdf(z_score)) / (int(two_side) + 1)
    else:
        return (stats.norm.cdf(z_score)) / (int(two_side) + 1)

def format_float(num):
    return "%.4f" % num

def print_key_info(feature_index):
    word = feature_names[feature_index]
    beta = format_float(coef[0][feature_index])
    std_error = format_float(std_erros[feature_index])
    std_beta = format_float(Z_stats[0][feature_index])
    p_val = format_float(get_p_value(Z_stats[0][feature_index]))
    print(f"Word: {word}, Beta: {beta}, Std Error: {std_error}, Std Beta: {std_beta}, P value: {p_val}")
    return word, std_beta

def get_most_significant_correlated(n=50, negative=False):
    #negative:bool --> whether to look for most significant positive or negative correlations

    if negative:
        print("Most negatively correlated words")
    else:
        print("Most positively correlated words")

    word_to_abs_std_beta = {}
    for i in range(n):
        if negative:
            feature_index = Z_ordered[0][i]
        else:
            feature_index = Z_ordered[0][-(i+1)]
        word, std_beta = print_key_info(feature_index)
        # to show on word cloud
        word_to_abs_std_beta[word] = abs(float(std_beta))
    return word_to_abs_std_beta


def draw_wordcloud(value_map):
    wordcloud = WordCloud(prefer_horizontal = 1, background_color = "white").generate_from_frequencies(value_map)
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()



if __name__ == "__main__":
    '''
    The original dataset (helpful_comments.json and unhelpful_comments.json)
    are not used for identifying significant predictors.
    This is because for those dataset, X.T @ V @ X is a singular matrix
    To overcome this issue, we randomly sampled one comment from each post and author
    to reduce the risk of collinearity (which contributes to the singularity).
    Details of this sampling are available in postprocess_reddit.py
    '''
    data_dir = "data/"
    
    with open(data_dir+"helpful_comments_one_post_one_author.json", "r") as read_file:
        helpful_comments = json.load(read_file)

    with open(data_dir+"unhelpful_comments_one_post_one_author.json", "r") as read_file:
        unhelpful_comments = json.load(read_file)

    positive_examples = helpful_comments
    negative_examples = unhelpful_comments

    X = positive_examples + negative_examples
    vect = CountVectorizer(min_df=10)
    X = vect.fit_transform(X)
    y = [1]*len(positive_examples) + [0]*len(negative_examples)

    std_erros, wald_stats, coef = get_se_and_wald(X,y)

    Z_stats = np.array([i for i in coef/ std_erros])

    Z_ordered = np.argsort([i for i in Z_stats]) #  coef Z_stats

    feature_names = vect.get_feature_names()

    most_negative_correlated_words = get_most_significant_correlated(n=50, negative=True)
    draw_wordcloud(most_negative_correlated_words)

    most_positive_correlated_words = get_most_significant_correlated(n=50, negative=False)
    draw_wordcloud(most_positive_correlated_words)
