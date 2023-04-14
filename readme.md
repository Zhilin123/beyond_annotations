This code is an example of how text can be labelled automatically based on social signals revealed in interpersonal communication.

Here, we label the helpfulness reveal in text based on Reddit conversations and how people responded to one another's messages.

1. **download_reddit.py**
  Download reddit conversations from the r/Advice subreddit

2. **postprocess_reddit.py** Process reddit conversation into a useful format

3. **predict_helpfulness.py** Predict the helpfulness of text based on whether Reddit user found the comment supportive to hear! Can be used with 4 different models

4. **predict_helpfulness_bert.py** Same as above but with a BERT model

5. **get_most_significant_predictors.py** See what the most significant predictors of helpfulness are

6. **correlate_labeled_features_against_helpfulness.py** Correlate between manually labelled features of helpfulness against comment helpfulness
