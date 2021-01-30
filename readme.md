## Beyond annotations: labelling empathy in text from the perspective of the listener

[Paper](https://github.com/Zhilin123/Publications/blob/master/empathy.pdf)

This code is an example of how text can be labelled automatically based on social signals revealed in interpersonal communication.

Here, we label the empathy reveal in text based on Reddit conversations and how people responded to one another's messages.

1. **download_reddit.py**
  Download reddit conversations from the r/Advice subreddit

2. **postprocess_reddit.py** Process reddit conversation into a useful format

3. **predict_empathy.py** Predict the empathy of text based on whether Reddit user found the comment supportive to hear! Can be used with 4 different models

4. **predict_empathy_bert.py** Same as above but with a BERT model

5. **get_most_significant_predictors.py** See what the most significant predictors of empathy are
