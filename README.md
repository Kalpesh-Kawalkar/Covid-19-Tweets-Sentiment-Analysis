# Covid_19_Tweets_Sentiments_Predictor
Created a classification model to predict the sentiment either Positive or
Negative based on Covid Tweets.

# Context
The tweets have been pulled from Twitter and manual tagging has been done then.
The names and usernames have been given codes to avoid any privacy concerns.

# Details of Features
The columns are described as follows:
1. UserName: UserName in encrypted numbers
2. ScreenName: ScreenName in encrypted numbers
3. Location: Country from where tweet was pulled from
4. TweetAt: Twee time
5. OriginalTweet: Tweet content
6. Sentiment: Positive, Negative, Neutral, Extremely Positive, Extremely Negative

# Libraries
Here we primarily used the following libraries from Python.
1. numpy
2. pandas
3. matplotlib

# Approach
1. Preprocessing the Covid tweets based on the following parameter:
a) Tokenizing words
b) Convert words to lower case
c) Removing Punctuations
d) Removing Stop words
e) Stemming or lemmatizing the words

2. Building 3 classification models in Python and identifying which works the best for our data. The 3 models are:
a) Multinomial Naïve Bayes Classification
b) SVM Classification 
c) KNN Classification

3. We used Hyperparameter Tuning to find the best parameters for SVM by using Random Grid Search method.

# Conclusion
After comparing all the models we can conclude that the SVM model with linear kernel works best on our data with an accuracy of about 74%.
The confusion matrix and classification report of the same is present in the text file best_model.txt
