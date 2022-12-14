#!/usr/bin/env python
# coding: utf-8

# # Project 3: Reddit Scraping & NLP
# 
# ## Part 3 - Preprocessing and Modeling

# In[2]:


# General imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# For Natural Language Processing
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# For Classification Modeling
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# For Evaluation
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, 
                             plot_roc_curve, roc_auc_score, classification_report,
                            precision_score, recall_score, f1_score)


# ### Preprocessing

# In[3]:


# Read in cleaned data and reset index
df = pd.read_csv('../data/cleaned_posts.csv', index_col=False)
df.reset_index(inplace=True, drop=True)


# In[4]:


df.head()


# In[5]:


# Check for missing values in title
df.title.isnull().sum()


# In[6]:


df2 = pd.read_csv('../data/shared_words.csv')


# In[7]:


shared_words = list(df2['shared_word'])


# In[8]:


print(shared_words)


# In[9]:


# Create my_stopwords model parameter combining English stop words and subreddits shared words
my_stopwords = list(TfidfVectorizer(stop_words = 'english').get_stop_words()) + shared_words


# ### Modeling
# 
# #### The Null Model

# In[10]:


# Define X and y
X = df['title']
y = df['is_mental']


# In[11]:


# Check class distribution
y.value_counts(normalize=True)


# We can see here that 50.04% of posts are from r/mentalhealth, and 49.96% of posts are from r/CoronavirusUS. The null model is showing that selecting the label r/mentalthealth would result in correctly predicting what subreddit a post came from 50.04% of the time. My goal is to build a model that classifies posts with an accuracy higher than 50.04%.

# #### Model 1: Logistic Regression

# In[12]:


# Define X and y
X = df['title']
y = df['is_mental']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Set up transformer and estimator via pipeline
pipe_lr = Pipeline([
    ('tvec', TfidfVectorizer(token_pattern=r'[\w{2,}\d*]\S*\w+')),
    ('logreg', LogisticRegression())
])

# Set up parameters for pipeline
lr_params = {
    'tvec__stop_words': [None, 'english', my_stopwords],
    'tvec__min_df': [1, 5, 10],
    'tvec__ngram_range': [(1,1), (1,2), (2,2)],
    'logreg__C': [0.1, 1, 10]  
}

# Instantiate GridSearchCV
gs_lr = GridSearchCV(estimator=pipe_lr,
                      param_grid=lr_params,
                      cv=5,
                      verbose=0)

# Fit GridSearchCV
gs_lr.fit(X_train, y_train)

# Accuracy score
gs_lr.score(X_train, y_train), gs_lr.score(X_test, y_test)


# In[13]:


# Predicted y test values
lr_preds = gs_lr.predict(X_test)

# Plot a confusion matrix
ConfusionMatrixDisplay.from_estimator(gs_lr, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');


# In[14]:


# Model evaluation metrics
print(classification_report(y_test, lr_preds, target_names=['MH', 'COVID']))


# In[15]:


# # Best parameters (uncomment to run)
# gs_lr.best_params_


# **Interpretation:** Model 1 is a logistic regression model that utilizes TF-IDF vectorizer and a pipline/gridsearch in order to predict the correct subreedit of a post. TF-IDF vectoriziation is used because it better differentiates rare words and gives weights to all of the words, unlike Countvectorizer which gives equal weights to all words. The most desirable parameters that maximizes accuracy include 'english" as the stop words, and an ngram range of (1,2), meaning the model creates unigram and bigram vectors. The testing accuracy score of 92.1% on unseen data means that this model surpasses the baseline model in its ability to classify reddit posts. However, given a training score of 99.86%, this model is clearly overfit. Between precision and recall, we are more concerned with not missing r/mentalhealth posts, and this model returns a recall score of .90, which is good. Overall, this model is still largely robust in it's ability to predict the correct subreddit of a post.

# #### Model 2: Decision Tree Classifier

# In[16]:


# Define X and y
X = df['title']
y = df['is_mental']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Set up pipeline
pipe_dtc = Pipeline([
    ('tvec', TfidfVectorizer(token_pattern=r'[\w{2,}\d*]\S*\w+')),
    ('dct', DecisionTreeClassifier(min_samples_leaf=2))
])

# Set up parameters for pipeline
dtc_params = {
    'tvec__stop_words': [None, 'english', my_stopwords],
    'tvec__min_df': [1, 5, 10],
    'tvec__ngram_range': [(1,1), (1,2), (2,2)],
    'dct__max_depth': [None, 10, 20, 40, 100]
}

# Instantiate GridSearchCV
gs_dtc = GridSearchCV(estimator=pipe_dtc,
                      param_grid=dtc_params,
                      cv=5,
                      verbose=0)

# Fit GridSearchCV
gs_dtc.fit(X_train, y_train)

# Accuracy score
gs_dtc.score(X_train, y_train), gs_dtc.score(X_test, y_test)


# In[17]:


# Predicted y test values
dtc_preds = gs_dtc.predict(X_test)

# Plot a confusion matrix
ConfusionMatrixDisplay.from_estimator(gs_dtc, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');


# In[18]:


# Model evaluation metrics
print(classification_report(y_test, dtc_preds, target_names=['MH', 'COVID']))


# In[19]:


# # Best parameters (uncomment to run)
# gs_dtc.best_params_


# **Interpretation:** Model 2's testing accuracy of 89.5% on unseen data means that this model surpasses the baseline model in its ability to classify reddit posts. Given a training score of 90.3%, this model shows a balanced bias-variance tradeoff. Between precision and recall, we are more concerned with not missing r/mentalhealth posts, and this model returns a recall score of .81, which is good, but not as high as Model 1.

# #### Model 3: Bagging Classifier

# In[20]:


# Define X and y
X = df['title']
y = df['is_mental']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Set up pipeline
pipe_bag = Pipeline([
    ('tvec', TfidfVectorizer(token_pattern=r'[\w{2,}\d*]\S*\w+')),
    ('bag', BaggingClassifier())
])

# Set up parameters for pipeline
bag_params = {
    'tvec__stop_words': [None, 'english', my_stopwords],
    'tvec__min_df': [1, 5, 10],
    'tvec__ngram_range': [(1,1), (1,2), (2,2)],
    'bag__n_estimators': [10, 15, 20]
}

# Instantiate GridSearchCV
gs_bag = GridSearchCV(estimator=pipe_bag,
                      param_grid=bag_params,
                      cv=5,
                      verbose=0)

# Fit GridSearchCV
gs_bag.fit(X_train, y_train)

# Accuracy score
gs_bag.score(X_train, y_train), gs_bag.score(X_test, y_test)


# In[21]:


# Predicted y test values
bag_preds = gs_bag.predict(X_test)

# Plot a confusion matrix
ConfusionMatrixDisplay.from_estimator(gs_bag, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');


# In[22]:


# Model evaluation metrics
print(classification_report(y_test, bag_preds, target_names=['MH', 'COVID']))


# In[23]:


# # Best parameters (uncomment to run)
# gs_bag.best_params_


# **Interpretation:** Model 3's testing accuracy is 91.6% on unseen data, which means that this model surpasses the baseline model in its ability to classify reddit posts. Given a training score of 99.1%, this model is quite overfit. This model returns a recall score for mentalhealth posts of .87, which is good, but not as high as Model 1.

# #### Model 4: Multinomial Naive Bayes

# In[24]:


# Define X and y
X = df['title']
y = df['is_mental']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Set up pipeline
pipe_mnb = Pipeline([
    ('tvec', TfidfVectorizer(token_pattern=r'[\w{2,}\d*]\S*\w+')),
    ('mnb', MultinomialNB())
])

# Set up parameters for pipeline
mnb_params = {
    'tvec__stop_words': [None, 'english', my_stopwords],
    'tvec__min_df': [1, 5, 10],
    'tvec__ngram_range': [(1,1), (1,2), (2,2)],
    'mnb__alpha': [1.0, 0.75, 0.5, 0.25]
}

# Instantiate GridSearchCV
gs_mnb = GridSearchCV(pipe_mnb,
                      param_grid=mnb_params,
                      cv=5,
                     verbose=0)

# Fit GridSearchCV
gs_mnb.fit(X_train, y_train)

# Accuracy score
gs_mnb.score(X_train, y_train), gs_mnb.score(X_test, y_test)


# In[25]:


# Predicted y test values
mnb_preds = gs_mnb.predict(X_test)

# Plot a confusion matrix
ConfusionMatrixDisplay.from_estimator(gs_mnb, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');


# In[26]:


# Model evaluation metrics
print(classification_report(y_test, mnb_preds, target_names=['MH', 'COVID']))


# In[27]:


# # Best parameters (uncomment to run)
# gs_mnb.best_params_


# **Interpretation:** Model 4's testing accuracy is 90.6% on unseen data, which means that this model surpasses the baseline model in its ability to classify reddit posts. Given a training score of 99.3%, this model is quite overfit. This model returns a recall score for mentalhealth posts of .89, which is high, but not as high as Model 1.

# #### Model 5: AdaBoost Classifier

# In[28]:


# Define X and y
X = df['title']
y = df['is_mental']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Set up transformer and estimator via pipeline
pipe_abc = Pipeline([
    ('tvec', TfidfVectorizer(token_pattern=r'[\w{2,}\d*]\S*\w+')),
    ('abc', AdaBoostClassifier())
])

# Set up parameters for pipeline
abc_params = {
    'tvec__stop_words': [None, 'english', my_stopwords],
    'tvec__min_df': [1, 5, 10],
    'tvec__ngram_range': [(1,1), (1,2), (2,2)],
    'abc__n_estimators': [25, 50, 100]
}

# Instantiate GridSearchCV    
gs_abc = GridSearchCV(pipe_abc,
                      param_grid=abc_params,
                      cv=5,
                     verbose=0)

# Fit GridSearchCV
gs_abc.fit(X_train,y_train)

# Accuracy score
gs_abc.score(X_train,y_train), gs_abc.score(X_test,y_test)


# In[29]:


# Predicted y test values
abc_preds = gs_abc.predict(X_test)

# Plot a confusion matrix
ConfusionMatrixDisplay.from_estimator(gs_abc, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');


# In[30]:


# Model evaluation metrics
print(classification_report(y_test, abc_preds, target_names=['MH', 'COVID']))


# In[31]:


# # Best parameters (uncomment to run)
# gs_abc.best_params_


# **Interpretation:** Model 5's testing accuracy is 88.9% on unseen data, which means that this model surpasses the baseline model in its ability to classify reddit posts. Given a training score of 91.6%, this model has a balanced bias-variance tradeoff. This model returns a recall score for mentalhealth posts of .83, which is good, but not as high as Model 1.

# #### Model 6: Random Forest Classifier

# In[32]:


# Define X and y
X = df['title']
y = df['is_mental']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Set up pipeline
pipe_rfc = Pipeline([
    ('tvec', TfidfVectorizer(token_pattern=r'[\w{2,}\d*]\S*\w+')),
    ('rfc', RandomForestClassifier())
])

# Set up parameters for pipeline
rfc_params = {
    'tvec__stop_words': [None, 'english', my_stopwords],
    'tvec__min_df': [1, 5, 10],
    'tvec__ngram_range': [(1,1), (1,2), (2,2)],
    'rfc__n_estimators': [50, 100, 200],
    'rfc__max_depth': [None, 5, 10, 20]
}

# Instantiate GridSearchCV
gs_rfc = GridSearchCV(estimator=pipe_rfc,
                      param_grid=rfc_params,
                      cv=5,
                      verbose=0)

# Fit GridSearchCV
gs_rfc.fit(X_train, y_train)

# Accuracy score for random forest
gs_rfc.score(X_train, y_train), gs_rfc.score(X_test, y_test)


# In[33]:


# Predicted y test values
rfc_preds = gs_rfc.predict(X_test)

# Plot a confusion matrix
ConfusionMatrixDisplay.from_estimator(gs_rfc, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');


# In[34]:


# Model evaluation metrics
print(classification_report(y_test, rfc_preds, target_names=['MH', 'COVID']))


# In[35]:


# # Best parameters (uncomment to run)
# gs_rfc.best_params_


# **Interpretation:** Model 6's testing accuracy is 92.5% on unseen data, which means that this model surpasses the baseline model in its ability to classify reddit posts. Given a training score of 99.8%, this model has extrememly high training accuracy but is extremely overfit. This model returns a recall score for mentalhealth posts of .86, which is good, but not as high as Model 1.

# #### Model 7: Support Vector Classifier

# In[36]:


# Define X and y
X = df['title']
y = df['is_mental']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Set up transformer and estimator via pipeline
pipe_svc = Pipeline([
    ('tvec', TfidfVectorizer(token_pattern=r'[\w{2,}\d*]\S*\w+')),
    ('svc', SVC())
])

# Set up parameters for pipeline
svc_params = {
    'tvec__stop_words': [None, 'english', my_stopwords],
    'tvec__min_df': [1, 5, 10],
    'tvec__ngram_range': [(1,1), (1,2), (2,2)],
    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

# Instantiate GridSearchCV    
gs_svc = GridSearchCV(pipe_svc,
                      param_grid=svc_params,
                      cv=5,
                     verbose=0)

# Fit GridSearchCV
gs_svc.fit(X_train,y_train)

# Accuracy score
gs_svc.score(X_train,y_train), gs_svc.score(X_test,y_test)


# In[37]:


# Predicted y test values
svc_preds = gs_svc.predict(X_test)

# Plot a confusion matrix
ConfusionMatrixDisplay.from_estimator(gs_svc, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');


# In[38]:


# Model evaluation metrics
print(classification_report(y_test, svc_preds, target_names=['MH', 'COVID']))


# In[39]:


# # Best parameters (uncomment to run)
# gs_svc.best_params_


# **Interpretation:** Model 7's testing accuracy is 91.8% on unseen data, which means that this model surpasses the baseline model in its ability to classify reddit posts. Given a training score of 99.5%, this model is quite overfit. This model returns a recall score for mentalhealth posts of .89, which is high, but not as high as Model 1.

# ### Production Model

# Because the logistic regression model had the highest performance in terms of accuracy, I did addtional analysis on posts incorrectly classified from this model.

# In[40]:


# Create Dataframe with correct and incorrect predictions
lr_df = pd.DataFrame({'title': X_test, 'true_subreddit': y_test,
                      'predicted_subreddit': gs_lr.predict(X_test)})
# Create Dataframe with only incorrect predictions
lr_df = lr_df.loc[lr_df['true_subreddit'] != lr_df['predicted_subreddit']]


# In[41]:


# View Dataframe with incorrect predictions
lr_df.head()


# In[42]:


# Define X
X = lr_df['title']

# Instantiate a CV object
cv = CountVectorizer(stop_words='english', token_pattern = r'[\w{2,}\d*]\S*\w+')

# Fit and transform the CV
X_cv = cv.fit_transform(X)

# Convert to a dataframe
cv_df = pd.DataFrame(X_cv.todense(), columns=cv.get_feature_names_out())

# Plot the top 10 words that are misclassified
cv_df.sum().sort_values(ascending=False).head(20).plot(kind='barh')
plt.title('Top 10 Misclassified Words')
plt.xlabel('Work Count')
plt.ylabel('Words');


# In[43]:


misclassified_words = cv_df.sum().sort_values(ascending=False)


# In[44]:


len(shared_words)


# In[45]:


# created list of common words between misclassified popular and my stop words
misclassified_shared_words = []
for word in misclassified_words.index:
    if word in shared_words:
        misclassified_shared_words.append(word)
print(len(misclassified_shared_words))
print(misclassified_shared_words)


# **Interpretation:** The model most frequently misclassifies posts where the 17 words shown above appear. Out of the 22 words included in the "shared_words" list, 17 of these words are present in the misclassified posts. The original logisitic regression model didn't included the "shared_word" list as part of it's stop words. How will the model perform if we include the "shared_words" list in the stop word list?

# In[46]:


# Define X and y
X = df['title']
y = df['is_mental']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Set up transformer and estimator via pipeline
pipe_lr = Pipeline([
    ('tvec', TfidfVectorizer(token_pattern=r'[\w{2,}\d*]\S*\w+')),
    ('logreg', LogisticRegression())
])

# Set up parameters for pipeline
lr_params = {
    'tvec__stop_words': [my_stopwords],
    'tvec__min_df': [1, 5, 10],
    'tvec__ngram_range': [(1,1), (1,2), (2,2)],
    'logreg__C': [0.1, .9, 1, 1.1, 10]  
}

# Instantiate GridSearchCV
gs_lr = GridSearchCV(estimator=pipe_lr,
                      param_grid=lr_params,
                      cv=5,
                      verbose=0)

# Fit GridSearchCV
gs_lr.fit(X_train, y_train)

# Accuracy score
gs_lr.score(X_train, y_train), gs_lr.score(X_test, y_test)


# In[47]:


# Predicted y test values
lr_preds = gs_lr.predict(X_test)

# Plot a confusion matrix
ConfusionMatrixDisplay.from_estimator(gs_lr, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');


# In[48]:


# Model evaluation metrics
print(classification_report(y_test, lr_preds, target_names=['MH', 'COVID']))


# **Interpretation:** This logistic regression model performs virtually the same - r/mentalhealth recall is marginally higher (it misclassified 48 posts instead of the original 50), but the model is still slightly overfit.

# #### Evaluate tfidf weights

# In[49]:


# Instantiate and fit the transformer
tvec = TfidfVectorizer(stop_words='english', min_df=1)
X_train_tvec = tvec.fit_transform(X_train)
X_test_tvec = tvec.transform(X_test)

# Instantiate and fit logistic regression
logreg = LogisticRegression(C=1)
logreg.fit(X_train_tvec, y_train)

# Create a dataframe showing the feature name and tf-idf weight of the word
my_dict = dict(zip(tvec.get_feature_names_out(), tvec.idf_))
features_df = pd.DataFrame(data=my_dict.items(), columns=['feature_name','tfidf_weight'])

# View the 10 highest weighted feature names
features_df.sort_values(by=['tfidf_weight'], ascending=False).head(10)


# **Interpretion:** Words with a weight of 8.294377 influence the classification the most.

# In[50]:


# ROC AUC Score
roc_auc_score(y_test, gs_lr.predict_proba(X_test)[:, 1])


# In[51]:


# Receiver Operating Characteristic (ROC) Curve
plot_roc_curve(gs_lr, X_test, y_test)
plt.plot([0, 1], [0, 1],
        label='baseline', linestyle='--')
plt.legend();


# **Interpretation:** The area under the ROC curve tells how much the model is capable of distingusihing between classes. More area under the curve means the distributions are better separated. This ROC AUC of 0.98 indicates that the true positive rate and the false positive rate are well separated.

# In[54]:


# Since my production model utilizing the custom stopword list has a marginally higher recall score
# for mentalhealth posts, I'm going to pickle this model 
with open('../data/production_model.pkl', 'wb') as pickle_out:
    pickle_out = pickle.dump(gs_lr, pickle_out)


# ## Conclusion
# 
# My baseline model had an accuracy score of 50%, and my logistic regression model has an accuracy of 92.1%, showing that the logisitc regression model is significantly better at predicting the correct subreddit.
# 
# 

# In[ ]:




