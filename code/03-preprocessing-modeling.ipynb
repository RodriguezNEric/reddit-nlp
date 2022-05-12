{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "# Project 3: Reddit Scraping & NLP\n",
    "\n",
    "## Part 3 - Preprocessing and Modeling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/ericrodriguez/opt/anaconda3/lib/python3.8/site-packages/xgboost/compat.py:31: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.\n",
      "  from pandas import MultiIndex, Int64Index\n"
     ]
    }
   ],
   "source": [
    "# General imports\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pickle\n",
    "\n",
    "# For Natural Language Processing\n",
    "from nltk.corpus import stopwords\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
    "\n",
    "# For Classification Modeling\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "# For Evaluation\n",
    "from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \n",
    "                             plot_roc_curve, roc_auc_score, classification_report,\n",
    "                            precision_score, recall_score, f1_score)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Read in cleaned data and reset index\n",
    "df = pd.read_csv('../data/cleaned_posts.csv', index_col=False)\n",
    "df.reset_index(inplace=True, drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>created_utc</th>\n",
       "      <th>url</th>\n",
       "      <th>full_link</th>\n",
       "      <th>id</th>\n",
       "      <th>num_comments</th>\n",
       "      <th>title</th>\n",
       "      <th>selftext</th>\n",
       "      <th>subreddit</th>\n",
       "      <th>timestamp</th>\n",
       "      <th>is_mental</th>\n",
       "      <th>title_char_length</th>\n",
       "      <th>title_word_count</th>\n",
       "      <th>sentiment_comp</th>\n",
       "      <th>sentiment_neg</th>\n",
       "      <th>sentiment_neu</th>\n",
       "      <th>sentiment_pos</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1611140689</td>\n",
       "      <td>https://www.reddit.com/r/mentalhealth/comments...</td>\n",
       "      <td>https://www.reddit.com/r/mentalhealth/comments...</td>\n",
       "      <td>l172x7</td>\n",
       "      <td>3</td>\n",
       "      <td>Has anyone else been completely unable to moni...</td>\n",
       "      <td>I had a psychotic break which seemed to come o...</td>\n",
       "      <td>mentalhealth</td>\n",
       "      <td>2021-01-20 06:04:49</td>\n",
       "      <td>1</td>\n",
       "      <td>70</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.000</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1611140753</td>\n",
       "      <td>https://www.reddit.com/r/mentalhealth/comments...</td>\n",
       "      <td>https://www.reddit.com/r/mentalhealth/comments...</td>\n",
       "      <td>l173gi</td>\n",
       "      <td>7</td>\n",
       "      <td>Is it normal to cry when you see someone you love</td>\n",
       "      <td>I know this sounds weird but i guess let me ex...</td>\n",
       "      <td>mentalhealth</td>\n",
       "      <td>2021-01-20 06:05:53</td>\n",
       "      <td>1</td>\n",
       "      <td>49</td>\n",
       "      <td>11</td>\n",
       "      <td>0.2732</td>\n",
       "      <td>0.19</td>\n",
       "      <td>0.552</td>\n",
       "      <td>0.258</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1611141523</td>\n",
       "      <td>https://www.reddit.com/r/mentalhealth/comments...</td>\n",
       "      <td>https://www.reddit.com/r/mentalhealth/comments...</td>\n",
       "      <td>l179ft</td>\n",
       "      <td>2</td>\n",
       "      <td>Somethings gotta give</td>\n",
       "      <td>Been living with my mom who has bad mental iss...</td>\n",
       "      <td>mentalhealth</td>\n",
       "      <td>2021-01-20 06:18:43</td>\n",
       "      <td>1</td>\n",
       "      <td>21</td>\n",
       "      <td>3</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.000</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1611141801</td>\n",
       "      <td>https://www.reddit.com/r/mentalhealth/comments...</td>\n",
       "      <td>https://www.reddit.com/r/mentalhealth/comments...</td>\n",
       "      <td>l17bmq</td>\n",
       "      <td>2</td>\n",
       "      <td>Train Your Brain</td>\n",
       "      <td>NaN</td>\n",
       "      <td>mentalhealth</td>\n",
       "      <td>2021-01-20 06:23:21</td>\n",
       "      <td>1</td>\n",
       "      <td>16</td>\n",
       "      <td>3</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.000</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1611143753</td>\n",
       "      <td>https://www.reddit.com/r/mentalhealth/comments...</td>\n",
       "      <td>https://www.reddit.com/r/mentalhealth/comments...</td>\n",
       "      <td>l17rke</td>\n",
       "      <td>3</td>\n",
       "      <td>Rage attacks?</td>\n",
       "      <td>Hello. \\nI usually talk to myself (and Im pret...</td>\n",
       "      <td>mentalhealth</td>\n",
       "      <td>2021-01-20 06:55:53</td>\n",
       "      <td>1</td>\n",
       "      <td>13</td>\n",
       "      <td>2</td>\n",
       "      <td>-0.7579</td>\n",
       "      <td>1.00</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   created_utc                                                url  \\\n",
       "0   1611140689  https://www.reddit.com/r/mentalhealth/comments...   \n",
       "1   1611140753  https://www.reddit.com/r/mentalhealth/comments...   \n",
       "2   1611141523  https://www.reddit.com/r/mentalhealth/comments...   \n",
       "3   1611141801  https://www.reddit.com/r/mentalhealth/comments...   \n",
       "4   1611143753  https://www.reddit.com/r/mentalhealth/comments...   \n",
       "\n",
       "                                           full_link      id  num_comments  \\\n",
       "0  https://www.reddit.com/r/mentalhealth/comments...  l172x7             3   \n",
       "1  https://www.reddit.com/r/mentalhealth/comments...  l173gi             7   \n",
       "2  https://www.reddit.com/r/mentalhealth/comments...  l179ft             2   \n",
       "3  https://www.reddit.com/r/mentalhealth/comments...  l17bmq             2   \n",
       "4  https://www.reddit.com/r/mentalhealth/comments...  l17rke             3   \n",
       "\n",
       "                                               title  \\\n",
       "0  Has anyone else been completely unable to moni...   \n",
       "1  Is it normal to cry when you see someone you love   \n",
       "2                              Somethings gotta give   \n",
       "3                                   Train Your Brain   \n",
       "4                                      Rage attacks?   \n",
       "\n",
       "                                            selftext     subreddit  \\\n",
       "0  I had a psychotic break which seemed to come o...  mentalhealth   \n",
       "1  I know this sounds weird but i guess let me ex...  mentalhealth   \n",
       "2  Been living with my mom who has bad mental iss...  mentalhealth   \n",
       "3                                                NaN  mentalhealth   \n",
       "4  Hello. \\nI usually talk to myself (and Im pret...  mentalhealth   \n",
       "\n",
       "             timestamp  is_mental  title_char_length  title_word_count  \\\n",
       "0  2021-01-20 06:04:49          1                 70                11   \n",
       "1  2021-01-20 06:05:53          1                 49                11   \n",
       "2  2021-01-20 06:18:43          1                 21                 3   \n",
       "3  2021-01-20 06:23:21          1                 16                 3   \n",
       "4  2021-01-20 06:55:53          1                 13                 2   \n",
       "\n",
       "   sentiment_comp  sentiment_neg  sentiment_neu  sentiment_pos  \n",
       "0          0.0000           0.00          1.000          0.000  \n",
       "1          0.2732           0.19          0.552          0.258  \n",
       "2          0.0000           0.00          1.000          0.000  \n",
       "3          0.0000           0.00          1.000          0.000  \n",
       "4         -0.7579           1.00          0.000          0.000  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check for missing values in title\n",
    "df.title.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "df2 = pd.read_csv('../data/shared_words.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "shared_words = list(df2['shared_word'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['help', 'like', 'health', 'know', 'i’m', 'just', 'need', 'does', 'don’t', 'think', 'people', 'time', 'going', 'getting', 'make', 'today', 'new', 'right', 'got', 'long', 'best', 'say']\n"
     ]
    }
   ],
   "source": [
    "print(shared_words)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create my_stopwords model parameter combining English stop words and subreddits shared words\n",
    "my_stopwords = list(TfidfVectorizer(stop_words = 'english').get_stop_words()) + shared_words"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Modeling\n",
    "\n",
    "#### The Null Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define X and y\n",
    "X = df['title']\n",
    "y = df['is_mental']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    0.500382\n",
       "0    0.499618\n",
       "Name: is_mental, dtype: float64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check class distribution\n",
    "y.value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see here that 50.04% of posts are from r/mentalhealth, and 49.96% of posts are from r/CoronavirusUS. The null model is showing that selecting the label r/mentalthealth would result in correctly predicting what subreddit a post came from 50.04% of the time. My goal is to build a model that classifies posts with an accuracy higher than 50.04%."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Model 1: Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.9979612640163099, 0.9205702647657841)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Define X and y\n",
    "X = df['title']\n",
    "y = df['is_mental']\n",
    "\n",
    "# Train test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n",
    "\n",
    "# Set up transformer and estimator via pipeline\n",
    "pipe_lr = Pipeline([\n",
    "    ('tvec', TfidfVectorizer(token_pattern=r'[\\w{2,}\\d*]\\S*\\w+')),\n",
    "    ('logreg', LogisticRegression())\n",
    "])\n",
    "\n",
    "# Set up parameters for pipeline\n",
    "lr_params = {\n",
    "    'tvec__stop_words': [None, 'english', my_stopwords],\n",
    "    'tvec__min_df': [1, 5, 10],\n",
    "    'tvec__ngram_range': [(1,1), (1,2), (2,2)],\n",
    "    'logreg__C': [0.1, 1, 10]  \n",
    "}\n",
    "\n",
    "# Instantiate GridSearchCV\n",
    "gs_lr = GridSearchCV(estimator=pipe_lr,\n",
    "                      param_grid=lr_params,\n",
    "                      cv=5,\n",
    "                      verbose=0)\n",
    "\n",
    "# Fit GridSearchCV\n",
    "gs_lr.fit(X_train, y_train)\n",
    "\n",
    "# Accuracy score\n",
    "gs_lr.score(X_train, y_train), gs_lr.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Predicted y test values\n",
    "lr_preds = gs_lr.predict(X_test)\n",
    "\n",
    "# Plot a confusion matrix\n",
    "ConfusionMatrixDisplay.from_estimator(gs_lr, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "          MH       0.94      0.90      0.92       485\n",
      "       COVID       0.90      0.94      0.92       497\n",
      "\n",
      "    accuracy                           0.92       982\n",
      "   macro avg       0.92      0.92      0.92       982\n",
      "weighted avg       0.92      0.92      0.92       982\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Model evaluation metrics\n",
    "print(classification_report(y_test, lr_preds, target_names=['MH', 'COVID']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'logreg__C': 10,\n",
       " 'tvec__min_df': 1,\n",
       " 'tvec__ngram_range': (1, 2),\n",
       " 'tvec__stop_words': 'english'}"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# # Best parameters (uncomment to run)\n",
    "# gs_lr.best_params_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** Model 1 is a logistic regression model that utilizes TF-IDF vectorizer and a pipline/gridsearch in order to predict the correct subreedit of a post. TF-IDF vectoriziation is used because it better differentiates rare words and gives weights to all of the words, unlike Countvectorizer which gives equal weights to all words. The most desirable parameters that maximizes accuracy include 'english\" as the stop words, and an ngram range of (1,2), meaning the model creates unigram and bigram vectors. The testing accuracy score of 92.1% on unseen data means that this model surpasses the baseline model in its ability to classify reddit posts. However, given a training score of 99.86%, this model is clearly overfit. Between precision and recall, we are more concerned with not missing r/mentalhealth posts, and this model returns a recall score of .90, which is good. Overall, this model is still largely robust in it's ability to predict the correct subreddit of a post."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Model 2: Decision Tree Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.9034998301053347, 0.895112016293279)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Define X and y\n",
    "X = df['title']\n",
    "y = df['is_mental']\n",
    "\n",
    "# Train test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n",
    "\n",
    "# Set up pipeline\n",
    "pipe_dtc = Pipeline([\n",
    "    ('tvec', TfidfVectorizer(token_pattern=r'[\\w{2,}\\d*]\\S*\\w+')),\n",
    "    ('dct', DecisionTreeClassifier(min_samples_leaf=2))\n",
    "])\n",
    "\n",
    "# Set up parameters for pipeline\n",
    "dtc_params = {\n",
    "    'tvec__stop_words': [None, 'english', my_stopwords],\n",
    "    'tvec__min_df': [1, 5, 10],\n",
    "    'tvec__ngram_range': [(1,1), (1,2), (2,2)],\n",
    "    'dct__max_depth': [None, 10, 20, 40, 100]\n",
    "}\n",
    "\n",
    "# Instantiate GridSearchCV\n",
    "gs_dtc = GridSearchCV(estimator=pipe_dtc,\n",
    "                      param_grid=dtc_params,\n",
    "                      cv=5,\n",
    "                      verbose=0)\n",
    "\n",
    "# Fit GridSearchCV\n",
    "gs_dtc.fit(X_train, y_train)\n",
    "\n",
    "# Accuracy score\n",
    "gs_dtc.score(X_train, y_train), gs_dtc.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Predicted y test values\n",
    "dtc_preds = gs_dtc.predict(X_test)\n",
    "\n",
    "# Plot a confusion matrix\n",
    "ConfusionMatrixDisplay.from_estimator(gs_dtc, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "          MH       0.98      0.81      0.88       485\n",
      "       COVID       0.84      0.98      0.90       497\n",
      "\n",
      "    accuracy                           0.90       982\n",
      "   macro avg       0.91      0.89      0.89       982\n",
      "weighted avg       0.91      0.90      0.89       982\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Model evaluation metrics\n",
    "print(classification_report(y_test, dtc_preds, target_names=['MH', 'COVID']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'dct__max_depth': 40,\n",
       " 'tvec__min_df': 10,\n",
       " 'tvec__ngram_range': (1, 2),\n",
       " 'tvec__stop_words': ['toward',\n",
       "  'hereby',\n",
       "  'eleven',\n",
       "  'against',\n",
       "  'though',\n",
       "  'who',\n",
       "  'system',\n",
       "  'all',\n",
       "  'most',\n",
       "  'empty',\n",
       "  'myself',\n",
       "  'thereupon',\n",
       "  'only',\n",
       "  'may',\n",
       "  'where',\n",
       "  'because',\n",
       "  'together',\n",
       "  'fire',\n",
       "  'etc',\n",
       "  'every',\n",
       "  'own',\n",
       "  'a',\n",
       "  'take',\n",
       "  'across',\n",
       "  'forty',\n",
       "  'less',\n",
       "  'this',\n",
       "  'than',\n",
       "  'move',\n",
       "  'serious',\n",
       "  'its',\n",
       "  'about',\n",
       "  'has',\n",
       "  'keep',\n",
       "  'among',\n",
       "  'becoming',\n",
       "  'down',\n",
       "  'beforehand',\n",
       "  'everything',\n",
       "  'elsewhere',\n",
       "  'few',\n",
       "  'with',\n",
       "  'everyone',\n",
       "  'after',\n",
       "  'whoever',\n",
       "  'or',\n",
       "  'besides',\n",
       "  'here',\n",
       "  'nowhere',\n",
       "  'were',\n",
       "  'should',\n",
       "  'namely',\n",
       "  'anyway',\n",
       "  'ltd',\n",
       "  'seeming',\n",
       "  'already',\n",
       "  'rather',\n",
       "  'almost',\n",
       "  'through',\n",
       "  'being',\n",
       "  'once',\n",
       "  'seemed',\n",
       "  'next',\n",
       "  'do',\n",
       "  'below',\n",
       "  'when',\n",
       "  'our',\n",
       "  'around',\n",
       "  'enough',\n",
       "  'if',\n",
       "  'over',\n",
       "  'often',\n",
       "  'none',\n",
       "  'always',\n",
       "  'hers',\n",
       "  'thereby',\n",
       "  'although',\n",
       "  'themselves',\n",
       "  'these',\n",
       "  'yours',\n",
       "  'fill',\n",
       "  'anywhere',\n",
       "  'you',\n",
       "  'out',\n",
       "  'sixty',\n",
       "  'whereby',\n",
       "  'why',\n",
       "  'four',\n",
       "  'but',\n",
       "  'along',\n",
       "  'within',\n",
       "  'ourselves',\n",
       "  'cant',\n",
       "  'eg',\n",
       "  'further',\n",
       "  'towards',\n",
       "  'moreover',\n",
       "  'sincere',\n",
       "  'even',\n",
       "  'put',\n",
       "  'whose',\n",
       "  'whole',\n",
       "  'find',\n",
       "  'to',\n",
       "  'thick',\n",
       "  'part',\n",
       "  'under',\n",
       "  'us',\n",
       "  'any',\n",
       "  'ie',\n",
       "  'upon',\n",
       "  'detail',\n",
       "  'we',\n",
       "  'inc',\n",
       "  'up',\n",
       "  'anyone',\n",
       "  'found',\n",
       "  'thereafter',\n",
       "  'am',\n",
       "  'whereas',\n",
       "  'what',\n",
       "  'nobody',\n",
       "  'least',\n",
       "  'will',\n",
       "  'neither',\n",
       "  'not',\n",
       "  'whither',\n",
       "  'get',\n",
       "  'behind',\n",
       "  'itself',\n",
       "  'now',\n",
       "  'indeed',\n",
       "  'someone',\n",
       "  'give',\n",
       "  'mine',\n",
       "  'of',\n",
       "  'wherein',\n",
       "  'until',\n",
       "  'sometimes',\n",
       "  'noone',\n",
       "  'that',\n",
       "  'whom',\n",
       "  'last',\n",
       "  'an',\n",
       "  'cry',\n",
       "  'by',\n",
       "  'describe',\n",
       "  'however',\n",
       "  'front',\n",
       "  'how',\n",
       "  'cannot',\n",
       "  'formerly',\n",
       "  'show',\n",
       "  'amount',\n",
       "  'either',\n",
       "  'co',\n",
       "  'and',\n",
       "  'herein',\n",
       "  'onto',\n",
       "  'therefore',\n",
       "  'could',\n",
       "  'else',\n",
       "  'on',\n",
       "  'latterly',\n",
       "  'name',\n",
       "  'more',\n",
       "  'five',\n",
       "  'please',\n",
       "  'nevertheless',\n",
       "  'thus',\n",
       "  'via',\n",
       "  'whereafter',\n",
       "  'one',\n",
       "  'fifty',\n",
       "  'other',\n",
       "  'very',\n",
       "  'meanwhile',\n",
       "  'same',\n",
       "  'while',\n",
       "  'also',\n",
       "  'which',\n",
       "  'would',\n",
       "  'hereafter',\n",
       "  'them',\n",
       "  'into',\n",
       "  'since',\n",
       "  'nothing',\n",
       "  'in',\n",
       "  'can',\n",
       "  'nor',\n",
       "  'bottom',\n",
       "  'interest',\n",
       "  'never',\n",
       "  'two',\n",
       "  'six',\n",
       "  'become',\n",
       "  'otherwise',\n",
       "  'first',\n",
       "  'hence',\n",
       "  'de',\n",
       "  'they',\n",
       "  'somehow',\n",
       "  'thin',\n",
       "  'might',\n",
       "  'those',\n",
       "  'beside',\n",
       "  'herself',\n",
       "  'yourself',\n",
       "  'such',\n",
       "  'whereupon',\n",
       "  'too',\n",
       "  're',\n",
       "  'seem',\n",
       "  'both',\n",
       "  'hasnt',\n",
       "  'whenever',\n",
       "  'each',\n",
       "  'was',\n",
       "  'my',\n",
       "  'ever',\n",
       "  'the',\n",
       "  'amoungst',\n",
       "  'yet',\n",
       "  'thence',\n",
       "  'full',\n",
       "  'becomes',\n",
       "  'she',\n",
       "  'mill',\n",
       "  'therein',\n",
       "  'yourselves',\n",
       "  'before',\n",
       "  'himself',\n",
       "  'without',\n",
       "  'so',\n",
       "  'some',\n",
       "  'anyhow',\n",
       "  'anything',\n",
       "  'fifteen',\n",
       "  'their',\n",
       "  'above',\n",
       "  'became',\n",
       "  'see',\n",
       "  'side',\n",
       "  'must',\n",
       "  'ten',\n",
       "  'former',\n",
       "  'wherever',\n",
       "  'throughout',\n",
       "  'much',\n",
       "  'due',\n",
       "  'back',\n",
       "  'whatever',\n",
       "  'three',\n",
       "  'others',\n",
       "  'had',\n",
       "  'seems',\n",
       "  'still',\n",
       "  'somewhere',\n",
       "  'for',\n",
       "  'something',\n",
       "  'made',\n",
       "  'alone',\n",
       "  'several',\n",
       "  'couldnt',\n",
       "  'again',\n",
       "  'sometime',\n",
       "  'done',\n",
       "  'him',\n",
       "  'eight',\n",
       "  'there',\n",
       "  'no',\n",
       "  'off',\n",
       "  'me',\n",
       "  'have',\n",
       "  'been',\n",
       "  'are',\n",
       "  'her',\n",
       "  'he',\n",
       "  'beyond',\n",
       "  'many',\n",
       "  'your',\n",
       "  'perhaps',\n",
       "  'then',\n",
       "  'mostly',\n",
       "  'at',\n",
       "  'con',\n",
       "  'except',\n",
       "  'it',\n",
       "  'call',\n",
       "  'afterwards',\n",
       "  'everywhere',\n",
       "  'top',\n",
       "  'whence',\n",
       "  'during',\n",
       "  'ours',\n",
       "  'well',\n",
       "  'another',\n",
       "  'i',\n",
       "  'hundred',\n",
       "  'amongst',\n",
       "  'nine',\n",
       "  'per',\n",
       "  'be',\n",
       "  'is',\n",
       "  'between',\n",
       "  'go',\n",
       "  'un',\n",
       "  'from',\n",
       "  'twelve',\n",
       "  'latter',\n",
       "  'bill',\n",
       "  'twenty',\n",
       "  'his',\n",
       "  'third',\n",
       "  'hereupon',\n",
       "  'as',\n",
       "  'thru',\n",
       "  'whether',\n",
       "  'help',\n",
       "  'like',\n",
       "  'health',\n",
       "  'know',\n",
       "  'i’m',\n",
       "  'just',\n",
       "  'need',\n",
       "  'does',\n",
       "  'don’t',\n",
       "  'think',\n",
       "  'people',\n",
       "  'time',\n",
       "  'going',\n",
       "  'getting',\n",
       "  'make',\n",
       "  'today',\n",
       "  'new',\n",
       "  'right',\n",
       "  'got',\n",
       "  'long',\n",
       "  'best',\n",
       "  'say']}"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# # Best parameters (uncomment to run)\n",
    "# gs_dtc.best_params_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** Model 2's testing accuracy of 89.5% on unseen data means that this model surpasses the baseline model in its ability to classify reddit posts. Given a training score of 90.3%, this model shows a balanced bias-variance tradeoff. Between precision and recall, we are more concerned with not missing r/mentalhealth posts, and this model returns a recall score of .81, which is good, but not as high as Model 1."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Model 3: Bagging Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.9908256880733946, 0.9164969450101833)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Define X and y\n",
    "X = df['title']\n",
    "y = df['is_mental']\n",
    "\n",
    "# Train test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n",
    "\n",
    "# Set up pipeline\n",
    "pipe_bag = Pipeline([\n",
    "    ('tvec', TfidfVectorizer(token_pattern=r'[\\w{2,}\\d*]\\S*\\w+')),\n",
    "    ('bag', BaggingClassifier())\n",
    "])\n",
    "\n",
    "# Set up parameters for pipeline\n",
    "bag_params = {\n",
    "    'tvec__stop_words': [None, 'english', my_stopwords],\n",
    "    'tvec__min_df': [1, 5, 10],\n",
    "    'tvec__ngram_range': [(1,1), (1,2), (2,2)],\n",
    "    'bag__n_estimators': [10, 15, 20]\n",
    "}\n",
    "\n",
    "# Instantiate GridSearchCV\n",
    "gs_bag = GridSearchCV(estimator=pipe_bag,\n",
    "                      param_grid=bag_params,\n",
    "                      cv=5,\n",
    "                      verbose=0)\n",
    "\n",
    "# Fit GridSearchCV\n",
    "gs_bag.fit(X_train, y_train)\n",
    "\n",
    "# Accuracy score\n",
    "gs_bag.score(X_train, y_train), gs_bag.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Predicted y test values\n",
    "bag_preds = gs_bag.predict(X_test)\n",
    "\n",
    "# Plot a confusion matrix\n",
    "ConfusionMatrixDisplay.from_estimator(gs_bag, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "          MH       0.96      0.87      0.91       485\n",
      "       COVID       0.88      0.97      0.92       497\n",
      "\n",
      "    accuracy                           0.92       982\n",
      "   macro avg       0.92      0.92      0.92       982\n",
      "weighted avg       0.92      0.92      0.92       982\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Model evaluation metrics\n",
    "print(classification_report(y_test, bag_preds, target_names=['MH', 'COVID']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'bag__n_estimators': 15,\n",
       " 'tvec__min_df': 1,\n",
       " 'tvec__ngram_range': (1, 1),\n",
       " 'tvec__stop_words': ['toward',\n",
       "  'hereby',\n",
       "  'eleven',\n",
       "  'against',\n",
       "  'though',\n",
       "  'who',\n",
       "  'system',\n",
       "  'all',\n",
       "  'most',\n",
       "  'empty',\n",
       "  'myself',\n",
       "  'thereupon',\n",
       "  'only',\n",
       "  'may',\n",
       "  'where',\n",
       "  'because',\n",
       "  'together',\n",
       "  'fire',\n",
       "  'etc',\n",
       "  'every',\n",
       "  'own',\n",
       "  'a',\n",
       "  'take',\n",
       "  'across',\n",
       "  'forty',\n",
       "  'less',\n",
       "  'this',\n",
       "  'than',\n",
       "  'move',\n",
       "  'serious',\n",
       "  'its',\n",
       "  'about',\n",
       "  'has',\n",
       "  'keep',\n",
       "  'among',\n",
       "  'becoming',\n",
       "  'down',\n",
       "  'beforehand',\n",
       "  'everything',\n",
       "  'elsewhere',\n",
       "  'few',\n",
       "  'with',\n",
       "  'everyone',\n",
       "  'after',\n",
       "  'whoever',\n",
       "  'or',\n",
       "  'besides',\n",
       "  'here',\n",
       "  'nowhere',\n",
       "  'were',\n",
       "  'should',\n",
       "  'namely',\n",
       "  'anyway',\n",
       "  'ltd',\n",
       "  'seeming',\n",
       "  'already',\n",
       "  'rather',\n",
       "  'almost',\n",
       "  'through',\n",
       "  'being',\n",
       "  'once',\n",
       "  'seemed',\n",
       "  'next',\n",
       "  'do',\n",
       "  'below',\n",
       "  'when',\n",
       "  'our',\n",
       "  'around',\n",
       "  'enough',\n",
       "  'if',\n",
       "  'over',\n",
       "  'often',\n",
       "  'none',\n",
       "  'always',\n",
       "  'hers',\n",
       "  'thereby',\n",
       "  'although',\n",
       "  'themselves',\n",
       "  'these',\n",
       "  'yours',\n",
       "  'fill',\n",
       "  'anywhere',\n",
       "  'you',\n",
       "  'out',\n",
       "  'sixty',\n",
       "  'whereby',\n",
       "  'why',\n",
       "  'four',\n",
       "  'but',\n",
       "  'along',\n",
       "  'within',\n",
       "  'ourselves',\n",
       "  'cant',\n",
       "  'eg',\n",
       "  'further',\n",
       "  'towards',\n",
       "  'moreover',\n",
       "  'sincere',\n",
       "  'even',\n",
       "  'put',\n",
       "  'whose',\n",
       "  'whole',\n",
       "  'find',\n",
       "  'to',\n",
       "  'thick',\n",
       "  'part',\n",
       "  'under',\n",
       "  'us',\n",
       "  'any',\n",
       "  'ie',\n",
       "  'upon',\n",
       "  'detail',\n",
       "  'we',\n",
       "  'inc',\n",
       "  'up',\n",
       "  'anyone',\n",
       "  'found',\n",
       "  'thereafter',\n",
       "  'am',\n",
       "  'whereas',\n",
       "  'what',\n",
       "  'nobody',\n",
       "  'least',\n",
       "  'will',\n",
       "  'neither',\n",
       "  'not',\n",
       "  'whither',\n",
       "  'get',\n",
       "  'behind',\n",
       "  'itself',\n",
       "  'now',\n",
       "  'indeed',\n",
       "  'someone',\n",
       "  'give',\n",
       "  'mine',\n",
       "  'of',\n",
       "  'wherein',\n",
       "  'until',\n",
       "  'sometimes',\n",
       "  'noone',\n",
       "  'that',\n",
       "  'whom',\n",
       "  'last',\n",
       "  'an',\n",
       "  'cry',\n",
       "  'by',\n",
       "  'describe',\n",
       "  'however',\n",
       "  'front',\n",
       "  'how',\n",
       "  'cannot',\n",
       "  'formerly',\n",
       "  'show',\n",
       "  'amount',\n",
       "  'either',\n",
       "  'co',\n",
       "  'and',\n",
       "  'herein',\n",
       "  'onto',\n",
       "  'therefore',\n",
       "  'could',\n",
       "  'else',\n",
       "  'on',\n",
       "  'latterly',\n",
       "  'name',\n",
       "  'more',\n",
       "  'five',\n",
       "  'please',\n",
       "  'nevertheless',\n",
       "  'thus',\n",
       "  'via',\n",
       "  'whereafter',\n",
       "  'one',\n",
       "  'fifty',\n",
       "  'other',\n",
       "  'very',\n",
       "  'meanwhile',\n",
       "  'same',\n",
       "  'while',\n",
       "  'also',\n",
       "  'which',\n",
       "  'would',\n",
       "  'hereafter',\n",
       "  'them',\n",
       "  'into',\n",
       "  'since',\n",
       "  'nothing',\n",
       "  'in',\n",
       "  'can',\n",
       "  'nor',\n",
       "  'bottom',\n",
       "  'interest',\n",
       "  'never',\n",
       "  'two',\n",
       "  'six',\n",
       "  'become',\n",
       "  'otherwise',\n",
       "  'first',\n",
       "  'hence',\n",
       "  'de',\n",
       "  'they',\n",
       "  'somehow',\n",
       "  'thin',\n",
       "  'might',\n",
       "  'those',\n",
       "  'beside',\n",
       "  'herself',\n",
       "  'yourself',\n",
       "  'such',\n",
       "  'whereupon',\n",
       "  'too',\n",
       "  're',\n",
       "  'seem',\n",
       "  'both',\n",
       "  'hasnt',\n",
       "  'whenever',\n",
       "  'each',\n",
       "  'was',\n",
       "  'my',\n",
       "  'ever',\n",
       "  'the',\n",
       "  'amoungst',\n",
       "  'yet',\n",
       "  'thence',\n",
       "  'full',\n",
       "  'becomes',\n",
       "  'she',\n",
       "  'mill',\n",
       "  'therein',\n",
       "  'yourselves',\n",
       "  'before',\n",
       "  'himself',\n",
       "  'without',\n",
       "  'so',\n",
       "  'some',\n",
       "  'anyhow',\n",
       "  'anything',\n",
       "  'fifteen',\n",
       "  'their',\n",
       "  'above',\n",
       "  'became',\n",
       "  'see',\n",
       "  'side',\n",
       "  'must',\n",
       "  'ten',\n",
       "  'former',\n",
       "  'wherever',\n",
       "  'throughout',\n",
       "  'much',\n",
       "  'due',\n",
       "  'back',\n",
       "  'whatever',\n",
       "  'three',\n",
       "  'others',\n",
       "  'had',\n",
       "  'seems',\n",
       "  'still',\n",
       "  'somewhere',\n",
       "  'for',\n",
       "  'something',\n",
       "  'made',\n",
       "  'alone',\n",
       "  'several',\n",
       "  'couldnt',\n",
       "  'again',\n",
       "  'sometime',\n",
       "  'done',\n",
       "  'him',\n",
       "  'eight',\n",
       "  'there',\n",
       "  'no',\n",
       "  'off',\n",
       "  'me',\n",
       "  'have',\n",
       "  'been',\n",
       "  'are',\n",
       "  'her',\n",
       "  'he',\n",
       "  'beyond',\n",
       "  'many',\n",
       "  'your',\n",
       "  'perhaps',\n",
       "  'then',\n",
       "  'mostly',\n",
       "  'at',\n",
       "  'con',\n",
       "  'except',\n",
       "  'it',\n",
       "  'call',\n",
       "  'afterwards',\n",
       "  'everywhere',\n",
       "  'top',\n",
       "  'whence',\n",
       "  'during',\n",
       "  'ours',\n",
       "  'well',\n",
       "  'another',\n",
       "  'i',\n",
       "  'hundred',\n",
       "  'amongst',\n",
       "  'nine',\n",
       "  'per',\n",
       "  'be',\n",
       "  'is',\n",
       "  'between',\n",
       "  'go',\n",
       "  'un',\n",
       "  'from',\n",
       "  'twelve',\n",
       "  'latter',\n",
       "  'bill',\n",
       "  'twenty',\n",
       "  'his',\n",
       "  'third',\n",
       "  'hereupon',\n",
       "  'as',\n",
       "  'thru',\n",
       "  'whether',\n",
       "  'help',\n",
       "  'like',\n",
       "  'health',\n",
       "  'know',\n",
       "  'i’m',\n",
       "  'just',\n",
       "  'need',\n",
       "  'does',\n",
       "  'don’t',\n",
       "  'think',\n",
       "  'people',\n",
       "  'time',\n",
       "  'going',\n",
       "  'getting',\n",
       "  'make',\n",
       "  'today',\n",
       "  'new',\n",
       "  'right',\n",
       "  'got',\n",
       "  'long',\n",
       "  'best',\n",
       "  'say']}"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# # Best parameters (uncomment to run)\n",
    "# gs_bag.best_params_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** Model 3's testing accuracy is 91.6% on unseen data, which means that this model surpasses the baseline model in its ability to classify reddit posts. Given a training score of 99.1%, this model is quite overfit. This model returns a recall score for mentalhealth posts of .87, which is good, but not as high as Model 1."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Model 4: Multinomial Naive Bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.9932042133876996, 0.9063136456211812)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Define X and y\n",
    "X = df['title']\n",
    "y = df['is_mental']\n",
    "\n",
    "# Train test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n",
    "\n",
    "# Set up pipeline\n",
    "pipe_mnb = Pipeline([\n",
    "    ('tvec', TfidfVectorizer(token_pattern=r'[\\w{2,}\\d*]\\S*\\w+')),\n",
    "    ('mnb', MultinomialNB())\n",
    "])\n",
    "\n",
    "# Set up parameters for pipeline\n",
    "mnb_params = {\n",
    "    'tvec__stop_words': [None, 'english', my_stopwords],\n",
    "    'tvec__min_df': [1, 5, 10],\n",
    "    'tvec__ngram_range': [(1,1), (1,2), (2,2)],\n",
    "    'mnb__alpha': [1.0, 0.75, 0.5, 0.25]\n",
    "}\n",
    "\n",
    "# Instantiate GridSearchCV\n",
    "gs_mnb = GridSearchCV(pipe_mnb,\n",
    "                      param_grid=mnb_params,\n",
    "                      cv=5,\n",
    "                     verbose=0)\n",
    "\n",
    "# Fit GridSearchCV\n",
    "gs_mnb.fit(X_train, y_train)\n",
    "\n",
    "# Accuracy score\n",
    "gs_mnb.score(X_train, y_train), gs_mnb.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Predicted y test values\n",
    "mnb_preds = gs_mnb.predict(X_test)\n",
    "\n",
    "# Plot a confusion matrix\n",
    "ConfusionMatrixDisplay.from_estimator(gs_mnb, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "          MH       0.92      0.89      0.90       485\n",
      "       COVID       0.89      0.92      0.91       497\n",
      "\n",
      "    accuracy                           0.91       982\n",
      "   macro avg       0.91      0.91      0.91       982\n",
      "weighted avg       0.91      0.91      0.91       982\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Model evaluation metrics\n",
    "print(classification_report(y_test, mnb_preds, target_names=['MH', 'COVID']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'mnb__alpha': 0.25,\n",
       " 'tvec__min_df': 1,\n",
       " 'tvec__ngram_range': (1, 2),\n",
       " 'tvec__stop_words': None}"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# # Best parameters (uncomment to run)\n",
    "# gs_mnb.best_params_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** Model 4's testing accuracy is 90.6% on unseen data, which means that this model surpasses the baseline model in its ability to classify reddit posts. Given a training score of 99.3%, this model is quite overfit. This model returns a recall score for mentalhealth posts of .89, which is high, but not as high as Model 1."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Model 5: AdaBoost Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.9164118246687054, 0.8890020366598778)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Define X and y\n",
    "X = df['title']\n",
    "y = df['is_mental']\n",
    "\n",
    "# Train test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n",
    "\n",
    "# Set up transformer and estimator via pipeline\n",
    "pipe_abc = Pipeline([\n",
    "    ('tvec', TfidfVectorizer(token_pattern=r'[\\w{2,}\\d*]\\S*\\w+')),\n",
    "    ('abc', AdaBoostClassifier())\n",
    "])\n",
    "\n",
    "# Set up parameters for pipeline\n",
    "abc_params = {\n",
    "    'tvec__stop_words': [None, 'english', my_stopwords],\n",
    "    'tvec__min_df': [1, 5, 10],\n",
    "    'tvec__ngram_range': [(1,1), (1,2), (2,2)],\n",
    "    'abc__n_estimators': [25, 50, 100]\n",
    "}\n",
    "\n",
    "# Instantiate GridSearchCV    \n",
    "gs_abc = GridSearchCV(pipe_abc,\n",
    "                      param_grid=abc_params,\n",
    "                      cv=5,\n",
    "                     verbose=0)\n",
    "\n",
    "# Fit GridSearchCV\n",
    "gs_abc.fit(X_train,y_train)\n",
    "\n",
    "# Accuracy score\n",
    "gs_abc.score(X_train,y_train), gs_abc.score(X_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Predicted y test values\n",
    "abc_preds = gs_abc.predict(X_test)\n",
    "\n",
    "# Plot a confusion matrix\n",
    "ConfusionMatrixDisplay.from_estimator(gs_abc, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "          MH       0.94      0.83      0.88       485\n",
      "       COVID       0.85      0.94      0.90       497\n",
      "\n",
      "    accuracy                           0.89       982\n",
      "   macro avg       0.89      0.89      0.89       982\n",
      "weighted avg       0.89      0.89      0.89       982\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Model evaluation metrics\n",
    "print(classification_report(y_test, abc_preds, target_names=['MH', 'COVID']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'abc__n_estimators': 100,\n",
       " 'tvec__min_df': 10,\n",
       " 'tvec__ngram_range': (1, 1),\n",
       " 'tvec__stop_words': 'english'}"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# # Best parameters (uncomment to run)\n",
    "# gs_abc.best_params_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** Model 5's testing accuracy is 88.9% on unseen data, which means that this model surpasses the baseline model in its ability to classify reddit posts. Given a training score of 91.6%, this model has a balanced bias-variance tradeoff. This model returns a recall score for mentalhealth posts of .83, which is good, but not as high as Model 1."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Model 6: Random Forest Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.9979612640163099, 0.924643584521385)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Define X and y\n",
    "X = df['title']\n",
    "y = df['is_mental']\n",
    "\n",
    "# Train test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n",
    "\n",
    "# Set up pipeline\n",
    "pipe_rfc = Pipeline([\n",
    "    ('tvec', TfidfVectorizer(token_pattern=r'[\\w{2,}\\d*]\\S*\\w+')),\n",
    "    ('rfc', RandomForestClassifier())\n",
    "])\n",
    "\n",
    "# Set up parameters for pipeline\n",
    "rfc_params = {\n",
    "    'tvec__stop_words': [None, 'english', my_stopwords],\n",
    "    'tvec__min_df': [1, 5, 10],\n",
    "    'tvec__ngram_range': [(1,1), (1,2), (2,2)],\n",
    "    'rfc__n_estimators': [50, 100, 200],\n",
    "    'rfc__max_depth': [None, 5, 10, 20]\n",
    "}\n",
    "\n",
    "# Instantiate GridSearchCV\n",
    "gs_rfc = GridSearchCV(estimator=pipe_rfc,\n",
    "                      param_grid=rfc_params,\n",
    "                      cv=5,\n",
    "                      verbose=0)\n",
    "\n",
    "# Fit GridSearchCV\n",
    "gs_rfc.fit(X_train, y_train)\n",
    "\n",
    "# Accuracy score for random forest\n",
    "gs_rfc.score(X_train, y_train), gs_rfc.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAVIAAAEGCAYAAAA3yh0OAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAcL0lEQVR4nO3de5xVdb3/8dd7ZpCLCIIgokh4QRHwllfUlMp7lll6xMwflYWWnvJkpViJ2Q+jftWvi2lideRXCtpRE0sJQ/nhLRAVUVAOGCYgIeCNQMGZ+Zw/9prc4syePay9Z6898376WI/Z+7vX5bMZ+fC9rO93KSIwM7NtV1PpAMzMqp0TqZlZSk6kZmYpOZGamaXkRGpmllJdpQNob/1618WQAV0qHYa1wdKlu1Q6BGujN3hhXUT039bjjz+xR6xf31DUvgue2PLniDh5W69VCp0ukQ4Z0IV51+1V6TCsDU476WuVDsHa6N7Gz/49zfHr1zcy++HBRe27Y/dl/dJcqxQ6XSI1syoQoEZVOoqiOZGaWTaFE6mZ2TYTrpGamaUToPpKB1E8J1Izy54AVdEyIE6kZpZJaqx0BMVzIjWzbGqsniqpE6mZZY+b9mZmJeCmvZnZtlOA6qunSupEamaZ5Ka9mVlabtqbmaUQvv3JzCy9KnowpxOpmWWPp4iamaXnwSYzs7TcR2pmlkLgRGpmloYAeWFnM7OUXCM1M0shgOIeIpoJTqRmlkl+1IiZWRqRbFXCidTMssk1UjOzlDzYZGaWgpv2ZmZpCRpqKh1E0ZxIzSx7vIyemVkJeLDJzCwl95GamaUQuEZqZpZagxOpmVkKAq/+ZGaWQkC4aW9mllIV1Uir545XM+tcGovciiSpVtKTkv6YvO8r6T5JS5OfffL2HS9pmaQlkk5q7dxOpGaWPUGuRlrMVryvAM/mvb8cmBURQ4FZyXskDQfGACOAk4HrJNUWOrETqZllUDJFtJitmLNJg4CPAL/KKz4dmJK8ngJ8PK98WkRsjojlwDLg8ELndyI1s+xpuo+0mK04PwG+wbs7AwZExGqA5OfOSfluwIq8/VYmZS1yIjWzbIoiN+gnaX7eNi7/NJJOA16OiMeLvHJz2bngPCuP2ptZJrXh9qd1EXFogc+PBj4m6VSgG9BL0u+ANZIGRsRqSQOBl5P9VwK75x0/CHipUACukZpZNpVosCkixkfEoIgYQm4Q6f6I+DQwHRib7DYWuCt5PR0YI6mrpD2AocC8QtdwjdTMsqd95tpPAm6TdD7wInAWQEQsknQbsBioBy6KiILPNHUiNbMMKs/CzhExG5idvF4PfLiF/SYCE4s9rxOpmWVORG6rFk6kZpZNVTRF1InUzLLJi5aYmaUQEK6Rmpml4aeIWpk0NIjRXz2FXftu4tYJs/nDQ4OZdMsBLFnZm/t/dC8HD30FgC1v13DJL45gwbK+SDBp3Hw+sP+aCkdvPXq/xbjr72bQ8JchxA0XfpSlc3fnpC/O48QLH6OxvoYnZ+zNLd88odKhZoJrpCUgKYDfRcR5yfs6YDUwNyJOk/QZ4NCIuDjvmNnA1yJifgVCLrvr7x7GvoNeZ8OmLgDs977X+O0Vc7jkF0e8a78pM/cG4JFr/8Ta17py5lUf4oEf30tN9fwD3yGN/eEMnpq5Fz/51FnUdmmga4+3GX7scg45bQmXHXYB9Vvq6NV/Y6XDzIagTUvkVVqW/2ptBEZK6p68PwFYVcF4KmrVuh7MfGxXzjtx2b/K9t39DYYOeuM9+y55sTfHHfgPAPrvuJne22/hyWU7tVus9l7dd9jMsGNe5IGbDgag4e1aNr3ejRPGPc70Hx5N/ZZcneaNtdtXMsxsKf0yemWT5UQKcC+5pa8AzgGmVjCWihp/4yFc/dkni6pVjtzjVe6ZO4j6BvHCP7ZnwfM7sXJtj/IHaS3aeY9XeWNdDy6cPJ3vPTqZL1x3N117bGGXvdcz7OgX+e6cX3HlzJvY85BOW1d4j2hUUVsWZD2RTiM357UbcAAwd6vPz5a0oGkDml24QNK4ppVh1r5ecKZXJs2Ytxv9e7/FQXu/UtT+nz7heXbdaROj/+MUxv/qUI4Ytpa62iq6u7kDqq1rZI+DVnPfjYcwftQ4Nm/qwse+9jC1dY1s3+ctvn3s+dx8xQl85Xe3U1UPdC+XYmujGamRZraPFCAiFkoaQq42ek8zu9zaTB9pc+eZDEwGOHSf7lX3f+ncZ/tz77xBzHx8NzZvqWXDpi6M+9HRTL704Wb3r6sNvveFd1YMO/HrJ7HXrhvaK1xrxvpVvXhlVS+ef2wQAHPv3I/TL32YV1b1Yt4fhgHi+fm7EY1ih36b2LDOTfzwqH1JTQd+CIwGOmVH34SxC5gwdgEADz49gGvv2K/FJAqw6a1aAti+WwMPPLkLtbWNDBv8evsEa816fU1P1q/sxcCh61i9tB8jRy9n5XP9eflvfRgxejnPPjiEXfZeT912DWxY524YIDO1zWJUQyL9DfB6RDwtaXSFY8mUux/dnctuOJR1r3fj367+IPvv8Sp3XH0/a1/vxicnfJgaBQN32sQNX32k0qEacNNXT+Hi/7yTuu0aWPNCH24Y9zHe2rgdF94wnR/Mv576LbVc//nTaX5d4c7Fc+1LLCJWAj+tdBxZ8YH91/zrntCPjlrBR0eteM8+7xuwkfm/nN7eoVkr/r5wF755zBfeU/6Lz51RgWiqQEYGkoqR2UQaET2bKZvNO0tg3QTctNXno8semJm1A/mGfDOz1JxIzcxSCI/am5ml5qa9mVka0aZn1lecE6mZZZJvfzIzSyFw097MLB0PNpmZlYBrpGZmafiGfDOz9Dxqb2aWghctMTNLx6P2ZmapyaP2ZmaphGukZmbpOZGamaXjGqmZWUrRWOkIiudEambZE7hpb2aWRiAaGz1qb2aWjmukZmYpBISniJqZpVNNo/bV0wlhZp1LFLm1QlI3SfMkPSVpkaTvJOV9Jd0naWnys0/eMeMlLZO0RNJJrV3DidTMMqdpsKmYrQibgQ9FxIHAQcDJko4ELgdmRcRQYFbyHknDgTHACOBk4DpJtYUu4ERqZtmT9JEWs7V6qpx/Jm+7JFsApwNTkvIpwMeT16cD0yJic0QsB5YBhxe6hhOpmWVTqLgN+kman7eN2/pUkmolLQBeBu6LiLnAgIhYDZD83DnZfTdgRd7hK5OyFrU42CTp5xTogYiILxc6sZlZGm0YbFoXEYcWPlc0AAdJ2hG4U9LIArs3d+GCvbGFRu3nFzrQzKx8yvOokYh4TdJscn2fayQNjIjVkgaSq61Crga6e95hg4CXCp23xUQaEVPy30vaPiI2bkvwZmZtUsIV8iX1B95Okmh34Hjg+8B0YCwwKfl5V3LIdOAWST8GdgWGAvMKXaPV+0gljQJ+DfQEBks6ELggIr60Td/KzKwVQUkfxzwQmJKMvNcAt0XEHyU9Ctwm6XzgReAsgIhYJOk2YDFQD1yUdA20qJgb8n8CnEQuSxMRT0k6dhu/kJlZUUrVtI+IhcDBzZSvBz7cwjETgYnFXqOomU0RsUJ615cqmJ3NzFLpgCvkr5B0FBCStgO+DDxb3rDMrHOrrufaF9MJcSFwEbn7qFaRmxlwURljMjMjQkVtWdBqjTQi1gHntkMsZmZAbsQ+GrKRJIvRao1U0p6S7pa0VtLLku6StGd7BGdmnVc11UiLadrfAtxG7haCXYHfA1PLGZSZWUdLpIqI30ZEfbL9jqIWrzIz21bFJdGsJNJCc+37Ji8fkHQ5MI1cAj0b+FM7xGZmnVhWkmQxCg02PU4ucTZ9mwvyPgvgu+UKysw6uY7yFNGI2KM9AzEzaxLQ8Z4imiw5NRzo1lQWEf+vXEGZWScXEI2VDqJ4xSxaMgEYTS6R3gOcAjwEOJGaWZlkZyCpGMXUnc8kN7H/HxHxWeBAoGtZozKzTq9DjNrneTMiGiXVS+pFbvFT35BvZmUTdJxR+ybzk+X5byQ3kv9PWlnk1MwsrQ6VSPMWcP6lpBlAr2R9PzOz8gh1jFF7Se8v9FlEPFGekMzMgCIetZwVhWqkPyrwWQAfKnEs7eLJZTvR+zQvZlVNVmy6utIhWBv16db6Pq3pEE37iPhgewZiZtYkOuAK+WZm7a5UTxFtD06kZpZBHWSwycyskqqpaV/MCvmS9GlJVybvB0s6vPyhmVln1dRHWi0zm4qpO18HjALOSd5vAH5RtojMzIBoVFFbFhTTtD8iIt4v6UmAiHg1eSyzmVnZZKW2WYxiEunbkmpJHi8iqT9QRQtcmVn1yU6zvRjFJNKfAXcCO0uaSG41qG+VNSoz69QiOtjCzhFxs6THyS2lJ+DjEfFs2SMzs06tQ9VIJQ0GNgF355dFxIvlDMzMOrcOlUjJPTG06SF43YA9gCXAiDLGZWadWgfrI42I/fPfJ6tCXdDC7mZm6QWZubWpGG2e2RQRT0g6rBzBmJlBB1whX9JX897WAO8H1pYtIjMzoKGD1Uh3yHtdT67P9PbyhGNmRq5p31FqpMmN+D0j4uvtFI+ZGdFRBpsk1UVEfaFHjpiZlUs1JdJCUweanhS6QNJ0SedJ+kTT1h7BmVnnVarVnyTtLukBSc9KWiTpK0l5X0n3SVqa/OyTd8x4ScskLZF0UmvXKKaPtC+wntwzmpruJw3gjiKONTNru4DGhpJNEa0HLk3uONoBeFzSfcBngFkRMUnS5cDlwGWShgNjyN0rvyvwF0n7RERDSxcolEh3Tkbsn+GdBNqkih4CYGbVppR9pBGxGlidvN4g6VlgN+B0YHSy2xRgNnBZUj4tIjYDyyUtAw4HHm3pGoUSaS3Qk3cn0H/F1pYvYmbWVm1IpP0kzc97PzkiJje3o6QhwMHAXGBAkmSJiNWSdk522w34a95hK5OyFhVKpKsjws/BNbOKaCw+ka6LiENb20lST3K3bl4SEW9ILZ6/zZXHQp0Q1TNkZmYdS4kfNSKpC7kkenNENI3vrJE0MPl8IPByUr4S2D3v8EHAS4XOXyiRfrioCM3MSqxpimiJRu0F/Bp4NiJ+nPfRdGBs8noscFde+RhJXSXtAQzlnbuYmtVi0z4iXmk1QjOzshCNDSVrFB8NnAc8LWlBUnYFMAm4TdL5wIvAWQARsUjSbcBiciP+FxUasQc/jtnMsija1Eda+FQRD9FyV2WzLe+ImAhMLPYaTqRmljkdbvUnM7NKcCI1M0vJidTMLBWVrI+0PTiRmlnmRFDKUfuycyI1s0xy097MLIWgdLc/tQcnUjPLnsg176uFE6mZZZKb9mZmKQTqcE8RNTNrd66RmpmlUcK59u3BidTMMikaKx1B8ZxIzSxzvGiJmVlqniJqZpZKBB61NzNLyzfkm5ml5D5SM7OUGl0jNTPbduG59mZm6XmwycwsJddIrV19/qKnOed/PUeEeG5xXy794rFs3uxfbRY0NIjTRp/BgF03ctOtf2bRwp244qvHsPmtWmrrgok/eoiDDlnLli01jL/kAyxc0J8aBVdNeoRRH1hd6fArptrWI60p58kl7SJpmqTnJS2WdI+kfSSNkHS/pP+WtFTSt5UzWtKjW52jTtIaSQMl3STpzKR8tqQlkhZKek7StZJ2LOf3yaJdBm7kcxc8w0eOO4PjjzyT2ppGPvbJv1U6LEv85vqR7L3va/96f82EI7jksieY8dAdXHrFfK658ggApk4ZBsB9j/wXN//hT3z3W0fSWEVTJMshityyoGyJVJKAO4HZEbFXRAwHrgAGANOBSRGxD3AgcBTwJWAOMEjSkLxTHQ88ExHN/fN8bkQcABwAbAbuKtf3ybK6uqBb93pqaxvp3qOeNf/oUemQDFi9antmzRzMmPOe+1eZFGzY0AWADW9sx4CBmwBYuqQPRx+3CoB+/d+iV+8tLHyyf/sHnRWRG7UvZsuCctZIPwi8HRG/bCqIiAXAPsDDETEzKdsEXAxcHhGNwO+Bs/POMwaYWuhCEbEF+AYwWNKBpfwSWfeP1dtzw88PYO6iqTyx9GY2vLEdc+4fVOmwDLhq/CiuuHouNTXv/G2f8L1HuebKIzlixKf4398+ksuunAfAfiPXM/OeIdTXixdf2IFnFvTjpZU9KxV6xQUqesuCcibSkcDjzZSP2Lo8Ip4HekrqRS5pjgGQ1BU4Fbi9tYtFRAPwFDBs688kjZM0X9L8iI1t/R6Z1nvHzZx46guM2n8Mh+xzLt171POJs5dWOqxO7y8zBtOv/5sccNC6d5X/9tfDuXLio8xddAtXXvMoX//3YwE4+9NLGLjrRk4bfQbfGT+KQ45YQ11d527bN0RxWxZUYkRCtNy1ERHxmKSekvYF9gP+GhGvtuHczZ10MjAZoLZmUEb+6EvjmNGrWPH3HXhlfXcA7r17CIccsYY7bh1a4cg6t/lzB3Dfve/jgZmD2by5lg0btuMr4z7IX2a8j+98/xEATvv437jsy7lEWlcXTPjeO8MDZ5z4MYbs9XpFYs+C3GBTpaMoXjlrpIuAQ1ooPzS/QNKewD8jYkNSNI1crbTVZn3eOWqB/YFntzXgavTSyp4cfNjLdOteDwTHHPcSy5bsWOmwOr3LJzzGvMW38MjTU7n217M46thV/HTyAwzYZSN/fWggAA/P2ZUhe+aS5Zubatm0MVevmfPAbtTWBvsMe61S4WdCNQ02lbNGej9wjaQvRMSNAJIOA5YCV0g6PiL+Iqk78DPgB3nHTiU3cNQbOL+1C0nqAkwEVkTEwhJ/j0x7cv7O3HPXnsx48A7q62tYtHAnbv7P/SodlrVg0k/ncNXlR9FQX0PXbg1M+umDAKxb253zPnkqNTXBgIEb+ckND1Q40sqrphqpoox3vUraFfgJuZrpW8ALwCVAN+DnwECgFvgtcHXkBSPpKeDZiBiTV3YT8MeI+C9Js5PjNwNdgb8A34yI1wrFVFszKHpsd1Epvp61kxWv31jpEKyN+nRb/nhEHNr6ns0bqL1irK4pat/vx5hU1yqFsvaRRsRLwL+18PHoVo59z+h7RHwm73XB482segVQTUNtnv5iZpnUUOkA2sCJ1MwyJ/fMpkpHUTwnUjPLJDftzcxSqqIKaXkXLTEz2xZNg03FbK2R9BtJL0t6Jq+sr6T7kkWT7pPUJ++z8ZKWJYsinVRMvE6kZpZJDUVuRbgJOHmrssuBWRExFJiVvEfScHITgUYkx1yXTPYpyInUzDKnlDXSiJgDvLJV8enAlOT1FODjeeXTImJzRCwHlgGHt3YNJ1Izy6Ao+j+gX9OiRMk2rogLDGhamjP5uXNSvhuwIm+/lUlZQR5sMrNMasOo/boSzmxqbuGjVse9XCM1s0wq86IlayQNBEh+vpyUrwR2z9tvEPBSaydzIjWzzCllH2kLpgNjk9djeefpGtOBMZK6StoDGArMa+1kbtqbWSY1qMj6Ziu7SZpKbm2PfpJWAhOAScBtks4HXgTOAoiIRZJuAxYD9cBFyaLxBTmRmlnmlHLRkog4p4WPPtzC/hPJLctZNCdSM8ukqKK5TU6kZpZJnmtvZpZCbkTeNVIzs1RcIzUzSyEo3ah9e3AiNbNMco3UzCyVcB+pmVkafvidmVkJNLpGama27do02JQBTqRmlknuIzUzS8l9pGZmKQThPlIzs7SqJ406kZpZRjV6sMnMbNsF0FBFdVInUjPLJPeRmpmlkJvZ5ERqZpaKb38yM0vFi5aYmaXipr2ZWUohqPftT2Zm6bhGamaWkvtIzcxS8Fx7M7MScCI1M0shgPoqupPUidTMMqlRlY6geE6kZpY5vo/UzCw1DzaZmaXiZfTMzErANVIzsxSC4G01VDqMojmRmlnmuGlvZlYCTqRmZikE0FBFqz8ponqCLQVJa4G/VzqOMukHrKt0EFa0jvz7el9E9N/WgyXNIPfnU4x1EXHytl6rFDpdIu3IJM2PiEMrHYcVx7+vjqOm0gGYmVU7J1Izs5ScSDuWyZUOwNrEv68Own2kZmYpuUZqZpaSE6mZWUpOpFVEUkj6bd77OklrJf0xef8ZSddudcxsSb7Fpowk7SJpmqTnJS2WdI+kfSSNkHS/pP+WtFTSt5UzWtKjW52jTtIaSQMl3STpzKR8tqQlkhZKek7StZJ2rMgXtRY5kVaXjcBISd2T9ycAqyoYT6cnScCdwOyI2CsihgNXAAOA6cCkiNgHOBA4CvgSMAcYJGlI3qmOB56JiNXNXObciDgAOADYDNxVru9j28aJtPrcC3wkeX0OMLWCsRh8EHg7In7ZVBARC4B9gIcjYmZStgm4GLg8IhqB3wNn551nDK38LiNiC/ANYLCkA0v5JSwdJ9LqMw0YI6kbuRrK3K0+P1vSgqYNcLO+vEYCjzdTPmLr8oh4HugpqRe5pDkGQFJX4FTg9tYuFhENwFPAsHRhWyl50ZIqExELkybhOcA9zexya0Rc3PRG0ux2Cs3eTdDi8kUREY9J6ilpX2A/4K8R8Wobzm0Z4kRanaYDPwRGAztVNpRObxFwZgvlx+YXSNoT+GdEbEiKppGrle5HkV00kmqB/YFntzVgKz037avTb4CrI+LpSgdi3A90lfSFpgJJhwFLgWMkHZ+UdQd+Bvwg79ipwKeBD5H7x7EgSV2A7wErImJhyb6BpeZEWoUiYmVE/LTScViujQ6cAZyQ3P60CLgKeAk4HfiWpCXA08BjwLV5xy4GNgH3R8TGApe5WdJC4Blg++S8liGeImpmlpJrpGZmKTmRmpml5ERqZpaSE6mZWUpOpGZmKTmR2ntIakimmD4j6feSeqQ4V/5KRr+SNLzAvqMlHbUN13hB0nueONlS+Vb7/LON17pK0tfaGqN1bE6k1pw3I+KgiBgJbAEuzP8wmV3TZhHx+eTeyZaMJrdCkllVcSK11jwI7J3UFh+QdAvwtKRaSf9H0mPJWpkXQG5ZuWTNzMWS/gTs3HSi/LVRJZ0s6QlJT0malawfcCHwH0lt+AOS+ku6PbnGY5KOTo7dSdJMSU9KuoEi5p5L+oOkxyUtkjRuq89+lMQyS1L/pGwvSTOSYx6U5EVCrEWea28tklQHnALMSIoOB0ZGxPIkGb0eEYclqxc9LGkmcDCwL7n54AOAxeSmtOaftz9wI3Bscq6+EfGKpF+Sm4v+w2S/W4D/GxEPSRoM/JncvPQJwEMRcbWkjwDvSowt+Fxyje7AY5Juj4j15GYKPRERl0q6Mjn3xeQeTHdhRCyVdARwHbmpnGbv4URqzemeLMEHuRrpr8k1uedFxPKk/ETggKb+T6A3MJTcQh1Tk+XeXpJ0fzPnPxKY03SuiHilhTiOB4bn1k4GoJekHZJrfCI59k+Silk16cuSzkhe757Euh5oBG5Nyn8H3CGpZ/J9f5937a5FXMM6KSdSa86bEXFQfkGSUPLngwv494j481b7nUrLy8flH1vM3OQaYFREvNlMLEXPbZY0mlxSHhURm5KlBbu1sHsk131t6z8Ds5a4j9S21Z+BLyYrEqHcM4q2J/cYjTFJH+pAcivIb+1R4DhJeyTH9k3KNwA75O03k1wzm2S/g5KXc4Bzk7JTgD6txNobeDVJosPI1Yib1PDOMnifItdl8AawXNJZyTUkr0hvBTiR2rb6Fbn+zyckPQPcQK6Fcye5JeSeBq4H/v/WB0bEWnL9mndIeop3mtZ3A2c0DTYBXwYOTQazFvPO3QPfAY6V9AS5LoYXW4l1BlCXrKD0XeCveZ9tBEZIepxcH+jVSfm5wPlJfIvwiktWgFd/MjNLyTVSM7OUnEjNzFJyIjUzS8mJ1MwsJSdSM7OUnEjNzFJyIjUzS+l/ABDK0eho4RW0AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Predicted y test values\n",
    "rfc_preds = gs_rfc.predict(X_test)\n",
    "\n",
    "# Plot a confusion matrix\n",
    "ConfusionMatrixDisplay.from_estimator(gs_rfc, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "          MH       0.98      0.86      0.92       485\n",
      "       COVID       0.88      0.98      0.93       497\n",
      "\n",
      "    accuracy                           0.92       982\n",
      "   macro avg       0.93      0.92      0.92       982\n",
      "weighted avg       0.93      0.92      0.92       982\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Model evaluation metrics\n",
    "print(classification_report(y_test, rfc_preds, target_names=['MH', 'COVID']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'rfc__max_depth': None,\n",
       " 'rfc__n_estimators': 200,\n",
       " 'tvec__min_df': 1,\n",
       " 'tvec__ngram_range': (1, 1),\n",
       " 'tvec__stop_words': ['toward',\n",
       "  'hereby',\n",
       "  'eleven',\n",
       "  'against',\n",
       "  'though',\n",
       "  'who',\n",
       "  'system',\n",
       "  'all',\n",
       "  'most',\n",
       "  'empty',\n",
       "  'myself',\n",
       "  'thereupon',\n",
       "  'only',\n",
       "  'may',\n",
       "  'where',\n",
       "  'because',\n",
       "  'together',\n",
       "  'fire',\n",
       "  'etc',\n",
       "  'every',\n",
       "  'own',\n",
       "  'a',\n",
       "  'take',\n",
       "  'across',\n",
       "  'forty',\n",
       "  'less',\n",
       "  'this',\n",
       "  'than',\n",
       "  'move',\n",
       "  'serious',\n",
       "  'its',\n",
       "  'about',\n",
       "  'has',\n",
       "  'keep',\n",
       "  'among',\n",
       "  'becoming',\n",
       "  'down',\n",
       "  'beforehand',\n",
       "  'everything',\n",
       "  'elsewhere',\n",
       "  'few',\n",
       "  'with',\n",
       "  'everyone',\n",
       "  'after',\n",
       "  'whoever',\n",
       "  'or',\n",
       "  'besides',\n",
       "  'here',\n",
       "  'nowhere',\n",
       "  'were',\n",
       "  'should',\n",
       "  'namely',\n",
       "  'anyway',\n",
       "  'ltd',\n",
       "  'seeming',\n",
       "  'already',\n",
       "  'rather',\n",
       "  'almost',\n",
       "  'through',\n",
       "  'being',\n",
       "  'once',\n",
       "  'seemed',\n",
       "  'next',\n",
       "  'do',\n",
       "  'below',\n",
       "  'when',\n",
       "  'our',\n",
       "  'around',\n",
       "  'enough',\n",
       "  'if',\n",
       "  'over',\n",
       "  'often',\n",
       "  'none',\n",
       "  'always',\n",
       "  'hers',\n",
       "  'thereby',\n",
       "  'although',\n",
       "  'themselves',\n",
       "  'these',\n",
       "  'yours',\n",
       "  'fill',\n",
       "  'anywhere',\n",
       "  'you',\n",
       "  'out',\n",
       "  'sixty',\n",
       "  'whereby',\n",
       "  'why',\n",
       "  'four',\n",
       "  'but',\n",
       "  'along',\n",
       "  'within',\n",
       "  'ourselves',\n",
       "  'cant',\n",
       "  'eg',\n",
       "  'further',\n",
       "  'towards',\n",
       "  'moreover',\n",
       "  'sincere',\n",
       "  'even',\n",
       "  'put',\n",
       "  'whose',\n",
       "  'whole',\n",
       "  'find',\n",
       "  'to',\n",
       "  'thick',\n",
       "  'part',\n",
       "  'under',\n",
       "  'us',\n",
       "  'any',\n",
       "  'ie',\n",
       "  'upon',\n",
       "  'detail',\n",
       "  'we',\n",
       "  'inc',\n",
       "  'up',\n",
       "  'anyone',\n",
       "  'found',\n",
       "  'thereafter',\n",
       "  'am',\n",
       "  'whereas',\n",
       "  'what',\n",
       "  'nobody',\n",
       "  'least',\n",
       "  'will',\n",
       "  'neither',\n",
       "  'not',\n",
       "  'whither',\n",
       "  'get',\n",
       "  'behind',\n",
       "  'itself',\n",
       "  'now',\n",
       "  'indeed',\n",
       "  'someone',\n",
       "  'give',\n",
       "  'mine',\n",
       "  'of',\n",
       "  'wherein',\n",
       "  'until',\n",
       "  'sometimes',\n",
       "  'noone',\n",
       "  'that',\n",
       "  'whom',\n",
       "  'last',\n",
       "  'an',\n",
       "  'cry',\n",
       "  'by',\n",
       "  'describe',\n",
       "  'however',\n",
       "  'front',\n",
       "  'how',\n",
       "  'cannot',\n",
       "  'formerly',\n",
       "  'show',\n",
       "  'amount',\n",
       "  'either',\n",
       "  'co',\n",
       "  'and',\n",
       "  'herein',\n",
       "  'onto',\n",
       "  'therefore',\n",
       "  'could',\n",
       "  'else',\n",
       "  'on',\n",
       "  'latterly',\n",
       "  'name',\n",
       "  'more',\n",
       "  'five',\n",
       "  'please',\n",
       "  'nevertheless',\n",
       "  'thus',\n",
       "  'via',\n",
       "  'whereafter',\n",
       "  'one',\n",
       "  'fifty',\n",
       "  'other',\n",
       "  'very',\n",
       "  'meanwhile',\n",
       "  'same',\n",
       "  'while',\n",
       "  'also',\n",
       "  'which',\n",
       "  'would',\n",
       "  'hereafter',\n",
       "  'them',\n",
       "  'into',\n",
       "  'since',\n",
       "  'nothing',\n",
       "  'in',\n",
       "  'can',\n",
       "  'nor',\n",
       "  'bottom',\n",
       "  'interest',\n",
       "  'never',\n",
       "  'two',\n",
       "  'six',\n",
       "  'become',\n",
       "  'otherwise',\n",
       "  'first',\n",
       "  'hence',\n",
       "  'de',\n",
       "  'they',\n",
       "  'somehow',\n",
       "  'thin',\n",
       "  'might',\n",
       "  'those',\n",
       "  'beside',\n",
       "  'herself',\n",
       "  'yourself',\n",
       "  'such',\n",
       "  'whereupon',\n",
       "  'too',\n",
       "  're',\n",
       "  'seem',\n",
       "  'both',\n",
       "  'hasnt',\n",
       "  'whenever',\n",
       "  'each',\n",
       "  'was',\n",
       "  'my',\n",
       "  'ever',\n",
       "  'the',\n",
       "  'amoungst',\n",
       "  'yet',\n",
       "  'thence',\n",
       "  'full',\n",
       "  'becomes',\n",
       "  'she',\n",
       "  'mill',\n",
       "  'therein',\n",
       "  'yourselves',\n",
       "  'before',\n",
       "  'himself',\n",
       "  'without',\n",
       "  'so',\n",
       "  'some',\n",
       "  'anyhow',\n",
       "  'anything',\n",
       "  'fifteen',\n",
       "  'their',\n",
       "  'above',\n",
       "  'became',\n",
       "  'see',\n",
       "  'side',\n",
       "  'must',\n",
       "  'ten',\n",
       "  'former',\n",
       "  'wherever',\n",
       "  'throughout',\n",
       "  'much',\n",
       "  'due',\n",
       "  'back',\n",
       "  'whatever',\n",
       "  'three',\n",
       "  'others',\n",
       "  'had',\n",
       "  'seems',\n",
       "  'still',\n",
       "  'somewhere',\n",
       "  'for',\n",
       "  'something',\n",
       "  'made',\n",
       "  'alone',\n",
       "  'several',\n",
       "  'couldnt',\n",
       "  'again',\n",
       "  'sometime',\n",
       "  'done',\n",
       "  'him',\n",
       "  'eight',\n",
       "  'there',\n",
       "  'no',\n",
       "  'off',\n",
       "  'me',\n",
       "  'have',\n",
       "  'been',\n",
       "  'are',\n",
       "  'her',\n",
       "  'he',\n",
       "  'beyond',\n",
       "  'many',\n",
       "  'your',\n",
       "  'perhaps',\n",
       "  'then',\n",
       "  'mostly',\n",
       "  'at',\n",
       "  'con',\n",
       "  'except',\n",
       "  'it',\n",
       "  'call',\n",
       "  'afterwards',\n",
       "  'everywhere',\n",
       "  'top',\n",
       "  'whence',\n",
       "  'during',\n",
       "  'ours',\n",
       "  'well',\n",
       "  'another',\n",
       "  'i',\n",
       "  'hundred',\n",
       "  'amongst',\n",
       "  'nine',\n",
       "  'per',\n",
       "  'be',\n",
       "  'is',\n",
       "  'between',\n",
       "  'go',\n",
       "  'un',\n",
       "  'from',\n",
       "  'twelve',\n",
       "  'latter',\n",
       "  'bill',\n",
       "  'twenty',\n",
       "  'his',\n",
       "  'third',\n",
       "  'hereupon',\n",
       "  'as',\n",
       "  'thru',\n",
       "  'whether',\n",
       "  'help',\n",
       "  'like',\n",
       "  'health',\n",
       "  'know',\n",
       "  'i’m',\n",
       "  'just',\n",
       "  'need',\n",
       "  'does',\n",
       "  'don’t',\n",
       "  'think',\n",
       "  'people',\n",
       "  'time',\n",
       "  'going',\n",
       "  'getting',\n",
       "  'make',\n",
       "  'today',\n",
       "  'new',\n",
       "  'right',\n",
       "  'got',\n",
       "  'long',\n",
       "  'best',\n",
       "  'say']}"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# # Best parameters (uncomment to run)\n",
    "# gs_rfc.best_params_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** Model 6's testing accuracy is 92.5% on unseen data, which means that this model surpasses the baseline model in its ability to classify reddit posts. Given a training score of 99.8%, this model has extrememly high training accuracy but is extremely overfit. This model returns a recall score for mentalhealth posts of .86, which is good, but not as high as Model 1."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Model 7: Support Vector Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.9945633707101597, 0.9175152749490835)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Define X and y\n",
    "X = df['title']\n",
    "y = df['is_mental']\n",
    "\n",
    "# Train test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n",
    "\n",
    "# Set up transformer and estimator via pipeline\n",
    "pipe_svc = Pipeline([\n",
    "    ('tvec', TfidfVectorizer(token_pattern=r'[\\w{2,}\\d*]\\S*\\w+')),\n",
    "    ('svc', SVC())\n",
    "])\n",
    "\n",
    "# Set up parameters for pipeline\n",
    "svc_params = {\n",
    "    'tvec__stop_words': [None, 'english', my_stopwords],\n",
    "    'tvec__min_df': [1, 5, 10],\n",
    "    'tvec__ngram_range': [(1,1), (1,2), (2,2)],\n",
    "    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']\n",
    "}\n",
    "\n",
    "# Instantiate GridSearchCV    \n",
    "gs_svc = GridSearchCV(pipe_svc,\n",
    "                      param_grid=svc_params,\n",
    "                      cv=5,\n",
    "                     verbose=0)\n",
    "\n",
    "# Fit GridSearchCV\n",
    "gs_svc.fit(X_train,y_train)\n",
    "\n",
    "# Accuracy score\n",
    "gs_svc.score(X_train,y_train), gs_svc.score(X_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Predicted y test values\n",
    "svc_preds = gs_svc.predict(X_test)\n",
    "\n",
    "# Plot a confusion matrix\n",
    "ConfusionMatrixDisplay.from_estimator(gs_svc, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "          MH       0.94      0.89      0.91       485\n",
      "       COVID       0.90      0.94      0.92       497\n",
      "\n",
      "    accuracy                           0.92       982\n",
      "   macro avg       0.92      0.92      0.92       982\n",
      "weighted avg       0.92      0.92      0.92       982\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Model evaluation metrics\n",
    "print(classification_report(y_test, svc_preds, target_names=['MH', 'COVID']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'svc__kernel': 'linear',\n",
       " 'tvec__min_df': 1,\n",
       " 'tvec__ngram_range': (1, 2),\n",
       " 'tvec__stop_words': 'english'}"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# # Best parameters (uncomment to run)\n",
    "# gs_svc.best_params_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** Model 7's testing accuracy is 91.8% on unseen data, which means that this model surpasses the baseline model in its ability to classify reddit posts. Given a training score of 99.5%, this model is quite overfit. This model returns a recall score for mentalhealth posts of .89, which is high, but not as high as Model 1."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Production Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Because the logistic regression model had the highest performance in terms of accuracy, I did addtional analysis on posts incorrectly classified from this model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create Dataframe with correct and incorrect predictions\n",
    "lr_df = pd.DataFrame({'title': X_test, 'true_subreddit': y_test,\n",
    "                      'predicted_subreddit': gs_lr.predict(X_test)})\n",
    "# Create Dataframe with only incorrect predictions\n",
    "lr_df = lr_df.loc[lr_df['true_subreddit'] != lr_df['predicted_subreddit']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>true_subreddit</th>\n",
       "      <th>predicted_subreddit</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>3598</th>\n",
       "      <td>Delta employee: “I f*cked the captain to keep ...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>270</th>\n",
       "      <td>Fake personality with EVERYONE</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>720</th>\n",
       "      <td>Associating negative emotions with positive im...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3487</th>\n",
       "      <td>Can someone explain to me as if I am a toddler</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3867</th>\n",
       "      <td>Employers just don't care and denial is real, ...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                  title  true_subreddit  \\\n",
       "3598  Delta employee: “I f*cked the captain to keep ...               0   \n",
       "270                      Fake personality with EVERYONE               1   \n",
       "720   Associating negative emotions with positive im...               1   \n",
       "3487     Can someone explain to me as if I am a toddler               0   \n",
       "3867  Employers just don't care and denial is real, ...               0   \n",
       "\n",
       "      predicted_subreddit  \n",
       "3598                    1  \n",
       "270                     0  \n",
       "720                     0  \n",
       "3487                    1  \n",
       "3867                    1  "
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# View Dataframe with incorrect predictions\n",
    "lr_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Define X\n",
    "X = lr_df['title']\n",
    "\n",
    "# Instantiate a CV object\n",
    "cv = CountVectorizer(stop_words='english', token_pattern = r'[\\w{2,}\\d*]\\S*\\w+')\n",
    "\n",
    "# Fit and transform the CV\n",
    "X_cv = cv.fit_transform(X)\n",
    "\n",
    "# Convert to a dataframe\n",
    "cv_df = pd.DataFrame(X_cv.todense(), columns=cv.get_feature_names_out())\n",
    "\n",
    "# Plot the top 10 words that are misclassified\n",
    "cv_df.sum().sort_values(ascending=False).head(20).plot(kind='barh')\n",
    "plt.title('Top 10 Misclassified Words')\n",
    "plt.xlabel('Work Count')\n",
    "plt.ylabel('Words');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "misclassified_words = cv_df.sum().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "22"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(shared_words)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "17\n",
      "['help', 'going', 'right', 'health', 'does', 'need', 'just', 'long', 'don’t', 'know', 'people', 'new', 'today', 'time', 'like', 'best', 'getting']\n"
     ]
    }
   ],
   "source": [
    "# created list of common words between misclassified popular and my stop words\n",
    "misclassified_shared_words = []\n",
    "for word in misclassified_words.index:\n",
    "    if word in shared_words:\n",
    "        misclassified_shared_words.append(word)\n",
    "print(len(misclassified_shared_words))\n",
    "print(misclassified_shared_words)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** The model most frequently misclassifies posts where the 17 words shown above appear. Out of the 22 words included in the \"shared_words\" list, 17 of these words are present in the misclassified posts. The original logisitic regression model didn't included the \"shared_word\" list as part of it's stop words. How will the model perform if we include the \"shared_words\" list in the stop word list?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.9979612640163099, 0.9205702647657841)"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Define X and y\n",
    "X = df['title']\n",
    "y = df['is_mental']\n",
    "\n",
    "# Train test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n",
    "\n",
    "# Set up transformer and estimator via pipeline\n",
    "pipe_lr = Pipeline([\n",
    "    ('tvec', TfidfVectorizer(token_pattern=r'[\\w{2,}\\d*]\\S*\\w+')),\n",
    "    ('logreg', LogisticRegression())\n",
    "])\n",
    "\n",
    "# Set up parameters for pipeline\n",
    "lr_params = {\n",
    "    'tvec__stop_words': [my_stopwords],\n",
    "    'tvec__min_df': [1, 5, 10],\n",
    "    'tvec__ngram_range': [(1,1), (1,2), (2,2)],\n",
    "    'logreg__C': [0.1, .9, 1, 1.1, 10]  \n",
    "}\n",
    "\n",
    "# Instantiate GridSearchCV\n",
    "gs_lr = GridSearchCV(estimator=pipe_lr,\n",
    "                      param_grid=lr_params,\n",
    "                      cv=5,\n",
    "                      verbose=0)\n",
    "\n",
    "# Fit GridSearchCV\n",
    "gs_lr.fit(X_train, y_train)\n",
    "\n",
    "# Accuracy score\n",
    "gs_lr.score(X_train, y_train), gs_lr.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Predicted y test values\n",
    "lr_preds = gs_lr.predict(X_test)\n",
    "\n",
    "# Plot a confusion matrix\n",
    "ConfusionMatrixDisplay.from_estimator(gs_lr, X_test, y_test, display_labels=['MH','COVID'], cmap='plasma');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "          MH       0.94      0.90      0.92       485\n",
      "       COVID       0.91      0.94      0.92       497\n",
      "\n",
      "    accuracy                           0.92       982\n",
      "   macro avg       0.92      0.92      0.92       982\n",
      "weighted avg       0.92      0.92      0.92       982\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Model evaluation metrics\n",
    "print(classification_report(y_test, lr_preds, target_names=['MH', 'COVID']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** This logistic regression model performs virtually the same - r/mentalhealth recall is marginally higher (it misclassified 48 posts instead of the original 50), but the model is still slightly overfit."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Evaluate tfidf weights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>feature_name</th>\n",
       "      <th>tfidf_weight</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2372</th>\n",
       "      <td>landlord</td>\n",
       "      <td>8.294377</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2763</th>\n",
       "      <td>mundo</td>\n",
       "      <td>8.294377</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2747</th>\n",
       "      <td>motive</td>\n",
       "      <td>8.294377</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2749</th>\n",
       "      <td>mounted</td>\n",
       "      <td>8.294377</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2750</th>\n",
       "      <td>mounts</td>\n",
       "      <td>8.294377</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2751</th>\n",
       "      <td>mourning</td>\n",
       "      <td>8.294377</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2752</th>\n",
       "      <td>movements</td>\n",
       "      <td>8.294377</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2753</th>\n",
       "      <td>movie</td>\n",
       "      <td>8.294377</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2756</th>\n",
       "      <td>mta</td>\n",
       "      <td>8.294377</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2757</th>\n",
       "      <td>mthfr</td>\n",
       "      <td>8.294377</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     feature_name  tfidf_weight\n",
       "2372     landlord      8.294377\n",
       "2763        mundo      8.294377\n",
       "2747       motive      8.294377\n",
       "2749      mounted      8.294377\n",
       "2750       mounts      8.294377\n",
       "2751     mourning      8.294377\n",
       "2752    movements      8.294377\n",
       "2753        movie      8.294377\n",
       "2756          mta      8.294377\n",
       "2757        mthfr      8.294377"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Instantiate and fit the transformer\n",
    "tvec = TfidfVectorizer(stop_words='english', min_df=1)\n",
    "X_train_tvec = tvec.fit_transform(X_train)\n",
    "X_test_tvec = tvec.transform(X_test)\n",
    "\n",
    "# Instantiate and fit logistic regression\n",
    "logreg = LogisticRegression(C=1)\n",
    "logreg.fit(X_train_tvec, y_train)\n",
    "\n",
    "# Create a dataframe showing the feature name and tf-idf weight of the word\n",
    "my_dict = dict(zip(tvec.get_feature_names_out(), tvec.idf_))\n",
    "features_df = pd.DataFrame(data=my_dict.items(), columns=['feature_name','tfidf_weight'])\n",
    "\n",
    "# View the 10 highest weighted feature names\n",
    "features_df.sort_values(by=['tfidf_weight'], ascending=False).head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretion:** Words with a weight of 8.294377 influence the classification the most."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9757721587255491"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# ROC AUC Score\n",
    "roc_auc_score(y_test, gs_lr.predict_proba(X_test)[:, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/ericrodriguez/opt/anaconda3/lib/python3.8/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_roc_curve is deprecated; Function :func:`plot_roc_curve` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: :meth:`sklearn.metric.RocCurveDisplay.from_predictions` or :meth:`sklearn.metric.RocCurveDisplay.from_estimator`.\n",
      "  warnings.warn(msg, category=FutureWarning)\n"
     ]
    },
    {
     "data": {
      "image/png": "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jh07hqqSlJREpUqVAGjfvj233HILycnJFCx4es8G+70iOIFXftqeIQhm0w/wdnv47iUoXNIViTMmG1atWkXLli2zXGf+/PmMHj2aGTNmnDD/rbfeolixYqxYsYLHHnuMJUuWANCsWTMqVapEXFwct956KxMnTkzbpl+/frz22mssWbKEYcOGcc899wBwwQUXsGDBApYtW0bPnj1POBNdsmQJX331FR9//DH33XcfF154IT/++CNLly6lUSOX5NauXUv//v1ZtWoVpUuXZvz48Rw4cIADBw5Qp06dDI+rV69ejBs3DoAFCxZQrlw56tate9J6c+fOpVWrVidMx8XFUadOHTp27MiUKcGGVydoLIGGDh1K8+bNT/q57777Tlp3y5YtVK/+10CN1apVY8uWkwd5bNasGePHjwfgyy+/5MCBA+zatYt27drRqVMnKleuTOXKlenatWta81iBAgU466yz+PHHH4PGHIyvNCIiDwZMFgBaAtnrnxRNjh6Eb59yJSJKVYM+4+Gsi8MdlTlNWZ2555b+/fvz/fffU7hwYRYtWgRwwhlnoDlz5qR9OTVt2jSt+ScmJoapU6eyaNEivv32Wx544AGWLFnCQw89xLx587juuuvS3uPo0aOAa0664YYb2LZtG8eOHTuhj/pVV11F0aJFAXcFk9quHhMTQ6lSpdizZw9xcXE0b94cgFatWrFhwwZUNcseLj179uS8887jpZdeYty4cSedRafatm0bFSpUSJseO3YsPXv2THuPDz/8kL/97W+Z7ktEgsYSaODAgQwcONDXuhk1LWW0n2HDhjFgwADef/99OnToQNWqVSlYsCDr1q1jzZo1aU2BXbp0Yc6cOXTo0AGAihUrsnXr1hMS4anwez0RWOUsGXevYPxp7Tk/SzkGq7+Ctn+HiwZZkThzyho1apR2pgjwxhtvsHPnTlq3/msgv+LFi2e6fVZffm3btqVt27Z06dKFW2+9lQcffJDSpUuzfPnyk9a/9957efDBB7nqqquYNWsWgwcP9rX/VGec8Ve59JiYGI4cOUJsbCzFixdn/fr1J7Tvp6pevTq1atVi9uzZjB8/nvnz52f43kWLFk178jslJYXx48cTHx/Ps88+m/aA1YEDByhXrtxJ7fO7d+8mLi4uaCyBhg4dypgxY06a36FDB4YPH37CvGrVqjFr1qy06YSEBDp2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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Receiver Operating Characteristic (ROC) Curve\n",
    "plot_roc_curve(gs_lr, X_test, y_test)\n",
    "plt.plot([0, 1], [0, 1],\n",
    "        label='baseline', linestyle='--')\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** The area under the ROC curve tells how much the model is capable of distingusihing between classes. More area under the curve means the distributions are better separated. This ROC AUC of 0.98 indicates that the true positive rate and the false positive rate are well separated."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Since my production model utilizing the custom stopword list has a marginally higher recall score\n",
    "# for mentalhealth posts, I'm going to pickle this model \n",
    "with open('../data/production_model.pkl', 'wb') as pickle_out:\n",
    "    pickle_out = pickle.dump(gs_lr, pickle_out)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusion\n",
    "\n",
    "My baseline model had an accuracy score of 50%, and my logistic regression model has an accuracy of 92.1%, showing that the logisitc regression model is significantly better at predicting the correct subreddit.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
