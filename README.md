# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3 - Reddit Classification Problem

---

## Overview
- [Web App](#Streamlit-Application)
- [Technologies Used](#Technology-&-Methods)
- [Executive Summary](#Executive-Summary)
- [Introduction](#Introduction)
  - [Problem Statement](#Problem-Statement)
  - [Industry Relevance](#Industry-Relevance)
- [Data](#Data)
  - [Data Source](#Data-Source)
  - [Data Dictionary](#Data-Dictionary)
  - [Data Cleaning](#Data-Cleaning)
- [Modeling](#Modeling)
  - [Pre-processing](#Pre-processing)
  - [Models](#Model-Evaluation)
- [Conclusion](#Conclusion)
  - [Results](#Results)
  - [Next Steps](#Next-Steps)

---

## Streamlit Application
Checkout the streamlit application I created that demonstrates how my model can be utilized - hosted by [Heroku!](https://ericnrodriguez-reddit-nlp-app.herokuapp.com)

---

## Technology & Methods

**Technology:** Python, Jupyter Notebook, GitHub, Git, nltk, requests, pandas, numpy, matplotlib, scikit-learn, requests, regex, time, pickle, PIL, wordcloud, Streamlit, Heroku, Tableau

**Models:** Logistic Regression, Decision Tree Classifier, Bagging Classifier, Multinomial Naive Bayes, AdaBoost Classifier, Random Forest Classifier, Support Vector Classifier, TF-IDF Vectorization

**Skills:** Webscraping API, Sentiment Analysis, Natural Language Processing (NLP), Cross Validation, Train-Test-Spliting, Count Vectorization, Confusion Matrix, Pipeline, Gridsearch, Word Clouds, Pickling, Custom Web Apps

## Executive Summary

In this project, I sought to achieve a model that correctly classifies reddit posts from either r/mentalhealth and r/CoronavirusUS with at least 80% accuracy. I accomplished this goal by utilizing natural language processing and tuning a Logistic Regression model that yielded a 92% testing accuracy.

---

## Introduction
### Problem Statement

The objective of this project is to pick two subreddits from Reddit and train a machine learning model that will be able to classify new posts into the correct subreddit.

I identified r/mentalhealth and r/CoronavirusUS as my two subreddits of interest because the two subreddits share enough similarities to provide a challenge, but they are also different enough that it should be feasible to utilize Natural Language Processing to classify new posts into the correct subreddit.

My models success will be defined as a model that has an accuracy score of over 80%. In the effort of creating a screening tool that minimizes the misclassification of posts from r/mentalhealth, the recall score for these posts will be evaluated as well in order to determine the production model.

### Industry Relevance

Reddit can be described as a social news aggregation, web content rating, and discussion website.

The platform contains two large health related subreddits.

The mental health subreddit consists of 227,000 members and is the largest and most active mental health focused subreddit on the website.

Individuals who are active on r/mentalhealth tend to discuss, vent, support, and share information about their mental health, both their struggles, and their successes.

The CoronavirusUS subreddit consists of 146,000 members and amassed over 100,000 members within its first two months of creation. At its peak, over 4,000 comments were being left in the subreddit every day.

Individuals who are active on the CoronavirusUS subreddit share news links and information regarding COVID-19 with respect to government mandates and medical news updates, as well as questions aiming to clarify any information regarding the status of COVID-19 and how to protect themselves from it.

According to the National Institute of Mental Health, nearly one in five US adults live with a mental illness (52.9 million in 2020). There are many barriers to screening for mental health conditions, including desire to receive care, lack of anonymity when seeking care, a shortage of mental health professionals, lack of culturally competent care, affordability, and transportation just to name a few.

The coronavirus pandemic exacerbates these barriers to screening. Furthermore, individuals have flocked to websites such as reddit in order to discuss physical health problems stemming from COVID, which creates a lot of noise in the context of mental health since these posts may use similar language to posts describing mental health challenges. As such, a model that can process text data and passively identify individuals who may require mental health care could be a valuable screening tool in order to connect these individuals with the care that they need, especially in the era of COVID.

## Data
### Data Source
Data was scraped from reddit using the pushshift.io API. I scraped reddit posts from January 2021, because January is known as the time of the year people are most depressed, anxious, and stressed. My rationale for scraping posts from this date range was that Google search interest for the term "seasonal affective disorder" peaks in January, which led me to assume that there would be a relatively high amount of people on Reddit posting about their seasonal affective disorder. Mental health experts consistently find that suicides peak in the Spring (for reasons that aren't fully clear); passive mental health screening in the months leading up to the Spring may be critical in curbing these high suicide rates.

### Data Dictionary
| Feature           | Type   | Description                                                             |
|-------------------|--------|-------------------------------------------------------------------------|
| created_utc       | int64  | Unix time post was created                                              |
| url               | object | URL of subreddit post                                                   |
| num_comments      | int    | Number of comments in the title of subreddit post                       |
| title             | object | Title of subreddit post                                                 |
| selftext          | object | Body of subreddit post                                                  |
| subreddit         | object | Name of subreddit post is pulled from                                   |
| timestamp         | object | The time the post was created in EST                                    |
| is_mental         | int    | Subreddit the post came from. 1 if r/mentalheatlh, 0 if r/CoronavirusUS |
| title_char_length | int    | Number of characters in the title                                       |
| title_word_count  | int    | Number of words in the title                                            |
| sentiment_comp    | float  | Composite sentiment score of title                                      |
| sentiment_neg     | float  | Negative sentiment score of title                                       |
| sentiment_neu     | float  | Neutral sentiment score of title                                        |
| sentiment_pos     | float  | Positive sentiment score of title                                       |

### Data Cleaning
After performing exploratory data analysis, random characters and hyperlinks were removed from the title and selftext columns.

## Modeling
### Pre-processing
Reddit post's title character length and word count length were calculated and added to the DataFrame. Sentiment analysis scores for titles were calculated and added to the DataFrame as well. I created a list for reach subreddit of the 100 words of highest frequency, and then created a DataFrame containing the words that each subreddit shared between these two lists, with the frequency of those words in each subreddit. I then defined my target variable and features, and train-test-split my data.

### Model Evaluation

I utilized seven different types of classification models, and the accuracy of each model is presented below:

| **Model**                 | **Train Accuracy** | **Test Accuracy** |
|---------------------------|--------------------|-------------------|
| Logistic Regression       | 99.8%              | 92.1%             |
| Decision Tree Classifier  | 90.3%              | 89.5%             |
| Bagging Classifier        | 99.1%              | 91.6%             |
| Multinomial Naive Bayes   | 99.3%              | 90.6%             |
| Adaboost Classifier       | 91.6%              | 88.9%             |
| Random Forest Classifier  | 99.8%              | 92.5%             |
| Support Vector Classifier | 99.5%              | 91.8%             |

## Conclusion

### Results
My baseline model had an accuracy score of 50%, and my Logistic Regression model had a test accuracy of 92.1%, showing that the Logistic Regression model is significantly better at predicting the correct subreddit than my baseline model.

The Logistic Regression model had the highest train accuracy, and the 3rd highest test accuracy. The overall recall score was 92%.  For posts that came from r/mentalhealth, the model correctly classified 90% of these posts.  For posts that came from r/CoronavirusUS, the model correctly classified 94% of these posts. Because of these results, I am implementing this model as my production model.

### Next Steps
Scraping more data from both subreddits could make my model stronger and decrease the amount of variance and overfitting exhibited by the production model. More data may also help illuminate the most important features that help predict the correct classification.

I would also like to test a model that utilizes all the features at my disposal, such as one that combines the title and selftext features, and models that utilize the polarity scores. Further analysis can then be done on these features in order to establish which ones have the highest influence on predictions.

The data cleaning and TF-IDF vectorization that I utilized omits certain characters and emojis during classification. A model which includes these types of characters and emojis could increase the generalization of my model and increase its ability to classify posts.
