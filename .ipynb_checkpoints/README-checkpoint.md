# Project 3 - Reddit Classification Problem

## Executive Summary

### Problem Statement

The objective is to pick two subreddits from Reddit and train a machine learning model that will be able to classify new posts into the correct subreddit. 

I identified r/mentalhealth and r/coronavirus as my two topics of interest because the two topics (i.e. based on real-world knowledge) share enough similarities to provide a good challenge, but are also differentiated enough that it should be possible to train a machine learning model. 

### Industry relevance

Reddit can be described as a social news aggregation, web content rating and discussion website 

The platform contains two large health related subreddits

The mental health subreddit consists of  227k members, and is the largest and most active mental health focused subreddit on the website

Individuals who are active on the site tend ot discuss, vent, support, and share information about thei rmental health, both their struggles, and their successfess

The coronavirus subreddit consists of 2.4 million members, and at one point was the 2nd more active subreddit in 2020. 

Individuals who are active on the coronavirus subreddit share news links and information regarding covid-19 with respect to government mandates and medical news updates, as well as questions aiming to clarify any information regarding the status of covid-19 and how to protect themselves from it

### Methodology

Data was scraped from reddit using the pushshift.io API. Post titles were cleaned, stop words were removed, and unecessary letters and symbols were removed. Posts from both subreddits were then merged onto one data frame.

Training testing and splitting was performed on the data frame, and clean titles were count vectorizered. From there I fit three models:

**Random Forest Classifier** which had:

CV score: 0.9390351391598873
Training accuracy: 0.9880239520958084
Testing accuracy: 0.9412724306688418

**Multinomial Naive Bayes** which had:

CV score: 0.9466497492598634
Training accuracy: 0.9831246597713663
Testing accuracy: 0.9592169657422512

**Logistic Regression with lasso regularization** which had:
CV score: 0.932504522563058
Training accuracy: 0.9624387588459444
Testing accuracy: 0.9429037520391517

The testing accuracy for the logistic regression model is closest to its training accuracy relative to the other two models, meaning that it is the least overfit model out of the three models

**Because of this, I created predictions using the Logistic Regression model**

### Model Evaluation

The accuracy score shows that the model is correctly predicting that a post comes from the correct subreddit 94.29% of the time, and misclassifying the posts 5.71% of the time.

The precision score shows that out of all of the posts that are actually from the r/mentalhealth subreddit, the model predicted that a post came from r/mentalhealth 90.52% of the time

Out of the posts predicted to be from r/mentalhealth, the model correctly predicted these posts 98.67% of the time.

Out of the posts predicted to be from r/Coronavirus, the model correctly predicted these posts 90.1% of the time.

### Conclusion and Recommendations

My baseline model had an accuracy score of 51%, and my logistic regression model has an accuracy of 94.29%, showing that the logisitc regression model is significantly better at predicting the correct subreddit.

The accuracy may have been high for a few reasons - although there was a lot of talk regarding mental health on r/Coronavirus at the beginning of the pandemic, there is less discussion about mental health now. The data scraped was from the last few weeks (as of 3.9.21), and the context of the posts on r/Coronavirus has changed to discussiong related to vaccines and government policy.

In order to make the model stronger, I need to scrape older data that has more overlap with the mental health subreddit. This may help illuminate the most important features that help predict correct classification.

Moving forward, I can use the model on the US Coronavirus subreddit. There are more foum based posts on that subreddit, and it may have more overlap than the general coronavirus subreddit which contains active users from all over the world. There appears to be more similair language on the US coronavirus subreddit, and it would challenge my model more than the general Coronavirus subreddit.

I can analyze the highest occuring words from both subreddits. This would help us understand what words the model considers the most important when predicting the outcome. This could also help us understand what words are indiciative of change sin mental health incidence and prevlence, which could allow us to utilize or generalize this model to text based web content outside of the mental health subreddit.

Next steps will be using my own list of stop words, which could exlcude words such as 'coronavirus' or 'mental health'. Words will also be lemmatized.