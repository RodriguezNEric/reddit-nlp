#!/usr/bin/env python
# coding: utf-8

# # Project 3: Reddit Scraping & NLP
# 
# ## Part 2 - Data Cleaning and EDA

# In[2]:


# General imports
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle

# For Natural Language Processing
import regex as re
import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ### Read in and Explore Data

# In[3]:


# Read in both subreddit datasets
mental_df = pd.read_csv('../data/mentalhealth_211001_171956.csv', encoding = 'utf-8')
covid_df = pd.read_csv('../data/CoronavirusUS_211001_172844.csv', encoding = 'utf-8')

# Combine into single dataframe
df = pd.concat([mental_df, covid_df])
df = df.reset_index(drop=True)

# Create target column where mentalhealth == 1, CoronavirusUS == 0
df['is_mental'] = df['subreddit'].map(lambda t: 1 if t == 'mentalhealth' else 0)


# In[4]:


# View dataframe
df.head()


# In[5]:


# View shape
df.shape


# In[6]:


# Check datatypes 
df.dtypes


# In[7]:


# Check for nulls
df.isnull().sum()


# There is 1244 null cells in the "selftext" column

# In[8]:


# Check the number of selftext nulls by subreddit group
df['selftext'].isnull().groupby(df['is_mental']).sum()


# It is clear here that there are more missing selftext values in the r/CoronavirusUS subreddit than in the r/mentalhealth subreddit.

# In[9]:


# View titles from r/mentalhealth
df.loc[df['is_mental'] ==1, 'title'].head(20)


# In[10]:


# View titles from r/CoronavirusUS
df.loc[df['is_mental'] == 0, 'title'].head(20)


# First 20 examples of r/CoronavirusUS contain language that could be applicable to mental health issues
# 
# Both subreddits contain examples of unnecessary capital lettering

# In[11]:


# Check for duplicate and empty non-NaN titles
df.title.value_counts()


# In[12]:


# Check for duplicate and empty non-NaN selftext values
df.selftext.value_counts()


# The selftext column has values that need to be converted from "[removed]", "[deleted]", "?", and "." to np.NaN
# 
# There are also links to other articles which starts with [http], which will need to be cleaned

# In[13]:


# Check for duplicate values
df['title'].duplicated().value_counts()


# There are 68 duplicate titles

# In[14]:


# Check the number of duplicates by subreddit group
df['title'].duplicated().groupby(df['is_mental']).sum()


# In[15]:


# Verify distrubtion
df.subreddit.value_counts(normalize=True)


# Getting rid of rows with duplicate titles doesn't significantly change the distriubtion of subreddit posts 

# ### Data Cleaning

# In[16]:


# Replace '[removed]' '[deleted]', '.', '?' with NaN in the 'selftext' column
df['selftext'] = df['selftext'].replace(
    {'[removed]': np.nan, '[deleted]': np.nan,'.': np.nan, '?': np.nan})

# Drop duplicates based on title column
df = df.drop_duplicates(subset='title')


# In[17]:


# Remove hyperlinks from title and selftext h/t Philip DiSarro on stackoverflow
df['title'] = df['title'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
df['selftext'] = df['selftext'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

# Replace cells that only contain "[" and "" and "\n" with NaN in the 'selftext' column 
# which are a result of splitting and removing hyperlinks
df['selftext'] = df['selftext'].replace({'[': np.nan, '\n': np.nan})
df['selftext'] = df['selftext'].replace(r'^\s*$', np.nan, regex=True)

# Replace "&amp;" with "and" from title and selftext
df['title'] = df['title'].replace('&amp;', 'and', regex=True)
df['selftext'] = df['selftext'].replace('&amp;', 'and', regex=True)


# In[18]:


# Drop inappropriate sequence of rows in the title column
# Drop like this to ommit innappropriate values
df = df[df.title.str.len() > 2]

# Reset index again
df.reset_index(inplace = True, drop = True)


# In[19]:


# Create a column for title character length
df['title_char_length'] = df['title'].apply(len)

# Create a column for title word count
df['title_word_count'] = df['title'].map(lambda x: len(x.split()))

df.head()


# **Interpretation:** Both subreddits are skewed to the right, with r/mentalhealth having a larger distribution of character lengths on the shorter side than r/CoronavirusUS.

# In[20]:


# Histogram of title character length
plt.figure(figsize=(10,8))
plt.hist(x=df[df['is_mental']==1]['title_char_length'],
         bins=25, alpha=0.5, label='MH')
plt.hist(x=df[df['is_mental']==0]['title_char_length'],
         bins=25, alpha=0.5, label='COVID')
plt.title('Distribution of Title Character Length')
plt.xlabel('Character Length')
plt.ylabel('Frequency')
plt.legend();


# **Interpretation:** The title character length of both subreddits are skewed to the right, following a similair distribution.

# In[21]:


# Histogram of title word count
plt.figure(figsize=(10,8))
plt.hist(x=df[df['is_mental']==1]['title_word_count'],
         bins=25, alpha=0.5, label='MH')
plt.hist(x=df[df['is_mental']==0]['title_word_count'],
         bins=25, alpha=0.5, label='COVID')
plt.title('Distribution of Title Word Count')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.legend();


# **Interpretation:** The title world count of both subreddits are skewed to the right, following a similair distribution.

# #### Out of curiousity, I'm going to check the sentiment of my posts

# In[22]:


# Instantiate sentiment intesnity analyzer
sia = SentimentIntensityAnalyzer()

# create a column for the negative, neutral, positive, and compound analysis scores:
df['sentiment_comp'] = df['title'].map(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment_neg'] = df['title'].map(lambda x: sia.polarity_scores(x)['neg'])
df['sentiment_neu'] = df['title'].map(lambda x: sia.polarity_scores(x)['neu'])
df['sentiment_pos'] = df['title'].map(lambda x: sia.polarity_scores(x)['pos'])

# Check code execution
df.head(3)


# In[23]:


# Average sentiment of r/mentalhealth
print(df[df['is_mental']==1][['sentiment_comp']].mean())
print(df[df['is_mental']==1][['sentiment_neg']].mean())
print(df[df['is_mental']==1][['sentiment_neu']].mean())
print(df[df['is_mental']==1][['sentiment_pos']].mean())


# In[24]:


# Average sentiment of r/CoronavirusUS
print(df[df['is_mental']==0][['sentiment_comp']].mean())
print(df[df['is_mental']==0][['sentiment_neg']].mean())
print(df[df['is_mental']==0][['sentiment_neu']].mean())
print(df[df['is_mental']==0][['sentiment_pos']].mean())


# In[25]:


# Boxplot for sentiment compound scores
data_1 = df[df['is_mental']==1]['sentiment_comp']
data_2 = df[df['is_mental']==0]['sentiment_comp']
data = [data_1, data_2]

fig, ax = plt.subplots(figsize=(8,10))
bp = ax.boxplot(data, widths = .4, patch_artist=True)
colors = ['lightblue', 'peachpuff']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
for median in bp['medians']:
    median.set(color='red')
for whisker in bp['whiskers']:
    whisker.set(color='peru', linewidth=2, linestyle=":")
for cap in bp['caps']:
    cap.set(color='saddlebrown')
for flier in bp['fliers']:
    flier.set(color='azure')
ax.set_xticklabels(['MH', 'COVID'])
plt.ylabel("Compound Score")
plt.xlabel("Subreddits")
plt.title("Boxplot of Sentiment Compound Score")
plt.show();


# **Interpretation:** The mentalhealth subreddit is distributed more towards a negative compound score, whereas the CoronavirusUS subreddit is distributed within a more neutral-positive coumpound score. Both median values are close to a coumpound sentiment value of zero.

# In[26]:


# Count the most commonly used words for r/mentalhealth

# Define X
X = df[df['is_mental']==1]['title']

# Instantiate a CV object
cv = CountVectorizer(stop_words='english', analyzer = 'word', token_pattern = r'[\w{2,}\d*]\S*\w+')

# Fit and transform the CV data on posts
X_cv = cv.fit_transform(X)

# Convert to a dataframe
cv_df = pd.DataFrame(X_cv.todense(),columns=cv.get_feature_names_out())

# Create a mh dataframe containing the 100 most common words and word count
mh_cv_df = pd.DataFrame(data=cv_df.sum().sort_values(ascending=False).head(100),
                        columns=['word_count'])

# Change index to a column
mh_cv_df.reset_index(inplace=True)
mh_cv_df = mh_cv_df.rename(columns={'index':'word'})

# Save wsb_cv_df to csv
mh_cv_df.to_csv('../data/mh_words.csv', index=False)

# Check code execution
mh_cv_df.head()


# In[28]:


# Bar chart of the 20 most commonly seen words in r/mentalhealth
plt.figure(figsize=(12,8))
plt.barh(y=mh_cv_df['word'].head(20), width=mh_cv_df['word_count'].head(20))
plt.title('20 Most Common Words in r/mentalhealth')
plt.xlabel('Word Count')
plt.ylabel('Word');


# In[29]:


# Count the most commonly used words for r/CoronavirusUS

# Define X
X = df[df['is_mental']==0]['title']

# Instantiate a CV object
cv = CountVectorizer(stop_words='english', analyzer = 'word', token_pattern = r'[\w{2,}\d*]\S*\w+')

# Fit and transform the CV data on posts
X_cv=cv.fit_transform(X)

# Convert to a dataframe
cv_df = pd.DataFrame(X_cv.todense(),columns=cv.get_feature_names_out())

# Create a mh dataframe containing the 100 most common words and word count
covid_cv_df = pd.DataFrame(data=cv_df.sum().sort_values(ascending=False).head(100),
                        columns=['word_count'])

# Change index to a column
covid_cv_df.reset_index(inplace=True)
covid_cv_df = covid_cv_df.rename(columns={'index':'word'})

# Save wsb_cv_df to csv
covid_cv_df.to_csv('../data/covid_words.csv', index=False)

# Check code execution
covid_cv_df.head()


# In[30]:


# Bar chart of the 20 most commonly seen words in r/CoronavirusUS
plt.figure(figsize=(12,8))
plt.barh(y=covid_cv_df['word'].head(20), width=covid_cv_df['word_count'].head(20))
plt.title('20 Most Common Words in r/CoronavirusUS')
plt.xlabel('Word Count')
plt.ylabel('Word');


# In[31]:


# Find matches for the 100 most common words in both subreddits
common_words = []
for word in mh_cv_df['word']:
    if word in list(covid_cv_df['word']):
        common_words.append(word)
print(len(common_words))
print(common_words)


# **Interpretation:** Of the 100 most commonly found words in the mentalhealth and CoronavirusUS subreddits, 22 words are shared between the two. This indicates that there is overlapping terminology in both subreddits, which may limit the accuracy of any predictive models built to distinguish the two subreddits.

# In[32]:


common_words_df = pd.DataFrame(data = common_words, columns = ['shared_word'])
common_words_df.head()


# In[33]:


# Create a dictionary for r/mentalhealth shared word count
mh_shared_word_count = {}
for word in common_words_df['shared_word']:
    index = 0
    for i in mh_cv_df['word']:
        if word == i:
            mh_shared_word_count[word] = mh_cv_df['word_count'][index]
        index += 1


# In[34]:


# Create a new column with r/mentalhealth word count
common_words_df['mh_word_count'] = common_words_df['shared_word'].map(mh_shared_word_count)

common_words_df.head()


# In[35]:


# Create a dictionary for r/CoronavirusUS shared word count
covid_shared_word_count = {}
for word in common_words_df['shared_word']:
    index = 0
    for i in covid_cv_df['word']:
        if word == i:
            covid_shared_word_count[word] = covid_cv_df['word_count'][index]
        index += 1


# In[36]:


# Create a new column with r/mentalhealth word count
common_words_df['covid_word_count'] = common_words_df['shared_word'].map(covid_shared_word_count)

common_words_df.head()


# In[37]:


# Check length of dataframe to make sure it equals 22
len(common_words_df)


# In[38]:


# Boxplot of 24 most common words shared between both subreddits
data_1 = common_words_df['mh_word_count']
data_2 = common_words_df['covid_word_count']
data = [data_1, data_2]

fig, ax = plt.subplots(figsize=(12,8))
words = common_words_df['shared_word']
x = np.arange(len(words))
ax = fig.add_axes([0,0,1,1])
bar_width = 0.375

br1 = np.arange(len(data[0]))
br2 = [x + bar_width for x in br1]
ax.bar(br1, data[0], color='deepskyblue', width=-bar_width, label='MH', align='edge')
ax.bar(br2, data[1], color='orange', width=-bar_width, label='COVID', align='edge')
ax.set_xticks(x, words.values)
          
plt.title('Barplot of Most Common Words Shared between r/mentalhealth and r/CoronavirusUS')
plt.xlabel('Shared Words')
plt.ylabel('Word Count')
plt.legend()
plt.show();


# **Interpretation:** Common words between the two subreddits typically aren't very similiair in word count. Many words that have a high word count in the mental health subreddit have a low word count in the coronavirusUS subreddit, and vice versa. 

# In[39]:


# Save common_words_df to csv
common_words_df.to_csv('../data/shared_words.csv', index = False)


# In[40]:


# Save cleaned posts as csv file
df.to_csv('../data/cleaned_posts.csv', index=False)

