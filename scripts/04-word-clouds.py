#!/usr/bin/env python
# coding: utf-8

# # Project 3: Reddit Scraping & NLP
# 
# ## Part 4 - Word Clouds

# In[1]:


# For wordclouds
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


# In[2]:


# Read in both subreddit datasets
df = pd.read_csv('../data/cleaned_posts.csv')


# **Wordcloud Function**
# 
# Generating wordclouds of most frequent words in each subreddit.
# Thanks to inspiration from [Creating Stylish, High-Quality Word Clouds Using Python and Font Awesome Icons](https://minimaxir.com/2016/05/wordclouds/) and [Martin Evans on stackoverflow for Color Adjustment](https://stackoverflow.com/questions/43043263/word-cloud-in-python-with-customised-random-colour-generation)

# In[3]:


# Function to create wordcloud
def gen_wordcloud(text, img_name, color):
    
    # Function to generate random word colors
    def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
        if color=='red':
            h_val = 0
        elif color=='green':
            h_val = 130
        elif color=='orange':
            h_val = 20

        h = int(h_val)
        s = int(70)
        l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

        return "hsl({}, {}%, {}%)".format(h, s, l)

    # Convert png to jpg with no transparency for mask
    icon = Image.open('../images/reddit.png')
    mask = Image.new("RGB", icon.size, (255,255,255))
    mask.paste(icon,icon)
    mask = np.array(mask)

    #Generate and save images
    wc = WordCloud(width=600,
                          height=600,
                          stopwords='english',
                          regexp=r'[\w{2,}\d*]\S*\w+',
                          mask=mask,
                          prefer_horizontal=1,
                          max_words=1000,
                          background_color='white',
                          color_func=random_color_func,
                          random_state=12).generate(text)
    plt.figure(figsize=(10,10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    wc.to_file(f'../images/{img_name}.png')


# In[4]:


# Define X
X = df[df['is_mental']==1]['title']

# Instantiate a CV object
cv = CountVectorizer(stop_words='english', analyzer = 'word', token_pattern = r'[\w{2,}\d*]\S*\w+')

# Fit and transform the CV data on posts
X_cv = cv.fit_transform(X)

# Convert to a dataframe
cv_df = pd.DataFrame(X_cv.todense(),columns=cv.get_feature_names_out())

# Create a mh dataframe containing the 100 most common words and word count
mh_cv_df = pd.DataFrame(data=cv_df.sum().sort_values(ascending=False)[:4000],
                        columns=['word_count'])

# Change index to a column
mh_cv_df.reset_index(inplace=True)
mh_cv_df = mh_cv_df.rename(columns={'index':'word'})


# In[5]:


# Wordcloud for mental health
text = " ".join(word for word in mh_cv_df.word)
gen_wordcloud(text, 'mentalhealth_wc', 'red')


# In[6]:


# Define X
X = df[df['is_mental']==0]['title']

# Instantiate a CV object
cv = CountVectorizer(stop_words='english', analyzer = 'word', token_pattern = r'[\w{2,}\d*]\S*\w+')

# Fit and transform the CV data on posts
X_cv = cv.fit_transform(X)

# Convert to a dataframe
cv_df = pd.DataFrame(X_cv.todense(),columns=cv.get_feature_names_out())

# Create a covid dataframe containing the 100 most common words and word count
covid_cv_df = pd.DataFrame(data=cv_df.sum().sort_values(ascending=False)[:4000],
                        columns=['word_count'])

# Change index to a column
covid_cv_df.reset_index(inplace=True)
covid_cv_df = covid_cv_df.rename(columns={'index':'word'})


# In[7]:


# Wordcloud for CoronavirusUS
text = " ".join(word for word in covid_cv_df.word)
gen_wordcloud(text, 'covid_wc', color = 'green')


# In[8]:


# Read in the shared word dataset
df = pd.read_csv('../data/shared_words.csv')


# In[9]:


# Create new column that combines word counts from each individuals subreddit
df['shared_word_count'] = df['mh_word_count']+df['covid_word_count']


# In[10]:


# Create new column that multiplies each individual shared_word by its total shared word count
for word in df['shared_word']:
    word_length = len(word)+10  
df['shared_word_full'] = df['shared_word'].str.ljust(word_length)*df['shared_word_count']


# In[11]:


# Check dataframe
df.head()


# In[12]:


# Define X
X = df['shared_word_full']

# Instantiate a CV object
cv = CountVectorizer(stop_words='english', analyzer = 'word', token_pattern = r'[\w{2,}\d*]\S*\w+')

# Fit and transform the CV data on posts
X_cv = cv.fit_transform(X)

# Convert to a dataframe
cv_df = pd.DataFrame(X_cv.todense(),columns=cv.get_feature_names_out())

# Create a dataframe containing the shared words in order of frequency
shared_cv_df = pd.DataFrame(data=cv_df.sum().sort_values(ascending=False)[:4000],
                        columns=['mh_word_count'])

# Change index to a column
shared_cv_df.reset_index(inplace=True)
shared_cv_df = shared_cv_df.rename(columns={'index':'word'})


# In[13]:


# Wordcloud for shared_words
text = " ".join(word for word in df.shared_word)
gen_wordcloud(text, 'shared_word_wc', 'orange')


# In[ ]:




