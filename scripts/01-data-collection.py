#!/usr/bin/env python
# coding: utf-8

# # Project 3: Reddit Scraping & NLP
# 
# ## Part 1 - Scraping

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import requests
import json
import csv
import time
import datetime as dt
import math
import itertools


# In[2]:


# pushshift url template
# https://api.pushshift.io/reddit/search/submission?subreddit={}&after={}&before={}&size={}
# max request size is 100!


# ### Data Collection

# #### Scraping data from mentalhealth subreddit

# In[3]:


# Subreddit to be scraped
subreddit = 'mentalhealth'

# Time parameters
after = 1611140400 # epoch timestamp for 1/20/2021 6am GMT -04:00 DST
before = 1618912800 # epoch timestamp for 4/20/2021 6am GMT -04:00 DST

# h/t stack overflow 
# Set up dict for info to collect
posts_data_mh = {'created_utc':[],
              'url':[],
              'full_link':[],
              'id':[],
              'num_comments':[],
              'title':[],
              'selftext':[],
              'subreddit':[]
              }

headers = {'User-agent': 'Reddit Post Collector'}

# Set up function to return submission data
def get_submissions(**kwargs):
    res = requests.get("https://api.pushshift.io/reddit/submission/search/",
                       params=kwargs,
                       headers=headers)
    if res.status_code == 200:
        data = res.json()
        return data['data']
    else:
        print(res.status_code)

count = 0

# Collect up to 2,000 posts as long as there are posts to collect
while True and len(set(posts_data_mh['created_utc'])) <= 1900:
    print(count)
    count += 1*100
    
    posts = get_submissions(subreddit=subreddit,
                            size=100,
                            after=after, #pulls submissions only after this date
                            before=before, #pulls submissions only before this date
                            sort='asc', #returns data with earliest date first
                            sort_type='created_utc')
    if not posts:
        break

    for post in posts:
        # Keep track of position for the next call in while loop
        after = post['created_utc']

        # Append info to posts_data dict
        posts_data_mh['created_utc'].append(post['created_utc'])
        posts_data_mh['url'].append(post['url'])
        posts_data_mh['full_link'].append(post['full_link'])
        posts_data_mh['id'].append(post['id'])
        posts_data_mh['num_comments'].append(post['num_comments'])
        posts_data_mh['title'].append(post['title'])
        posts_data_mh['selftext'].append(post['selftext'])
        posts_data_mh['subreddit'].append(post['subreddit'])

    time.sleep(1)

# Save posts to dataframe
mentalhealth = pd.DataFrame(posts_data_mh)

# Create `timestamp` column with `created_utc` translated into readable time
def get_date(created):
    return dt.datetime.fromtimestamp(created)

_timestamp = mentalhealth['created_utc'].apply(get_date)
mentalhealth = mentalhealth.assign(timestamp = _timestamp)


# In[4]:


mentalhealth.shape


# In[5]:


# Check r/mentalhealth dataframe
mentalhealth.tail()


# In[6]:


# Check data types
mentalhealth.dtypes


# In[7]:


# # export to csv
# filetime = time.strftime("%y%m%d_%H%M%S", time.localtime())
# mentalhealth.to_csv('/Users/ericrodriguez/Documents/Projects/project 3 - nlp reddit/data/{}_{}.csv'.format(subreddit, filetime), index=False)


# #### Scraping data from CoronavirusUS subreddit

# In[10]:


# Subreddit to be scraped
subreddit = 'CoronavirusUS'

# Time parameters
after = 1611140400 # epoch timestamp for 1/20/2021 6am GMT -04:00 DST
before = 1618912800 # epoch timestamp for 4/20/2021 6am GMT -04:00 DST

# h/t stack overflow 
# Set up dict for info to collect
posts_data_cv = {'created_utc':[],
              'url':[],
              'full_link':[],
              'id':[],
              'num_comments':[],
              'title':[],
              'selftext':[],
              'subreddit':[]
              }

headers = {'User-agent': 'Reddit Post Collector'}

# Set up function to return submission data
def get_submissions(**kwargs):
    res = requests.get("https://api.pushshift.io/reddit/submission/search/",
                       params=kwargs,
                       headers=headers)
    if res.status_code == 200:
        data = res.json()
        return data['data']
    else:
        print(res.status_code)

count = 0

# Collect up to 2,000 posts as long as there are posts to collect
while True and len(set(posts_data_cv['created_utc'])) <= 1900:
    print(count)
    count += 1*100
    
    posts = get_submissions(subreddit=subreddit,
                            size=100,
                            after=after, #pulls submissions only after this date
                            before=before, #pulls submissions only before this date
                            sort='asc', #returns data with earliest date first
                            sort_type='created_utc')
    if not posts:
        break

    for post in posts:
        # Keep track of position for the next call in while loop
        after = post['created_utc']

        # Append info to posts_data dict
        posts_data_cv['created_utc'].append(post['created_utc'])
        posts_data_cv['url'].append(post['url'])
        posts_data_cv['full_link'].append(post['full_link'])
        posts_data_cv['id'].append(post['id'])
        posts_data_cv['num_comments'].append(post['num_comments'])
        posts_data_cv['title'].append(post['title'])
        try:
            posts_data_cv['selftext'].append(post['selftext'])
        except KeyError:
            posts_data_cv['selftext'].append("NaN")
        posts_data_cv['subreddit'].append(post['subreddit'])

    time.sleep(1)

# Save posts to dataframe
coronavirus = pd.DataFrame(posts_data_cv)

# Create `timestamp` column with `created_utc` translated into readable time
def get_date(created):
    return dt.datetime.fromtimestamp(created)

_timestamp = coronavirus['created_utc'].apply(get_date)
coronavirus = coronavirus.assign(timestamp = _timestamp)


# In[ ]:


# if arrays are unequal lengths
# coronavirus = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in posts_data_cv.items()]))


# In[11]:


coronavirus.shape


# In[13]:


#evaluate r/CoronavirusUS data frame
coronavirus.head()


# In[14]:


# Check data types
coronavirus.dtypes


# In[16]:


# # export to csv
# filetime = time.strftime("%y%m%d_%H%M%S", time.localtime())
# coronavirus.to_csv('/Users/ericrodriguez/Documents/Projects/project 3 - nlp reddit/data/{}_{}.csv'.format(subreddit, filetime), index=False)

