{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true,
    "tags": []
   },
   "source": [
    "# Project 3: Reddit Scraping & NLP\n",
    "\n",
    "## Part 2 - Data Cleaning and EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# General imports\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pickle\n",
    "\n",
    "# For Natural Language Processing\n",
    "import regex as re\n",
    "import unidecode\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "from nltk.sentiment.vader import SentimentIntensityAnalyzer\n",
    "from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer\n",
    "from nltk.corpus import stopwords\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Read in and Explore Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Read in both subreddit datasets\n",
    "mental_df = pd.read_csv('../data/mentalhealth_211001_171956.csv', encoding = 'utf-8')\n",
    "covid_df = pd.read_csv('../data/CoronavirusUS_211001_172844.csv', encoding = 'utf-8')\n",
    "\n",
    "# Combine into single dataframe\n",
    "df = pd.concat([mental_df, covid_df])\n",
    "df = df.reset_index(drop=True)\n",
    "\n",
    "# Create target column where mentalhealth == 1, CoronavirusUS == 0\n",
    "df['is_mental'] = df['subreddit'].map(lambda t: 1 if t == 'mentalhealth' else 0)"
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
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1611141801</td>\n",
       "      <td>https://www.reddit.com/r/mentalhealth/comments...</td>\n",
       "      <td>https://www.reddit.com/r/mentalhealth/comments...</td>\n",
       "      <td>l17bmq</td>\n",
       "      <td>2</td>\n",
       "      <td>Train Your Brain</td>\n",
       "      <td>[https://psynaesthesia.blogspot.com/2021/01/TY...</td>\n",
       "      <td>mentalhealth</td>\n",
       "      <td>2021-01-20 06:23:21</td>\n",
       "      <td>1</td>\n",
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
       "3  [https://psynaesthesia.blogspot.com/2021/01/TY...  mentalhealth   \n",
       "4  Hello. \\nI usually talk to myself (and Im pret...  mentalhealth   \n",
       "\n",
       "             timestamp  is_mental  \n",
       "0  2021-01-20 06:04:49          1  \n",
       "1  2021-01-20 06:05:53          1  \n",
       "2  2021-01-20 06:18:43          1  \n",
       "3  2021-01-20 06:23:21          1  \n",
       "4  2021-01-20 06:55:53          1  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# View dataframe\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3999, 10)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# View shape\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "created_utc      int64\n",
       "url             object\n",
       "full_link       object\n",
       "id              object\n",
       "num_comments     int64\n",
       "title           object\n",
       "selftext        object\n",
       "subreddit       object\n",
       "timestamp       object\n",
       "is_mental        int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check datatypes \n",
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "created_utc        0\n",
       "url                0\n",
       "full_link          0\n",
       "id                 0\n",
       "num_comments       0\n",
       "title              0\n",
       "selftext        1244\n",
       "subreddit          0\n",
       "timestamp          0\n",
       "is_mental          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check for nulls\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There is 1244 null cells in the \"selftext\" column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "is_mental\n",
       "0    1226\n",
       "1      18\n",
       "Name: selftext, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check the number of selftext nulls by subreddit group\n",
    "df['selftext'].isnull().groupby(df['is_mental']).sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It is clear here that there are more missing selftext values in the r/CoronavirusUS subreddit than in the r/mentalhealth subreddit."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     Has anyone else been completely unable to moni...\n",
       "1     Is it normal to cry when you see someone you love\n",
       "2                                 Somethings gotta give\n",
       "3                                      Train Your Brain\n",
       "4                                         Rage attacks?\n",
       "5                                     I have a Question\n",
       "6     Taking off of work once a month because of maj...\n",
       "7                               I can't do this anymore\n",
       "8     I have an irrational fear of a \"trope\" in fict...\n",
       "9     My old friends from school are anxiety trigger...\n",
       "10    Does this count as an ED? If so, is there a pr...\n",
       "11                   Why dont I feel sad after a death.\n",
       "12    if i no longer live in a toxic environment, wh...\n",
       "13                                         Hopelessness\n",
       "14          What are the best things for mental health?\n",
       "15          Complex PTSD and in-patient hospitalization\n",
       "16                               Am I Really Depressed?\n",
       "17                                Am I self-sabotaging?\n",
       "18    I hate how I am but won't do anything about it...\n",
       "19    My partner's depression is worsening and I fee...\n",
       "Name: title, dtype: object"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# View titles from r/mentalhealth\n",
    "df.loc[df['is_mental'] ==1, 'title'].head(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2000    Doubling Up Masks Creates ‘Obstacle Course’ Fo...\n",
       "2001                                    Covid 19 sweating\n",
       "2002    COVID-19 2021 - How to stay safe from COVID-19...\n",
       "2003        64% of Doomers Don’t Believe Fauci Said This:\n",
       "2004    Nearly 12,000 doses of Moderna COVID-19 vaccin...\n",
       "2005    Whitmer: It will take 2 years to get 70% of Mi...\n",
       "2006                                          Frustrated!\n",
       "2007    Now, within his first 100 days in office, Pres...\n",
       "2008    [OC] Top Countries by Number of Covid-19 Vacci...\n",
       "2009    I don’t think I’ll ever be able to forgive people\n",
       "2010    Will the vaccine be less effective in stopping...\n",
       "2011                         Contact traced for no reason\n",
       "2012    400,000 Americans have died of COVID-19. As a ...\n",
       "2013    Amazon offers to help Biden administration wit...\n",
       "2014    About 4.2% of the US population have received ...\n",
       "2015    COVID19 can affect people of all ages! [Research]\n",
       "2016    Need a COVID test? Four South Florida malls wi...\n",
       "2017    Can someone help me understand why more than 4...\n",
       "2018    Hiding COVID-19: Why People Keep Illness a Sec...\n",
       "2019                                     New CDC Director\n",
       "Name: title, dtype: object"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# View titles from r/CoronavirusUS\n",
    "df.loc[df['is_mental'] == 0, 'title'].head(20)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First 20 examples of r/CoronavirusUS contain language that could be applicable to mental health issues\n",
    "\n",
    "Both subreddits contain examples of unnecessary capital lettering"
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
       "AIs from AI Dungeon 2 to sexy to funny and one based wholly on Reddit!    9\n",
       "I need help                                                               6\n",
       "Antihistamines may protect against the coronavirus                        4\n",
       "What is wrong with me?                                                    3\n",
       "I have a question                                                         3\n",
       "                                                                         ..\n",
       "I feel like I’m literally forgetting myself                               1\n",
       "How do I motivate myself to do things?                                    1\n",
       "Feeling worthless all the time                                            1\n",
       "My boyfriend is being mean to me and blaming it on \"his other self\".      1\n",
       "COVID-19 vaccines work well against California variant, scientists say    1\n",
       "Name: title, Length: 3931, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check for duplicate and empty non-NaN titles\n",
    "df.title.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "scrolled": true,
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[removed]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   196\n",
       "[deleted]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    53\n",
       ".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             4\n",
       "please fill out my questionnaire on mental health for my college course\\n\\n[https://forms.gle/Qso1v8QsWmRroV9h8](https://forms.gle/Qso1v8QsWmRroV9h8)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         3\n",
       "Hello, I've been struggling with my mental health and I think I might be suffering from bipolar disorder and I wanted to know someone's opinion on if I may have this. Some days I feel very productive and like I can accomplish anything and others I find it hard to focus on any task. Sometimes I'm very happy and others I'm very sad or very mad for no reason and yell at others in my family for no reason. Sometimes I feel very empty other times I'm so hyper and can talk a million miles a minute and can't control it. Some days I feel extremely overconfident and others feel worthless and a failure. I cant focus some days and ill do anything but what I should be doing. I constantly have a racing mind with obsessive thoughts that I can get away from, I'm constantly paranoid about something. Sometimes I'm in the heat of the moment and do things I usually wouldn't and end up regretting it (unprotect sex). I often feel like I am losing control of having fluctuating perspectives of those around me and feel like they are trying to hurt me. These symptoms get in the way of my daily tasks and cause me a lot of distress and consequences. Every bipolar test I've taken online indicates i have symptoms of BPD but I understand that this is not a real diagnosis, therefore I have an appointment with a psychiatrist. I just want someone's opinion on this to see if having BPD is a possibility.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               2\n",
       "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ... \n",
       "I’ve been thinking about this lately. As soon as 2019 arrived, I turned into a very different person than who I was. I became less empathetic, less conscientious, more risk prone, and even started to sleep around, tried drugs and became a heavy drinker too, though I did not build a dependence on it... thankfully I have some good genes that keep me from building addictions. \\n\\nI also got involved in more unethical behaviors and lost many friends. Also, my rage is uncontrollable when I am angry. I’m very clam all the time but if someone crosses me, I feel this intense uncontrollable rage and it isn’t pretty. \\n\\nAre any of you out there in a similar position regarding a change in personality? I’d like to know what your diagnosis is or thoughts about my situation. \\n\\nThanks!\\n\\nP.S I am not looking for a a diagnosis for myself. I’m solely curious about you guys. I’ve already been diagnosed with something.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         1\n",
       "Okay so I know this is a weird one and kind of trivial, but I'm starting to believe I have agoraphobia and usually I'll stay in bed for hours at a time and don't want to get out, the reason why is because I feel safe (and this feels a little pathetic for me to admit) in my bed and under my covers, but whenever it's night time it feels like the same excitement you'd get whenever you were a kid and had Christmas I can throw my covers off and it feels good, like today's a fresh day and it won't be that bad.\\n I know it's probably just a placebo effect, but for some reason the night just makes me feel more comfortable and more at ease, maybe it's because I feel like I can't be seen all that much at night again I know this sounds kind of weird, but I'm just curious if this makes any sense or if anyone can relate?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           1\n",
       "Hey guys,\\nI have struggled with mental illness for a while. PMDD, OCD, Anxiety,ADHD and depression. However all of that I want to feel stable. I don’t want to spiral out and die when something hurts or when anxiety hits. I need help knowing how to be stable mentally. I take medication it helps but i want something I can do to make sure I can continue living a life without feeling like it’s better to die so that I don’t have to feel like this. I want normality in my mind. I want stability. Any advice or help?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            1\n",
       "When this whole virus had school turn into online school, I thought it was a blessing. \\nI had more time to workout, play games, be with friends, etc. In the span from March to now, I've tried Runescape, gotten 9 days worth of gameplay over summer vacation. Even made a friend, who I still hope is ok. I've gotten a puppy and kitten. Seen improvement in my weightlifting. I've been with my parents more as well. \\nThat's just the summary of it, but then there's the bad that's happened. \\nThat friend I've made in Runescape, he told me his girlfriend cheated on him, had a mental breakdown, and we haven't talked much since. Holiday's feel depressing now, since before lots of family would show up to see eachother, where as Christmas was so much fewer people. When I said I got to be with friends more, I meant online through games. Lots of them I barely even talk to now. I've had a history of having mood swings here and there, but over quarantine I feel like they've gotten worse. I miss the social aspect of being at school, instead of being in online classes, where half the time the teacher gets disconnected because they have trash internet. \\nThen I thought, you know what, things have been pretty bad and good at the same time, let's try and make things better so I can forget about the bad.\\nI ranted to my crush whom knows that I like, about how I feel, and they actually liked that I told them about it, and to be honest, I was expecting them to react a lot less, since I was doing it to mainly get it off my chest, and even told them so.\\nThis is all over Instagram btw, remember, we can't see eachother because of the virus. They asked me why I decided to text them, and I responded honestly by saying that since I've got a lot more time on my hands, I ended up thinking of them from time to time. \\nI think that was a bad response, because I ended up getting left on read. I know this isn't a subreddit on relationship advice, and that's not what I'm asking here, let me continue. \\nIt's been two weeks since that happened, and ever since, I feel kind of empty, been sleeping more throughout the day, falling behind on school, and just not seeing a purpose in life, even though I have so much of, well, everything. \\nAm I being selfish here, or is it normal to feel this way. The more I think about it the more I feel like this feeling has just been bottled up, and me texting that person was the last push I needed for me to be in this mood. I'm of course not blaming them however, I actually kind of blame myself for texting them, I should've just moved on. \\nIf you're worried about me hurting anyone or myself, don't worry, I'm just sad, not stupid. I say that because as I read what I've wrote, people might think that, so I'm just giving reassurance. Don't be afraid to tell me I'm overreacting or just being a little edgy teenager, I want to know how you see this, in honest truth.      1\n",
       "So far, the vaccines are said to be effective against variants of COVID-19 that have risen.\\n\\nBut since the population won’t all be vaccinated at once, could the virus slowly mutate to overcome the obstacles that the vaccine presents?\\n\\nSerious answers only, not here for conspiracies. Have other viruses exhibited this type of behavior in the past?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               1\n",
       "Name: selftext, Length: 2489, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check for duplicate and empty non-NaN selftext values\n",
    "df.selftext.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The selftext column has values that need to be converted from \"[removed]\", \"[deleted]\", \"?\", and \".\" to np.NaN\n",
    "\n",
    "There are also links to other articles which starts with [http], which will need to be cleaned"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False    3931\n",
       "True       68\n",
       "Name: title, dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check for duplicate values\n",
    "df['title'].duplicated().value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There are 68 duplicate titles"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "is_mental\n",
       "0    32\n",
       "1    36\n",
       "Name: title, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check the number of duplicates by subreddit group\n",
    "df['title'].duplicated().groupby(df['is_mental']).sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "mentalhealth     0.500125\n",
       "CoronavirusUS    0.499875\n",
       "Name: subreddit, dtype: float64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Verify distrubtion\n",
    "df.subreddit.value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Getting rid of rows with duplicate titles doesn't significantly change the distriubtion of subreddit posts "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "scrolled": true,
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Replace '[removed]' '[deleted]', '.', '?' with NaN in the 'selftext' column\n",
    "df['selftext'] = df['selftext'].replace(\n",
    "    {'[removed]': np.nan, '[deleted]': np.nan,'.': np.nan, '?': np.nan})\n",
    "\n",
    "# Drop duplicates based on title column\n",
    "df = df.drop_duplicates(subset='title')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove hyperlinks from title and selftext h/t Philip DiSarro on stackoverflow\n",
    "df['title'] = df['title'].replace(r'http\\S+', '', regex=True).replace(r'www\\S+', '', regex=True)\n",
    "df['selftext'] = df['selftext'].replace(r'http\\S+', '', regex=True).replace(r'www\\S+', '', regex=True)\n",
    "\n",
    "# Replace cells that only contain \"[\" and \"\" and \"\\n\" with NaN in the 'selftext' column \n",
    "# which are a result of splitting and removing hyperlinks\n",
    "df['selftext'] = df['selftext'].replace({'[': np.nan, '\\n': np.nan})\n",
    "df['selftext'] = df['selftext'].replace(r'^\\s*$', np.nan, regex=True)\n",
    "\n",
    "# Replace \"&amp;\" with \"and\" from title and selftext\n",
    "df['title'] = df['title'].replace('&amp;', 'and', regex=True)\n",
    "df['selftext'] = df['selftext'].replace('&amp;', 'and', regex=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop inappropriate sequence of rows in the title column\n",
    "# Drop like this to ommit innappropriate values\n",
    "df = df[df.title.str.len() > 2]\n",
    "\n",
    "# Reset index again\n",
    "df.reset_index(inplace = True, drop = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
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
       "             timestamp  is_mental  title_char_length  title_word_count  \n",
       "0  2021-01-20 06:04:49          1                 70                11  \n",
       "1  2021-01-20 06:05:53          1                 49                11  \n",
       "2  2021-01-20 06:18:43          1                 21                 3  \n",
       "3  2021-01-20 06:23:21          1                 16                 3  \n",
       "4  2021-01-20 06:55:53          1                 13                 2  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create a column for title character length\n",
    "df['title_char_length'] = df['title'].apply(len)\n",
    "\n",
    "# Create a column for title word count\n",
    "df['title_word_count'] = df['title'].map(lambda x: len(x.split()))\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** Both subreddits are skewed to the right, with r/mentalhealth having a larger distribution of character lengths on the shorter side than r/CoronavirusUS."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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IJMkSVuM+fPns2jRIh599FGuv/56XvSiF/3M8gsvvJC5c+c+8TMVDleChywlSdIW5LDDDuOuu+7iggsu4Pjjj3/S8te//vWcffbZT7w/5phjJrG6iRnIJEnSFuXVr34173rXu1i6dCkrV65sXc5ADGSSJGmL8pu/+Zs8/elP57nPfS5Lly5tXc5APIdMkiRtUWbNmsXb3/721mVskFRV6xo22rx582qqnIwnSdJ0dvPNN/Oc5zyndRlTxnjjkeSaqpo33vrOkEmSJDVmIJMkSWrMQCZJktSYV1luQVo8pkKSJG06Z8gkSZIaM5BJkiQ1ZiCTJElbjP/4j/9g/vz5HHDAARx88MEcf/zx3HbbbSxbtozjjjuOAw88kDlz5vDHf/zHVBVLly7lqKOO+pk+Vq9ezR577MGKFSt405vexBe+8AWg95ilgw46iMMOO4xnP/vZnHbaafzgBz8YSt2eQyZJkobvsg8Pt79jF653larixBNPZMGCBSxatAiAa6+9lvvuu483velNfOITn+BlL3sZP/rRj3jta1/Lxz/+cX7rt36L5cuXc9dddzF79mwALr30Ug499FD23HPPJ33G5z73OebNm8ePf/xjFi5cyAknnMA3vvGNTf56zpBJkqQtwmWXXcY222zD2972tifa5s6dy2233cbRRx/Ny172MgC23357zj77bM4880ye8pSncNJJJ3HhhRc+sc2iRYs4+eST1/lZT33qU/nIRz7C3XffzXXXXbfJtRvIJEnSFuHGG2/khS984ZPaly1b9qT2Aw44gFWrVvHggw9y8sknPzGj9thjj/GVr3yF1772tev9vK222ornPe953HLLLZtcu4csJUnSFq2qSDLusiQcfvjhrFq1iltvvZWbb76ZI488kl122WXgvofBQCZJkrYIhxxyyBMn4K/d/s1vfvNn2u644w523HFHdtppJwDmz5/PokWLuPnmm9d7uHLMmjVruOGGG4byDE8PWUqSpC3Ccccdx2OPPca55577RNtVV13FnDlzuPzyy7n00ksBeOSRR/jd3/1d3v3udz+x3sknn8xnP/tZvv71r/PqV796vZ/1+OOPs3DhQvbZZx8OO+ywTa7dQCZJkrYISbjoootYsmQJBxxwAIcccginn346e+21FxdffDEf+tCHOOigg3juc5/L4YcfzmmnnfbEtgcffDDbb789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\n",
      "text/plain": [
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Histogram of title character length\n",
    "plt.figure(figsize=(10,8))\n",
    "plt.hist(x=df[df['is_mental']==1]['title_char_length'],\n",
    "         bins=25, alpha=0.5, label='MH')\n",
    "plt.hist(x=df[df['is_mental']==0]['title_char_length'],\n",
    "         bins=25, alpha=0.5, label='COVID')\n",
    "plt.title('Distribution of Title Character Length')\n",
    "plt.xlabel('Character Length')\n",
    "plt.ylabel('Frequency')\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** The title character length of both subreddits are skewed to the right, following a similair distribution."
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
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Histogram of title word count\n",
    "plt.figure(figsize=(10,8))\n",
    "plt.hist(x=df[df['is_mental']==1]['title_word_count'],\n",
    "         bins=25, alpha=0.5, label='MH')\n",
    "plt.hist(x=df[df['is_mental']==0]['title_word_count'],\n",
    "         bins=25, alpha=0.5, label='COVID')\n",
    "plt.title('Distribution of Title Word Count')\n",
    "plt.xlabel('Word Count')\n",
    "plt.ylabel('Frequency')\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** The title world count of both subreddits are skewed to the right, following a similair distribution."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Out of curiousity, I'm going to check the sentiment of my posts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
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
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   created_utc                                                url  \\\n",
       "0   1611140689  https://www.reddit.com/r/mentalhealth/comments...   \n",
       "1   1611140753  https://www.reddit.com/r/mentalhealth/comments...   \n",
       "2   1611141523  https://www.reddit.com/r/mentalhealth/comments...   \n",
       "\n",
       "                                           full_link      id  num_comments  \\\n",
       "0  https://www.reddit.com/r/mentalhealth/comments...  l172x7             3   \n",
       "1  https://www.reddit.com/r/mentalhealth/comments...  l173gi             7   \n",
       "2  https://www.reddit.com/r/mentalhealth/comments...  l179ft             2   \n",
       "\n",
       "                                               title  \\\n",
       "0  Has anyone else been completely unable to moni...   \n",
       "1  Is it normal to cry when you see someone you love   \n",
       "2                              Somethings gotta give   \n",
       "\n",
       "                                            selftext     subreddit  \\\n",
       "0  I had a psychotic break which seemed to come o...  mentalhealth   \n",
       "1  I know this sounds weird but i guess let me ex...  mentalhealth   \n",
       "2  Been living with my mom who has bad mental iss...  mentalhealth   \n",
       "\n",
       "             timestamp  is_mental  title_char_length  title_word_count  \\\n",
       "0  2021-01-20 06:04:49          1                 70                11   \n",
       "1  2021-01-20 06:05:53          1                 49                11   \n",
       "2  2021-01-20 06:18:43          1                 21                 3   \n",
       "\n",
       "   sentiment_comp  sentiment_neg  sentiment_neu  sentiment_pos  \n",
       "0          0.0000           0.00          1.000          0.000  \n",
       "1          0.2732           0.19          0.552          0.258  \n",
       "2          0.0000           0.00          1.000          0.000  "
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Instantiate sentiment intesnity analyzer\n",
    "sia = SentimentIntensityAnalyzer()\n",
    "\n",
    "# create a column for the negative, neutral, positive, and compound analysis scores:\n",
    "df['sentiment_comp'] = df['title'].map(lambda x: sia.polarity_scores(x)['compound'])\n",
    "df['sentiment_neg'] = df['title'].map(lambda x: sia.polarity_scores(x)['neg'])\n",
    "df['sentiment_neu'] = df['title'].map(lambda x: sia.polarity_scores(x)['neu'])\n",
    "df['sentiment_pos'] = df['title'].map(lambda x: sia.polarity_scores(x)['pos'])\n",
    "\n",
    "# Check code execution\n",
    "df.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sentiment_comp   -0.109791\n",
      "dtype: float64\n",
      "sentiment_neg    0.196247\n",
      "dtype: float64\n",
      "sentiment_neu    0.690436\n",
      "dtype: float64\n",
      "sentiment_pos    0.11332\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# Average sentiment of r/mentalhealth\n",
    "print(df[df['is_mental']==1][['sentiment_comp']].mean())\n",
    "print(df[df['is_mental']==1][['sentiment_neg']].mean())\n",
    "print(df[df['is_mental']==1][['sentiment_neu']].mean())\n",
    "print(df[df['is_mental']==1][['sentiment_pos']].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sentiment_comp    0.006475\n",
      "dtype: float64\n",
      "sentiment_neg    0.072777\n",
      "dtype: float64\n",
      "sentiment_neu    0.853274\n",
      "dtype: float64\n",
      "sentiment_pos    0.073945\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# Average sentiment of r/CoronavirusUS\n",
    "print(df[df['is_mental']==0][['sentiment_comp']].mean())\n",
    "print(df[df['is_mental']==0][['sentiment_neg']].mean())\n",
    "print(df[df['is_mental']==0][['sentiment_neu']].mean())\n",
    "print(df[df['is_mental']==0][['sentiment_pos']].mean())"
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
       "<Figure size 576x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Boxplot for sentiment compound scores\n",
    "data_1 = df[df['is_mental']==1]['sentiment_comp']\n",
    "data_2 = df[df['is_mental']==0]['sentiment_comp']\n",
    "data = [data_1, data_2]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8,10))\n",
    "bp = ax.boxplot(data, widths = .4, patch_artist=True)\n",
    "colors = ['lightblue', 'peachpuff']\n",
    "for patch, color in zip(bp['boxes'], colors):\n",
    "    patch.set_facecolor(color)\n",
    "for median in bp['medians']:\n",
    "    median.set(color='red')\n",
    "for whisker in bp['whiskers']:\n",
    "    whisker.set(color='peru', linewidth=2, linestyle=\":\")\n",
    "for cap in bp['caps']:\n",
    "    cap.set(color='saddlebrown')\n",
    "for flier in bp['fliers']:\n",
    "    flier.set(color='azure')\n",
    "ax.set_xticklabels(['MH', 'COVID'])\n",
    "plt.ylabel(\"Compound Score\")\n",
    "plt.xlabel(\"Subreddits\")\n",
    "plt.title(\"Boxplot of Sentiment Compound Score\")\n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true,
    "tags": []
   },
   "source": [
    "**Interpretation:** The mentalhealth subreddit is distributed more towards a negative compound score, whereas the CoronavirusUS subreddit is distributed within a more neutral-positive coumpound score. Both median values are close to a coumpound sentiment value of zero."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
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
       "      <th>word</th>\n",
       "      <th>word_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>help</td>\n",
       "      <td>156</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>mental</td>\n",
       "      <td>153</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>feel</td>\n",
       "      <td>151</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>i'm</td>\n",
       "      <td>122</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>like</td>\n",
       "      <td>112</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     word  word_count\n",
       "0    help         156\n",
       "1  mental         153\n",
       "2    feel         151\n",
       "3     i'm         122\n",
       "4    like         112"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Count the most commonly used words for r/mentalhealth\n",
    "\n",
    "# Define X\n",
    "X = df[df['is_mental']==1]['title']\n",
    "\n",
    "# Instantiate a CV object\n",
    "cv = CountVectorizer(stop_words='english', analyzer = 'word', token_pattern = r'[\\w{2,}\\d*]\\S*\\w+')\n",
    "\n",
    "# Fit and transform the CV data on posts\n",
    "X_cv = cv.fit_transform(X)\n",
    "\n",
    "# Convert to a dataframe\n",
    "cv_df = pd.DataFrame(X_cv.todense(),columns=cv.get_feature_names_out())\n",
    "\n",
    "# Create a mh dataframe containing the 100 most common words and word count\n",
    "mh_cv_df = pd.DataFrame(data=cv_df.sum().sort_values(ascending=False).head(100),\n",
    "                        columns=['word_count'])\n",
    "\n",
    "# Change index to a column\n",
    "mh_cv_df.reset_index(inplace=True)\n",
    "mh_cv_df = mh_cv_df.rename(columns={'index':'word'})\n",
    "\n",
    "# Save wsb_cv_df to csv\n",
    "mh_cv_df.to_csv('../data/mh_words.csv', index=False)\n",
    "\n",
    "# Check code execution\n",
    "mh_cv_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Bar chart of the 20 most commonly seen words in r/mentalhealth\n",
    "plt.figure(figsize=(12,8))\n",
    "plt.barh(y=mh_cv_df['word'].head(20), width=mh_cv_df['word_count'].head(20))\n",
    "plt.title('20 Most Common Words in r/mentalhealth')\n",
    "plt.xlabel('Word Count')\n",
    "plt.ylabel('Word');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
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
       "      <th>word</th>\n",
       "      <th>word_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>covid-19</td>\n",
       "      <td>454</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>vaccine</td>\n",
       "      <td>442</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>covid</td>\n",
       "      <td>304</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>coronavirus</td>\n",
       "      <td>153</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>new</td>\n",
       "      <td>134</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          word  word_count\n",
       "0     covid-19         454\n",
       "1      vaccine         442\n",
       "2        covid         304\n",
       "3  coronavirus         153\n",
       "4          new         134"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Count the most commonly used words for r/CoronavirusUS\n",
    "\n",
    "# Define X\n",
    "X = df[df['is_mental']==0]['title']\n",
    "\n",
    "# Instantiate a CV object\n",
    "cv = CountVectorizer(stop_words='english', analyzer = 'word', token_pattern = r'[\\w{2,}\\d*]\\S*\\w+')\n",
    "\n",
    "# Fit and transform the CV data on posts\n",
    "X_cv=cv.fit_transform(X)\n",
    "\n",
    "# Convert to a dataframe\n",
    "cv_df = pd.DataFrame(X_cv.todense(),columns=cv.get_feature_names_out())\n",
    "\n",
    "# Create a mh dataframe containing the 100 most common words and word count\n",
    "covid_cv_df = pd.DataFrame(data=cv_df.sum().sort_values(ascending=False).head(100),\n",
    "                        columns=['word_count'])\n",
    "\n",
    "# Change index to a column\n",
    "covid_cv_df.reset_index(inplace=True)\n",
    "covid_cv_df = covid_cv_df.rename(columns={'index':'word'})\n",
    "\n",
    "# Save wsb_cv_df to csv\n",
    "covid_cv_df.to_csv('../data/covid_words.csv', index=False)\n",
    "\n",
    "# Check code execution\n",
    "covid_cv_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Bar chart of the 20 most commonly seen words in r/CoronavirusUS\n",
    "plt.figure(figsize=(12,8))\n",
    "plt.barh(y=covid_cv_df['word'].head(20), width=covid_cv_df['word_count'].head(20))\n",
    "plt.title('20 Most Common Words in r/CoronavirusUS')\n",
    "plt.xlabel('Word Count')\n",
    "plt.ylabel('Word');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "22\n",
      "['help', 'like', 'health', 'know', 'i’m', 'just', 'need', 'does', 'don’t', 'think', 'people', 'time', 'going', 'getting', 'make', 'today', 'new', 'right', 'got', 'long', 'best', 'say']\n"
     ]
    }
   ],
   "source": [
    "# Find matches for the 100 most common words in both subreddits\n",
    "common_words = []\n",
    "for word in mh_cv_df['word']:\n",
    "    if word in list(covid_cv_df['word']):\n",
    "        common_words.append(word)\n",
    "print(len(common_words))\n",
    "print(common_words)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** Of the 100 most commonly found words in the mentalhealth and CoronavirusUS subreddits, 22 words are shared between the two. This indicates that there is overlapping terminology in both subreddits, which may limit the accuracy of any predictive models built to distinguish the two subreddits."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
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
       "      <th>shared_word</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>help</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>like</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>health</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>know</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>i’m</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  shared_word\n",
       "0        help\n",
       "1        like\n",
       "2      health\n",
       "3        know\n",
       "4         i’m"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "common_words_df = pd.DataFrame(data = common_words, columns = ['shared_word'])\n",
    "common_words_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a dictionary for r/mentalhealth shared word count\n",
    "mh_shared_word_count = {}\n",
    "for word in common_words_df['shared_word']:\n",
    "    index = 0\n",
    "    for i in mh_cv_df['word']:\n",
    "        if word == i:\n",
    "            mh_shared_word_count[word] = mh_cv_df['word_count'][index]\n",
    "        index += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
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
       "      <th>shared_word</th>\n",
       "      <th>mh_word_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>help</td>\n",
       "      <td>156</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>like</td>\n",
       "      <td>112</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>health</td>\n",
       "      <td>111</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>know</td>\n",
       "      <td>94</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>i’m</td>\n",
       "      <td>93</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  shared_word  mh_word_count\n",
       "0        help            156\n",
       "1        like            112\n",
       "2      health            111\n",
       "3        know             94\n",
       "4         i’m             93"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create a new column with r/mentalhealth word count\n",
    "common_words_df['mh_word_count'] = common_words_df['shared_word'].map(mh_shared_word_count)\n",
    "\n",
    "common_words_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a dictionary for r/CoronavirusUS shared word count\n",
    "covid_shared_word_count = {}\n",
    "for word in common_words_df['shared_word']:\n",
    "    index = 0\n",
    "    for i in covid_cv_df['word']:\n",
    "        if word == i:\n",
    "            covid_shared_word_count[word] = covid_cv_df['word_count'][index]\n",
    "        index += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
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
       "      <th>shared_word</th>\n",
       "      <th>mh_word_count</th>\n",
       "      <th>covid_word_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>help</td>\n",
       "      <td>156</td>\n",
       "      <td>41</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>like</td>\n",
       "      <td>112</td>\n",
       "      <td>24</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>health</td>\n",
       "      <td>111</td>\n",
       "      <td>66</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>know</td>\n",
       "      <td>94</td>\n",
       "      <td>25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>i’m</td>\n",
       "      <td>93</td>\n",
       "      <td>23</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  shared_word  mh_word_count  covid_word_count\n",
       "0        help            156                41\n",
       "1        like            112                24\n",
       "2      health            111                66\n",
       "3        know             94                25\n",
       "4         i’m             93                23"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create a new column with r/mentalhealth word count\n",
    "common_words_df['covid_word_count'] = common_words_df['shared_word'].map(covid_shared_word_count)\n",
    "\n",
    "common_words_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "22"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check length of dataframe to make sure it equals 22\n",
    "len(common_words_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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O917032QPCVdK/Kv0m3/ybwn3fmJ307yrdyyxfl26bbdZem6Ff5+uv15Ii3dFSl/3E//h0me2HcBns7fpus6ek2ST+XWrd63XFFrlybZP905nevTfWj+q9z8XvisdBe++UmS1+fW3bPH+1G6D5qXpbvQyItaa+f3416V7nX4teq66X4+XYtk+mmOTXJhv4+OdVU9OV03um+M3N86t/w5mKmWe1q6C7u8s6/rgnTdzAerqsotu0nOxaS1z8B/Jtmj317/1bfo/u90X4xcke78zq/MpajW2sfSXcTmuL6uc5KMXW31s+lCwvfStUL9KpN0o+xfS29K8pW+zgfPso6hj2m2++94T8hIsGytXZ/uXPjz03XN/lm6fXO7dF92/TrdzwM9Pt3r911JnjfyGriF2U4/zkyehxPSXZzsitbat2ewzFF/ku5YcFW6C6yNdvee6r3mz9Jt77PSnRN8aLrj11hPj+vSPac/SveYX5zuIl0XzrI+2CCMXUkM2MBV9/MWFyXZdIpWJoBZ63tNvLO1NuvfL2T5q6q7pfvCaYdpupACTEqLJQAwE69f6gJYMNskeZlQCQyhxRJuI7RYAgCwVARLAAAABtEVFgAAgEEESwAAAAZZsdQFDLHddtu1VatWLXUZAAAAt3mnn376j1trKycat0EHy1WrVuW00yb7+TsAAADmS1X9YLJxusICAAAwiGAJAADAIIIlAAAAg2zQ51gCAADMh9/85jdZt25dfvWrXy11KUtus802y0477ZRNN910xvMIlgAAwEZv3bp12XrrrbNq1apU1VKXs2Raa7nqqquybt267LrrrjOeT1dYAABgo/erX/0qd7nLXTbqUJkkVZW73OUus265FSwBAACSjT5UjpnLdhAsAQAAloGqynOf+9yb7t9www1ZuXJlnvSkJyVJjjzyyBx66KG3mGfNmjU57bTTFrXOiSzYOZZV9d4kT0pyZWttr5Hhf57k0CQ3JPlUa+2V/fDDkrwwyY1JXtJa++xC1QYAADCVWju/y2trpp9myy23zDnnnJPrrrsum2++eU488cTsuOOO81vIAlnIFssjkzxudEBV/UGS/ZPs3VrbM8lb++F7JDkgyZ79PO+qqk0WsDYAAIBl5/GPf3w+9alPJUmOPfbYHHjggUtc0cwsWLBsrZ2S5CfjBv9pkje31q7vp7myH75/kuNaa9e31i5KckGSBy5UbQAAAMvRAQcckOOOOy6/+tWvctZZZ+VBD3rQLcZ/8IMfzD777HPT33LoBpss/jmW90zy8Kr6elWdXFX79sN3THLpyHTr+mG3UlWHVNVpVXXa+vXrF7hcAACAxbP33nvn4osvzrHHHpsnPOEJtxr/zGc+M2eeeeZNf6tXr16CKm9tsYPliiR3SvLgJH+V5PjqLjk00WWH2kQLaK0d0Vpb3VpbvXLlyoWrFAAAYAk85SlPySte8YoNphtssoAX75nEuiQfba21JN+oqt8m2a4fvvPIdDsluWyRawMAAFhyL3jBC7LNNtvkvve9b9auXbvU5czIYrdY/leSRyZJVd0zye2T/DjJCUkOqKo7VNWuSXZP8o1Frg0AAGDJ7bTTTnnpS1+61GXMSnWNhwuw4Kpjk6xJ1yJ5RZLXJ3l/kvcm2SfJr5O8orX2hX761yR5QbqfIfmL1tp/T7eO1atXt+VysioAALDhOu+883Kf+9xnqctYNibaHlV1emttwpM6F6wrbGttsg7Bz5lk+jcledNC1QMAAMDCWOyusAAAANzGCJYAAAAMIlgCAAAwiGAJAADAIIIlAAAAgwiWAAAAy8SPfvSjHHDAAdltt92yxx575AlPeEK+973v5dxzz80jH/nI3POe98zuu++eN7zhDWmtZe3atdlvv/1usYwbbrghd7vb3XL55Zfn4IMPzoc//OEkyZo1a3Kve90re++9d+5973vn0EMPzdVXXz0vdS/Yz40AAABssI6p+V3es9q0k7TW8rSnPS0HHXRQjjvuuCTJmWeemSuuuCIHH3xw3v3ud+cxj3lMfvnLX+bpT3963vWud+VP//RPs27dulx88cVZtWpVkuTzn/989tprr2y//fa3WsfRRx+d1atX59e//nUOO+yw7L///jn55JMHPzwtlgAAAMvAF7/4xWy66aZ50YtedNOwffbZJ9/73vfy0Ic+NI95zGOSJFtssUXe+c535s1vfnNud7vb5RnPeEY++MEP3jTPcccdlwMPPHDKdd3+9rfPW97yllxyySX59re/Pbh2LZYLqNbOfp62Zr6rAAAANgTnnHNOHvCAB9xq+Lnnnnur4bvttluuvfba/OxnP8uBBx6YQw45JK961aty/fXX59Of/nTe/va3T7u+TTbZJPe73/1y/vnn5373u9+g2gVLAACAZay1lqqJu+ZWVfbdd99ce+21+e53v5vzzjsvD37wg3OnO91pxsueD4IlAADAMrDnnnvedKGd8cNPOeWUWwy78MILs9VWW2XrrbdOkhxwwAE57rjjct55503bDXbMjTfemLPPPjv3uc99BtfuHEsAAIBl4JGPfGSuv/76vOc977lp2De/+c3svvvu+fKXv5zPf/7zSZLrrrsuL3nJS/LKV77ypukOPPDAfOADH8gXvvCFPOUpT5l2Xb/5zW9y2GGHZeedd87ee+89uHbBEgAAYBmoqnzsYx/LiSeemN122y177rlnDj/88Oywww75+Mc/nje+8Y25173ulfve977Zd999c+ihh9407x577JEtttgij3zkI7PllltOuo5nP/vZ2XvvvbPXXnvlF7/4RT7+8Y/PT+3z1ad2KaxevbqddtppS13GpFy8BwAANgznnXfevHQJva2YaHtU1emttdUTTa/FEgAAgEEESwAAAAYRLAEAABhEsAQAAMj8/abjhm4u20GwBAAANnqbbbZZrrrqqo0+XLbWctVVV2WzzTab1XwrFqgeAACADcZOO+2UdevWZf369UtdypLbbLPNstNOO81qHsESAADY6G266abZddddl7qMDZausAAAAAwiWAIAADCIYAkAAMAggiUAAACDCJYAAAAMIlgCAAAwiGAJAADAIIIlAAAAgwiWAAAADCJYAgAAMIhgCQAAwCCCJQAAAIMIlgAAAAwiWAIAADCIYAkAAMAggiUAAACDCJYAAAAMIlgCAAAwiGAJAADAIIIlAAAAgwiWAAAADCJYAgAAMIhgCQAAwCCCJQAAAIMIlgAAAAwiWAIAADCIYAkAAMAggiUAAACDCJYAAAAMIlgCAAAwiGAJAADAIIIlAAAAgwiWAAAADCJYAgAAMIhgCQAAwCCCJQAAAIMIlgAAAAwiWAIAADCIYAkAAMAggiUAAACDLFiwrKr3VtWVVXXOBONeUVWtqrYbGXZYVV1QVd+tqscuVF0AAADMr4VssTwyyePGD6yqnZP8YZJLRobtkeSAJHv287yrqjZZwNoAAACYJwsWLFtrpyT5yQSj3p7klUnayLD9kxzXWru+tXZRkguSPHChagMAAGD+LOo5llX1lCQ/bK19e9yoHZNcOnJ/XT9somUcUlWnVdVp69evX6BKAQAAmKlFC5ZVtUWS1yR53USjJxjWJhiW1toR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\n",
      "text/plain": [
       "<Figure size 864x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Boxplot of 24 most common words shared between both subreddits\n",
    "data_1 = common_words_df['mh_word_count']\n",
    "data_2 = common_words_df['covid_word_count']\n",
    "data = [data_1, data_2]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12,8))\n",
    "words = common_words_df['shared_word']\n",
    "x = np.arange(len(words))\n",
    "ax = fig.add_axes([0,0,1,1])\n",
    "bar_width = 0.375\n",
    "\n",
    "br1 = np.arange(len(data[0]))\n",
    "br2 = [x + bar_width for x in br1]\n",
    "ax.bar(br1, data[0], color='deepskyblue', width=-bar_width, label='MH', align='edge')\n",
    "ax.bar(br2, data[1], color='orange', width=-bar_width, label='COVID', align='edge')\n",
    "ax.set_xticks(x, words.values)\n",
    "          \n",
    "plt.title('Barplot of Most Common Words Shared between r/mentalhealth and r/CoronavirusUS')\n",
    "plt.xlabel('Shared Words')\n",
    "plt.ylabel('Word Count')\n",
    "plt.legend()\n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation:** Common words between the two subreddits typically aren't very similiair in word count. Many words that have a high word count in the mental health subreddit have a low word count in the coronavirusUS subreddit, and vice versa. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save common_words_df to csv\n",
    "common_words_df.to_csv('../data/shared_words.csv', index = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save cleaned posts as csv file\n",
    "df.to_csv('../data/cleaned_posts.csv', index=False)"
   ]
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
