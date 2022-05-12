#Mental Health NLP app

import pandas as pd
import pickle
import streamlit as st
from PIL import Image

st.set_page_config(
initial_sidebar_state='expanded'
)

st.title('Mental Health Screening Tool')

st.write('Use the sidebar to select a page to view')

page = st.sidebar.selectbox(
'Page',
('About', 'Words!', 'Mental Health Prediction')
)

if page == 'About'
    st.subheader('About This Project')
    st.write('''
    This is a Streamlit app that showcases my classification model.
    The model utilizes natural language processing and logistic regression to predict if a reddit post came from either r/mentalhealth or r/CoronavirusUS.
    A model that can process text data and passively identify individuals who may require mental health care could be a valuable screening tool in order to connect these individuals with the care that they need, especially in the era of COVID.
    This tool demonstrates my models predictive capabilities which is a first step in creating this type of screening tool.

    If you enjoy this project, visit my portfolio or get in touch with me via LinkedIn!
    -Portfolio: https://rodriguezneric.github.io
    -LinkedIn: https://www.linkedin.com/in/eric-n-rodriguez/)

    '''
    )

elif page == 'Words!':
    st.subheader('Visualizing the most common words in each subreddit')
    st.write('''
    Below you will find word clouds that visualize the most frequently appearing words in the data pulled from each subreddit.

    '''
    )

    wc_1 = Image.open('../images/mentalhealth_wc.png')
    st.image(wc_1, caption='Most Common Words in r/mentalhealth')
    st.write('''

    ''')

    wc_2 = Image.open('../images/covid_wc.png')
    st.image(wc_2, caption='Most Common Words in r/CoronavirusUS')
    st.write('''

    ''')

    wc_3 = Image.open('../images/shared_word_wc.png')
    st.image(wc_3, caption='Most Common Words in Both Subreddits')
    st.write('''

    ''')

elif page == 'Mental Health Prediction':
    st.subheader('Could this post indicate the author is in some form of mental health distress?')
    st.write('''
    Enter some text from a post to predict if the author of the post is in need of mental health services.

    You can enter up to 500 charachters!
    ''')

    st.write('''

    ''')

    # Pickle path
    with open('..data/production_model.pkl', 'rb') as pickle_in:
        model = pickle.load(pickle_in)

    # Text input
    your_text = st.text_input(
    label='Enter post here:',
    value="Please help, I'm in distress...",
    max_chars=500
    )

    # Prediction
    predicted_subreddit = model.predict([your_text])[0]

    # Labels
    my_label = 'None'
    my_asset = 'None'
    if predicted_subreddit == 'CoronavirusUS':
        my_label = 'r/CoronavirusUS or other subreddits'
        my_asset = "not need mental health services, but you should still check on them to see if they're ok because COVID sucks too!"
    elif predicted_subreddit == 'mentalhealth':
        my_label = 'r/mentalhealth'
        my_asset = 'be in need of mental health services'

    # Results
    st.write('''

    ''')
    st.subheader('Results:')
    st.write(f'What you entered resembles text you may find on {my_label}. The author of the text you entered may {my_asset}.')

# Referenced: https://git.generalassemb.ly/DSIR-Lancelot/streamlit_lesson/blob/master/solution-code/app.py
