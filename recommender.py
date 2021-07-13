#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


movies_df=pd.read_csv('tmdb_5000_movies.csv')
credits_df=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


credits_df.rename(columns={'movie_id':'id'},inplace=True)
merged_df=movies_df.merge(credits_df, on='id')


# In[4]:


merged_df[['overview','tagline']]=merged_df[['overview','tagline']].fillna('')


# In[5]:


features=['homepage','production_countries','release_date','runtime','title_y','original_title']
merged_df.drop(features, axis=1, inplace=True)
merged_df=merged_df.rename(columns={'title_x':'title'})


# In[6]:


merged_df['budget']/=1e6


# In[7]:


categories={
    "status":{"Released":0, "Post Production":1,"Rumored":2}
}
merged_df=merged_df.replace(categories)


# In[8]:


r=merged_df['vote_average']
v=merged_df['vote_count']
c=r.mean()
m=v.quantile(.90)
weighted_rating=(r*v + c*m)/(v+m)
merged_df['weighted_rating']=weighted_rating


# In[9]:


features=['genres','keywords','production_companies','cast','crew']
for feature in features:
    merged_df[feature]=merged_df[feature].apply(literal_eval)


# In[10]:


#extracting the director of the movie
def extract_director(crew):
    for i in crew:
        if i['job']=='Director':
            return i['name'];
    return np.nan #Nan if no director


# In[11]:


#extracting top 3 elements from each list
def get_top3(x):
    if isinstance(x,list):
        names=[i['name'] for i in x]
        if len(names)>3:
            return names[:3]
        return names
    return []


# In[12]:


merged_df['director']=merged_df['crew'].apply(extract_director)
features=['genres','keywords','production_companies','cast']
for feature in features:
    merged_df[feature]=merged_df[feature].apply(get_top3)


# In[13]:


directors=merged_df.groupby(by='director')['id'].count().sort_values(ascending=False)
directors=directors.to_frame(name='count')
directors=directors.reset_index()


# In[14]:


def remove_word_spaces(x):
    if isinstance(x,list):
        return [str.lower(i.replace(' ','')) for i in x]
    else: #must come from the director
        if isinstance(x,str):
            return str.lower(x.replace(' ',''))
        return '' #no director


# In[15]:


features=['genres','keywords','production_companies','cast','director']
for feature in features:
    merged_df[feature]=merged_df[feature].apply(remove_word_spaces)


# In[16]:


def create_string(x):
    return (' '.join(x['genres']) + ' ' 
            + ' '.join(x['keywords']) + ' ' 
            + ' '.join(x['production_companies']) + ' '
            + ' '.join(x['cast']) + ' '
            + x['director']
           )
merged_df['word_string']=merged_df.apply(create_string,axis=1)


# In[17]:


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(merged_df['word_string'])


# In[18]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[19]:


merged_df = merged_df.reset_index()
indices = pd.Series(merged_df.index, index=merged_df['title'])


# In[32]:


def get_recommendations(title, cosine_sim=cosine_sim):
    if(title!="Select an Option"):
        idx=indices[title]
        similarity=list(enumerate(cosine_sim[idx]))
        similarity.sort(key=lambda x:x[1],reverse=True)
        similarity=similarity[1:11] # first movie will be the same
        recommended_movies=[i[0] for i in similarity]
        for i in recommended_movies:
            print(merged_df['title'].iloc[i])


# In[33]:


get_recommendations('Tin Can Man')
# for idx in indexes:
#     print(merged_df['title'].iloc[idx])


# In[36]:


option = st.selectbox('Select your favourite movie', merged_df['title'])
get_recommendations(option)


# In[ ]:




