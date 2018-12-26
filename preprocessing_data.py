# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import deque

def preprocess_netflix(normalize = False) :

    movie_titles = pd.read_csv('/Volumes/EXT/netflix-prize-data/movie_titles.csv', 
                               encoding = 'ISO-8859-1', 
                               header = None, 
                               names = ['Id', 'Year', 'Name']).set_index('Id')
    
    print('Shape Movie-Titles:\t{}'.format(movie_titles.shape))
    
    df_raw = pd.read_csv('/Volumes/EXT/netflix-prize-data/combined_data_1.txt', header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])
    
    
    tmp_movies = df_raw[df_raw['Rating'].isna()]['User'].reset_index()
    movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]
    
    shifted_movie_indices = deque(movie_indices)
    shifted_movie_indices.rotate(-1)
    
    
    user_data = []
    
    for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):
        
        if df_id_1<df_id_2:
            tmp_df = df_raw.loc[df_id_1+1:df_id_2-1].copy()
        else:
            tmp_df = df_raw.loc[df_id_1+1:].copy()
            

        tmp_df['Movie'] = movie_id
        

        user_data.append(tmp_df)
    
    df = pd.concat(user_data)
    del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id
    print('Shape User-Ratings:\t{}'.format(df.shape))
    
    

    min_movie_ratings = 10000
    filter_movies = (df['Movie'].value_counts()>min_movie_ratings)
    filter_movies = filter_movies[filter_movies].index.tolist()
    

    min_user_ratings = 200
    filter_users = (df['User'].value_counts()>min_user_ratings)
    filter_users = filter_users[filter_users].index.tolist()
    

    df_filterd = df[(df['Movie'].isin(filter_movies)) & (df['User'].isin(filter_users))]
    del filter_movies, filter_users, min_movie_ratings, min_user_ratings
    print('Shape User-Ratings unfiltered:\t{}'.format(df.shape))
    print('Shape User-Ratings filtered:\t{}'.format(df_filterd.shape))
    
    df_filterd = df_filterd.drop('Date', axis=1).sample(frac=1).reset_index(drop=True)

    df_p = df_filterd.pivot_table(index='User', columns='Movie', values='Rating')
    print('Shape User-Movie-Matrix:\t{}'.format(df_p.shape))
    M = df_p.values
    
    d1,d2 = M.shape
    
    n = 10000
    
    temp = np.where(np.logical_not(df_p.isna()))
    Omega = d2*temp[0] + temp[1]
    np.random.shuffle(Omega)
    Omega_train = Omega[:-n]
    Omega_test  = Omega[-n:]
    y_train = M[Omega_train//d2,Omega_train%d2]
    y_test  = M[Omega_test//d2,Omega_test%d2]
    
    return df_p,M,df_filterd,y_train,y_test,Omega_train,Omega_test,d1,d2