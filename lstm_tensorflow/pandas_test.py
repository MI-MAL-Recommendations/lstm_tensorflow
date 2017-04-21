import pandas as pd
import numpy as np
import random
import time
from lstm_model import train_model
#COLUMNS = ["UserID",
#           "AnimeID",
#           "UserRating",
#           "Title",
#           "Genre1",
#           "Genre2",
#           "Genre3",
#           "Genre4",
#           "Genre5",
#           "Genre6",
#           "Genre7",
#           "Genre8",
#           "Genre9",
#           "Genre10",
#           "Genre11",
#           "Genre12",
#           "Genre13",
#           "Genre14",
#           "Genre15",
#           "Genre16",
#           "Genre17",
#           "Genre18",
#           "Genre19",
#           "Genre20",
#           "Genre21",
#           "Genre22",
#           "Genre23",
#           "Genre24",
#           "Genre25",
#           "Genre26",
#           "Genre27",
#           "Genre28",
#           "Genre29",
#           "Genre30",
#           "Genre31",
#           "Genre32",
#           "Genre33",
#           "Genre34",
#           "Genre35",
#           "Genre36",
#           "Genre37",
#           "Genre38",
#           "Genre39",
#           "Genre40",
#           "Genre41",
#           "Genre42",
#           "Genre43", #Genres 1-43
#           "MediaType",
#           "Episodes",
#           "OverallRating",
#           "ListMembership"]
#df_train = pd.read_csv("file:///C://Users//jaden//Downloads//data.csv", names = COLUMNS);

df_anime = pd.read_csv("file:///C:/Users/jaden/Documents/SYDE%20522/Data%20Set/anime.csv", header = 0);
df_ratings = pd.read_csv("file:///C:/Users/jaden/Documents/SYDE%20522/Data%20Set/rating.csv", header = 0);

df_ratings = df_ratings.head(100000) # for testing purposes only take the first 100000 ratings

max_ratings = 200

def build_training_sequences(ratings_data):
    train_seqs = []
    grouped = ratings_data.groupby('user_id')

    for name, group in grouped:
        group.sort_values('rating', ascending = True)
        group = group.head(max_ratings);
        anime_id_column = group['anime_id']

        user_ratings_number = len(group.index)
        index_of_label = random.choice(group.index)
        label = anime_id_column[index_of_label]
        anime_list = [anime_id_column[x] for x in group.index if x != index_of_label]

        train_seqs.append([anime_list,label,len(anime_list)])

    #i = 0
    #while i < len(ratings_data.index):
    #    user_id = user_id_column[i]
    #    anime_list = []
    #    user_start = i
    #    while user_id_column[i] == user_id: #use pandas groupby
    #        i += 1

    #    # Choose random index to be label of data
    #    index_of_label = random.randint(user_start, i-1)
    #    label = anime_id_column[index_of_label]

    #    anime_list = [anime_id_column[x] for x in range(user_start,i) if x != index_of_label]

    #    train_seqs.append([anime_list,label,len(anime_list)])
    return train_seqs

train_data = build_training_sequences(df_ratings)
seqs,lbls,lngths = zip(*train_data)
train_df = pd.DataFrame({'sub_seqs':seqs,
                         'sub_label':lbls,
                         'seq_length':lngths})
train_df.head()

split_perc=0.8
train_len, test_len = np.floor(len(train_df)*split_perc), np.floor(len(train_df)*(1-split_perc))
train, test = train_df.ix[:train_len-1], train_df.ix[train_len:train_len + test_len]
model = train_model(train,test,df_anime['anime_id'].max(),max_ratings)