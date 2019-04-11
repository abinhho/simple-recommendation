import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer



df = pd.read_csv('./movies.csv')

df = df[['Title','Genre','Director','Actors','Plot']]
df.head()

df['Key_words'] = ''

for index, row in df.iterrows():
	plot = row['Plot']

	r = Rake()

	r.extract_keywords_from_text(plot)

	key_words_dict_scores = r.get_word_degrees()

	row['Key_words'] = list(key_words_dict_scores.keys())

df.drop(columns = ['Plot'], inplace = True)

df.set_index('Title', inplace = True)
df.head()


df['bag_of_words'] = ''

columns = df.columns

for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col != 'Director':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['bag_of_words'] = words

df.drop(columns = [col for col in df.columns if col!= 'bag_of_words'], inplace = True)

count = CountVectorizer()

count_matrix = count.fit_transform(df['bag_of_words'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df.index)

print(indices[:5])

def recommendations(title, cosine_sim = cosine_sim):
	recommended_movies = []

	idx = indices[indices == title].index[0]

	score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

	top_10_indexes = list(score_series.iloc[1:11].index)

	for i in top_10_indexes:
		recommended_movies.append(list(df.index)[i])

	return recommended_movies

print(recommendations('Fargo'))
