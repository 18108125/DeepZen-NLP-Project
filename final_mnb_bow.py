import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter, defaultdict ,OrderedDict
from tqdm import tqdm
import json
import re
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
# from sklearn.metrics import precision_score, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from skmultilearn.adapt import MLkNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import random
import os
import csv

os.environ['PYTHONHASHSEED'] = '22'
current_directory = os.getcwd()
print("Current directory:", current_directory)

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocessing(passage, stop_words=True):
    """Function that takes sentences as input and preprocesses them to output tokens. The paper removes non-english characters, numbers/timestamps, converts to lowercase & removes stopwords.
    There is the option to remove stopwords, which the default is set to True.

    NOTE: Some of the subtitles are floats, so it checks if the 'passage' is a string first and if its not makes it a string before preprocessing it.
    
    Input: Passage of sentences/words from subtitle file.
    Output: List of preprocessed tokens"""
   
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(passage)
    tokens = [re.sub(r'[^a-zA-Z\s]', '', re.sub(r'/', '', token)).lower() for token in tokens if not token.isdigit() and not re.match(r'\d+:\d+:\d+', token) and re.sub(r'[^a-zA-Z\s]', '', re.sub(r'/', '', token)).isalpha()]

    if stop_words == True:
        stop_words = set(stopwords.words('english'))
        tokens = [tok for tok in tokens if tok not in stop_words]
    else:
        tokens = tokens
    return tokens

def mnb_input_preprocess(passage):
    """This function takes list of tokens (i.e. each row of the subtitle column of data) and coverts them to a string of tokens.
    """
    tokens = [token for token in passage]
    return ' '.join(tokens)

#%% PREPARE DATA
full_movies_subtitles = pd.read_csv("movies_subtitles.csv", sep=',', header = 0)
movies_meta = pd.read_csv("movies_meta.csv", sep=',', header = 0)
full_movies_subtitles_np = full_movies_subtitles.to_numpy() #Array shape = (10358496,4) columns = start time of passage, end time, text, imdb movie ID
movies_meta_np = movies_meta.to_numpy() #Array shape = (4690,24) columns = saved as movies_meta_headers variable below. Noteworthy cols: 3 = genres, 6 = imdb_id
movies_meta_headers = movies_meta.columns.tolist()
data = full_movies_subtitles_np[:,-2:]
extra_cols = np.zeros((data.shape[0], 20) ) #attempting for embeddings_v3.py
data = np.hstack((data,extra_cols))

#UNIQUE_GENRES - index for 1-hot encoding the genres into data
genres = []
for i in tqdm(range(movies_meta_np.shape[0])):
    genres_col = movies_meta['genres']
    genres_rc = genres_col.iloc[i]
    list_of_dicts = json.loads(genres_rc.replace("'", "\""))
    for dictionary in list_of_dicts:
        genres.append(dictionary['name'])

genres = set(genres)
genre_to_idx = dict(zip(genres, range(-20,0)))
print(genre_to_idx)
np.save('genre_to_idx.npy', genre_to_idx)
genre_to_idx = np.load('genre_to_idx.npy', allow_pickle=True).item()

# #DATA
for i in tqdm(range(data.shape[0])):
    genres = movies_meta.iloc[data[i,1] == movies_meta_np[:,6]]['genres']
    genres = genres.apply(eval)
    # indices = np.where(data[i,1] == movies_meta_np[:,6]) #Delete?
    # genres = movies_meta[indices,3] #Delete?
    # data[i,2] = []
    for item in genres:
        for dictionary in item:
                for key,value in dictionary.items():
                    if key == 'name':
                        idx = genre_to_idx[value]
                        data[i,idx] = 1
                
np.save('data_embeddings.npy', data)
data = np.load('data_embeddings.npy', allow_pickle=True) #Array shape = (10358496, 22) cols: subtitle line, imdb_id, genres (20 genres 1-hot encoded)

#%% DATA SPLIT
genres_to_keep = ['Romance', 'Action', 'Thriller', 'Comedy', 'Drama']

# Make a new DataFrame/array where all the subtitles for the movie are on a single line.
data = pd.DataFrame(data)
# assign names to columns
all_genres_list = [str(key) for key in genre_to_idx.keys()]
data.columns = ['subtitle', 'movie_id'] + all_genres_list
data = data.drop(list(set(all_genres_list) - set(genres_to_keep)), axis='columns')

data = data[data['subtitle'].apply(lambda x: isinstance(x, str))] #remove nan/float values in the subtitle column
# group by 'movie_id', combine 'subtitle' and take the first row for the rest, also make 'movie_id' column the new index
data = data.groupby('movie_id').agg({
    'subtitle': ' '.join,
    **{column: 'first' for column in data.columns[2:]}
}).reset_index().set_index('movie_id') #unsave if I want movie_id col to be the index


data['subtitle'] = data['subtitle'].apply(preprocessing)
df_filtered = data[data['subtitle'].apply(len) > 0]
data = df_filtered

# The data array(s) below consist of one line from the subtitle track, the imdb_id of the movie, then 1-hot encoding of which genre(s) it belongs to.
train_data_bow, test_data_bow = train_test_split(data, test_size=0.2, random_state=22)
train_data_bow, validation_data_bow = train_test_split(train_data_bow, test_size=0.25, random_state=22) #Final split 60/20/20
# corpus_raw = pd.DataFrame(train_data_bow)['subtitle']


def create_vocabulary(corpus):
    """ Creates a dictionary given input:data, where data contains all the preprocessed subtitle lines in the first column"""
    vocabulary = {}
    i = 0
    for line in tqdm(corpus):
        for word in line:
            if word not in vocabulary:
                vocabulary[word] = i
                i+=1
    return vocabulary

def remove_nan_from_list(list):
    return [x for x in list if not isinstance(x, float) or not np.isnan(x)]



corpus = train_data_bow['subtitle'] 


# Save preprocessed corpus
with open("corpus_bow.json", "w") as corpus_json: #UNSAVE IF STARTING FROM FRESH
    json.dump(corpus, corpus_json)
with open("corpus_bow.json", "r") as corpus_json: 
    corpus = json.load(corpus_json)


vocabulary_bow =  create_vocabulary(corpus)
#Save preprocessed vocabulary
with open("vocabulary_bow.json", "w") as vocabulary_json: #UNSAVE IF STARTING FROM FRESH
    json.dump(vocabulary_bow, vocabulary_json)
with open("vocabulary_bow.json", "r") as vocabulary_json:
    vocabulary_bow = json.load(vocabulary_json)


#%% TF-IDF
# genres_to_keep = ['Romance', 'Action', 'Thriller', 'Comedy', 'Drama']
romance_idx = genre_to_idx['Romance']
action_idx = genre_to_idx['Action']
thriller_idx = genre_to_idx['Thriller']
comedy_idx = genre_to_idx['Comedy']
drama_idx = genre_to_idx['Drama']

genres_to_keep_idx = [romance_idx, action_idx, thriller_idx, comedy_idx, drama_idx]


def inverse_document_frequency(subtitle, idf_dict):
    """ Counts the number of occurences of a word for each subtitle file, where subtitle contains the preprocessed subtitle lines.
    Inputs:
    subtitle (list): list containing preprocessed subtitle lines for a movie
    idf_dict (dict): dictionary where the full vocabulary of the corpus used to count the document frequency of words. 

    Output: idf_dict (dict): Updated idf_dict given latest subtitle track
    """

    doc_vocabulary = set()
    doc_vocabulary.update(set(subtitle))

    for word in doc_vocabulary:
        if word not in idf_dict:
            idf_dict[word] = 1
        else:
            idf_dict[word] += 1

    return idf_dict

def remove_words_with_document_freq_less_than(idf_dict, word_freq_movie_id, df_less_than):
    # Get words to remove from idf and word_freq_movie_id
    words_to_remove = [word for word, df in idf_dict.items() if df < df_less_than]

    # Remove words from word_freq_movie_id dicts
    for movie_id, word_freq_dict in word_freq_movie_id.items():
        words_to_remove = [word for word in word_freq_dict.keys() if word in words_to_remove]
        for word in words_to_remove:
            word_freq_dict.pop(word)

    return idf_dict, word_freq_movie_id

word_freq_movie_id = {}
idf = {}

for movie_id, row in tqdm(train_data_bow.iterrows()):
    subtitle = row['subtitle']
    idf = inverse_document_frequency(subtitle, idf)
    word_counts = dict(Counter(subtitle))
    word_freq_movie_id[movie_id] = word_counts


idf, word_freq_movie_id = remove_words_with_document_freq_less_than(idf_dict=idf, word_freq_movie_id=word_freq_movie_id, df_less_than=4)

np.save('idf.npy', idf)
np.save('word_freq_movie_id.npy', word_freq_movie_id)
idf = np.load('idf.npy', allow_pickle=True).item()
word_freq_movie_id = np.load('word_freq_movie_id.npy', allow_pickle=True).item() #D_ij in paper

#Count number of documents (movies)
number_of_unique_movies = len(word_freq_movie_id.keys())
print("number of unique movies: {}".format(number_of_unique_movies))

#Multiply word_freq by log(number of unique movies / number of movies word occurs in)
log_idf = {key: np.log(number_of_unique_movies/value) for key, value in idf.items()}

def downweight_word_frequencies(dictionary_of_word_frequencies, log_idf):
    """This function performs the downweighting of word frequencies.
    Inputs:
    dictionary_of_word_frequencies (dict): a dictionary for a given subtitle file of word frequencies. key = word, value = word frequency in the document.
    log_idf (dict): a dictionary of the log_idf values. log_idf values = np.log(total number of documents / number of document word_i occurs in). Key = word, value = log_idf

    Output:
    tf_idf_values_movie_id(dict): The downweighted word frequency values
    """
    tf_idf_values_movie_id = {key: value * log_idf[key] for key, value in dictionary_of_word_frequencies.items()}
    return tf_idf_values_movie_id

def normalise_word_frequencies(dictionary_of_word_frequencies):
    """This function performs the normalisation of word frequencies for a document (aka subtitle track) so length doesnt affect the classification. 
    It divides the word frequencies by the square root of the sum of squares of term frequencies for all words in the document.
    Inputs:
    dictionary_of_word_frequencies (dict): a dictionary for a given subtitle file of word frequencies. key = word, value = word frequency in the document.

    Output:
    normalised_tf_idf_values_movie_id(dict): The normalised downweighted word frequency values
    """
    word_frequencies = np.fromiter(dictionary_of_word_frequencies.values(), dtype=float)
    normalising_factor = np.sqrt(np.sum(np.power(word_frequencies, 2)))
    normalised_tf_idf_values_movie_id = {key: value / normalising_factor for key, value in dictionary_of_word_frequencies.items()}
    return normalised_tf_idf_values_movie_id

downweighted_word_frequencies = {}
#Perform equation 1.1 in paper: downweight the word frequencies by an implementation of idf
for movie_id, dictionary in tqdm(word_freq_movie_id.items()):
    downweighted_word_frequencies[movie_id] = downweight_word_frequencies(dictionary, log_idf)

np.save('downweighted_word_frequencies.npy', downweighted_word_frequencies)
downweighted_word_frequencies = np.load('downweighted_word_frequencies.npy', allow_pickle=True).item()

normalised_downweighted_word_frequencies = {}
#Perform equation 1.1 in paper: downweight the word frequencies by an implementation of idf
for movie_id, dictionary in tqdm(downweighted_word_frequencies.items()):
    normalised_downweighted_word_frequencies[movie_id] = normalise_word_frequencies(dictionary)

np.save('normalised_downweighted_word_frequencies.npy', normalised_downweighted_word_frequencies)
normalised_downweighted_word_frequencies = np.load('normalised_downweighted_word_frequencies.npy', allow_pickle=True).item()


def create_word_genre_df(normalised_downweighted_word_frequencies, data):
    # Initialize an empty list to store data
    rows = []

    # Iterate over each movie
    for movie_id, word_freqs in tqdm(normalised_downweighted_word_frequencies.items()):
        # Get genres for the movie
        genres = data.loc[movie_id][data.loc[movie_id]==1].index.tolist()

        # Iterate over each word and its frequency
        for word, freq in word_freqs.items():
            # Add an entry for each genre the movie is in
            for genre in genres:
                rows.append({'word': word, 'genre': genre, 'frequency': freq})

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(rows)

    # Pivot the DataFrame to have words as rows and genres as columns
    word_genre_df = df.pivot_table(index='word', columns='genre', values='frequency', aggfunc='sum', fill_value=0)

    return word_genre_df


word_genre_weights_df = create_word_genre_df(normalised_downweighted_word_frequencies=normalised_downweighted_word_frequencies, data=data)
word_genre_weights_df = word_genre_weights_df.applymap(lambda x: np.log(x) if x > 0 else 0) #Take the log of the summed values (as in paper)
word_genre_weights_df.to_pickle('word_genre_weights_df.pkl')
word_genre_weights_df = pd.read_pickle('word_genre_weights_df.pkl')


#%% MULTINOMIAL NAIVE BAYES - CLASSIFICATION
print("Multinomial Naive Bayes.")

def calculate_document_weights(word_genre_weights_df, test_word_freqs):
    """Calculate the weighted frequency of genres for a given test document.

    Input:
    - word_genre_weights_df (pd.DataFrame): DataFrame where the index corresponds to words
      and the columns are different genres. Each cell contains the weight of a word for a given genre.
    - test_word_freqs (dict): Dictionary where keys are words in the test document and values are their frequencies.

    Output:
    - pd.DataFrame: A series containing the log-transformed weighted sum of each genre's score based on the words in the test document.

    Notes:
    - The function first filters the words that are present in the test document.
    - It then multiplies these words' frequencies by their respective weights for each genre.
    - Finally, the function returns the log-transformed sum of weighted frequencies for each genre.
    """
    # Filter word_genre_weights_df to include only the words that appear in the test document
    filtered_word_genre_df = word_genre_weights_df.loc[word_genre_weights_df.index.intersection(test_word_freqs.keys())]
    
    # Multiply the word frequencies in the test document by the corresponding weights in filtered_word_genre_df
    weighted_freqs = filtered_word_genre_df.mul(pd.Series(test_word_freqs,dtype='float64'), axis=0)
    
    # Sum the weighted frequencies for each genre to get the total scores
    genre_scores = weighted_freqs.sum()

    #Ensure there are no negative scores as they would be invalid when you take the log of them
    valid_scores = np.where(genre_scores <= 0, 1e-6, genre_scores)
    
    # Take logs of each value in the dataframe
    log_genre_scores = np.log(valid_scores)

    return log_genre_scores


def normalise_genre_prediction(genre_predictions_DataFrame):
    """Normalise the genre predictions by subtracting each genre weight by the minimum weight, 
    then dividing by the difference between the maximum genre weight and minimum genre weight
    Input:
    genre_predictions_DataFrame (pd.DataFrame): predictions of each genre
    Output:
    normalised_predictions
    """
    minimum = genre_predictions_DataFrame.min(axis=0)
    maximum = genre_predictions_DataFrame.max(axis=0)
    genre_predictions_DataFrame = genre_predictions_DataFrame - minimum # Subtract minimum from the entire array
    denominator = maximum - minimum
    denominator = denominator + 1e-6 #Ensure denominator is not 0
    
    normalised_predictions = genre_predictions_DataFrame / denominator # Divide by the denominator array
    
    return pd.DataFrame(normalised_predictions)


#%% VALIDATION SET
#Get validation word freqs

genres_to_keep = ['Romance', 'Action', 'Thriller', 'Comedy', 'Drama']
romance_idx = genre_to_idx['Romance']
action_idx = genre_to_idx['Action']
thriller_idx = genre_to_idx['Thriller']
comedy_idx = genre_to_idx['Comedy']
drama_idx = genre_to_idx['Drama']
genres_to_keep_idx = [romance_idx, action_idx, thriller_idx, comedy_idx, drama_idx]

validation_word_freq_movie_id = {}
threshold = np.linspace(0.3, 0.7, num=5) # use validation set to find this
movie_id_index = []

validation_labels = validation_data_bow.iloc[:, 1:]
validation_data = pd.DataFrame(validation_data_bow.iloc[:,0])

#%% VALIDATION LOOP
validation_results = {}
best_f1 = 0
for thresh in threshold:
    validation_genre_predictions = pd.DataFrame()
    for movie_id, row in tqdm(validation_data.iterrows()):
        subtitle = row['subtitle']
        word_counts = Counter(subtitle)
        filtered_word_counts = {word: count for word, count in word_counts.items() if idf.get(word, 0) >= 4} #The paper states that for MNB only words that occur more than three times are taken into account (assuming they mean document frequency)
        validation_word_freq_movie_id[movie_id] = dict(filtered_word_counts)
        latest_genre_prediction = calculate_document_weights(word_genre_weights_df, validation_word_freq_movie_id[movie_id])
        normalised_genre_prediction = normalise_genre_prediction(latest_genre_prediction)
        validation_genre_predictions = pd.concat([validation_genre_predictions, normalised_genre_prediction], axis=1)

    validation_genre_predictions = validation_genre_predictions.T
    y_pred = (validation_genre_predictions.values > thresh).astype(int)
    classification_report_threshold = classification_report(validation_labels.astype(int), y_pred, zero_division=0, output_dict=True, target_names=genres_to_keep)
    weightedavg_f1_score_threshold = classification_report_threshold['weighted avg']['f1-score']
    validation_results['{}'.format(thresh)] = weightedavg_f1_score_threshold
best_threshold = max(validation_results, key=validation_results.get)
best_threshold = float(best_threshold)

print("best threshold", best_threshold)
    
#%% HOLD-OUT TEST DATA
#Get test word freqs
genres_to_keep = ['Romance', 'Action', 'Thriller', 'Comedy', 'Drama']
romance_idx = genre_to_idx['Romance']
action_idx = genre_to_idx['Action']
thriller_idx = genre_to_idx['Thriller']
comedy_idx = genre_to_idx['Comedy']
drama_idx = genre_to_idx['Drama']
genres_to_keep_idx = [romance_idx, action_idx, thriller_idx, comedy_idx, drama_idx]

test_word_freq_movie_id = {}
test_genre_predictions = pd.DataFrame()
movie_id_index = []
test_labels = test_data_bow.iloc[:, 1:]
test_data = pd.DataFrame(test_data_bow.iloc[:,0])

#%% TEST LOOP

for movie_id, row in tqdm(test_data.iterrows()):
    subtitle = row['subtitle']
    word_counts = Counter(subtitle)
    filtered_word_counts = {word: count for word, count in word_counts.items() if idf.get(word, 0) >= 4} #The paper states that for MNB only words that occur more than three times are taken into account (assuming they mean document frequency)
    test_word_freq_movie_id[movie_id] = dict(filtered_word_counts)
    latest_genre_prediction = calculate_document_weights(word_genre_weights_df, test_word_freq_movie_id[movie_id])
    normalised_genre_prediction = normalise_genre_prediction(latest_genre_prediction)
    test_genre_predictions = pd.concat([test_genre_predictions, normalised_genre_prediction], axis=1)
    
test_genre_predictions = test_genre_predictions.T
y_test_pred = (test_genre_predictions.values > best_threshold).astype(int)
test_classification_report = pd.DataFrame(classification_report(test_labels.astype(int), y_test_pred, zero_division=0, output_dict=True, target_names=genres_to_keep))
test_classification_report.to_csv('mnb_bow_report_new.csv')


print("Finished")


