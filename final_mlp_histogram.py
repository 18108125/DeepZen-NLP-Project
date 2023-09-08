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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
# from sklearn.metrics import precision_score, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import classification_report
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
from skorch import NeuralNetClassifier

data = np.load('data_embeddings.npy', allow_pickle=True) #Array shape = (10358496, 22) cols: subtitle line, imdb_id, genres (20 genres 1-hot encoded) 
embeddings = np.load('embeddings.npy', allow_pickle=True)
genre_to_idx = np.load('genre_to_idx.npy', allow_pickle=True).item()

genres_to_keep = ['Romance', 'Action', 'Thriller', 'Comedy', 'Drama']
romance_idx = genre_to_idx['Romance']
action_idx = genre_to_idx['Action']
thriller_idx = genre_to_idx['Thriller']
comedy_idx = genre_to_idx['Comedy']
drama_idx = genre_to_idx['Drama']

genres_to_keep_idx = [romance_idx, action_idx, thriller_idx, comedy_idx, drama_idx]

### NEWLY ADDED
corpus_raw = data[:,0]

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


def under_sampler(dataframe, genre_to_undersample, isolate_col=False, dont_remove_these_genres=[], removal_ratio=0.5):
    """
    Multi-label under sampler for cases of label inbalance. Removes rows based on the genre specified.
    
    Args:
        - dataframe (pandas.DataFrame): The input DataFrame.
        - genre_to_undersample (str): The column name to check for the value to remove rows.
        - isolate_col (bool): Toggle flag. If true and dont_remove_these_genres has at least one other genre, the movies containing 
        the specified genre(s) are not removed, thus ensuring genres with low count values are not removed during under sampling.
        - dont_remove_these_genres (list): List of column names (genres) to protect when performing undersampling (if isolate_col=True).
        - removal_ratio (float): The ratio of the rows to be removed.
    
    Returns:
        train_df (pandas.DataFrame): The DataFrame after adding rows.
        label_df (pandas.DataFrame): The label DataFrame after adding rows.
    """
    mask = dataframe[genre_to_undersample] == 1  #Initial mask based on the specified column
    
    if isolate_col and dont_remove_these_genres:  #Check additional conditions if toggle=True and dont_remove_these_genres is specified
        for col in dont_remove_these_genres:
            mask &= dataframe[col] == 0  #Additional condition to check the value in other specified columns

    label_rows_to_remove = dataframe[mask]  #DataFrame with the rows to remove
    label_num_rows = int(removal_ratio * len(label_rows_to_remove))
    random_rows = np.random.choice(label_rows_to_remove.index, label_num_rows, replace=False)
    dataframe = dataframe.drop(random_rows)
    return dataframe

def over_sampler(dataframe, genre_to_oversample, isolate_col=False, dont_duplicate_these_genres=[], duplication_ratio=2.0):
    """
    Multi-label over sampler for cases of label inbalance. Duplicates rows based on the genre specified.
    
    Args:
        - dataframe (pandas.DataFrame): The input DataFrame.
        - genre_to_oversample (str): The column name to check for the value to duplicate rows.
        - isolate_col (bool): Toggle flag. If true and dont_duplicate_these_genres has at least one other genre, the movies containing 
          the specified genre(s) are not duplicated, thus ensuring genres with high count values are not overly represented during over sampling.
        - dont_duplicate_these_genres (list): List of column names (genres) to protect from being duplicated (if isolate_col=True).
        - duplication_ratio (float): The ratio of the rows to be added.
    
    Returns:
        (pandas.DataFrame): The DataFrame after adding rows.
        dataframe (pandas.DataFrame): The label DataFrame after adding rows.
    """


    mask = dataframe[genre_to_oversample] == 1  #Initial mask based on the specified column

    
    if isolate_col and dont_duplicate_these_genres:  #Check additional conditions if toggle=True and dont_duplicate_these_genres is specified
        for col in dont_duplicate_these_genres:
            mask &= dataframe[col] == 0  #Additional condition to check the value in other specified columns

    label_rows_to_duplicate = dataframe[mask]  #All potential dataframe rows to duplicate
    label_num_rows = int(duplication_ratio * len(label_rows_to_duplicate))
    duplicated_label_rows = label_rows_to_duplicate.sample(n=label_num_rows, replace=True) #Duplicate the rows
    dataframe = pd.concat([dataframe, duplicated_label_rows])

    return  dataframe

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

corpus_raw = remove_nan_from_list(corpus_raw)
corpus = []

# for line in tqdm(corpus_raw): #UNSAVE IF STARTING FROM FRESH
#     line = preprocessing(line)
#     corpus.append(line)
# # Save preprocessed corpus
# with open("corpus.json", "w") as corpus_json: #UNSAVE IF STARTING FROM FRESH
#     json.dump(corpus, corpus_json)
with open("corpus.json", "r") as corpus_json: #UNSAVE 
    corpus = json.load(corpus_json)

# vocabulary =  create_vocabulary(corpus)
# #Save preprocessed vocabulary
# with open("vocabulary.json", "w") as vocabulary_json: #UNSAVE IF STARTING FROM FRESH
#     json.dump(vocabulary, vocabulary_json)
with open("vocabulary.json", "r") as vocabulary_json:
    vocabulary = json.load(vocabulary_json)

def vocab_to_int_int_to_vocab(corpus):
    """
    Create lookup tables for vocabulary
    Input:
    Corpus (list of lists): list of lists of words
    Output:
    vocab_to_int (dict): dict to covert words from the vocabulary into a unique integer
    int_to_vocab (dict): dict to covert integers related to words from the vocabulary back into words
    """

    words = [word for words_list in corpus for word in words_list] #Make the list of lists a single list of the words
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

vocab_to_int, int_to_vocab = vocab_to_int_int_to_vocab(corpus)
### NEWLY ADDED

#Filter the training set, so only the movies which are at least one of the three genres are kept
experimental_dataset = data[np.any(data[:,genres_to_keep_idx] ==1, axis=1)]
print(experimental_dataset.shape)
# np.save('experimental_dataset.npy', experimental_dataset)
# experimental_dataset = np.load('experimental_dataset.npy', allow_pickle=True)

# Make a new array where all the subtitles for the movie are on a single line.
experimental_dataset_df = pd.DataFrame(experimental_dataset)
# assign names to columns
all_genres_list = [str(key) for key in genre_to_idx.keys()]
experimental_dataset_df.columns = ['subtitle', 'movie_id'] + all_genres_list
experimental_dataset_df = experimental_dataset_df.drop(list(set(all_genres_list) - set(genres_to_keep)), axis='columns')

experimental_dataset_df = experimental_dataset_df[experimental_dataset_df['subtitle'].apply(lambda x: isinstance(x, str))] #remove nan/float values in the subtitle column
# group by 'movie_id', combine 'subtitle' and take the first row for the rest
experimental_dataset_df = experimental_dataset_df.groupby('movie_id').agg({
    'subtitle': ' '.join,
    **{column: 'first' for column in experimental_dataset_df.columns[2:]}
}).reset_index().set_index('movie_id')

experimental_dataset_df['subtitle'] = experimental_dataset_df['subtitle'].apply(preprocessing) #Now each subtitle track is a list of preprocessed tokens
experimental_dataset_df = experimental_dataset_df[experimental_dataset_df['subtitle'].apply(len) > 0] #remove any empty lists in the preprocessed subtitles column of lists.


# preprocessed_subtitles = [preprocessing(subtitle) for subtitle in subtitles_col] 
# with open("preprocessed_subtitles.json", "w") as preprocessed_subtitles_json: 
#     json.dump(preprocessed_subtitles, preprocessed_subtitles_json)
with open("preprocessed_subtitles.json", "r") as preprocessed_subtitles_json:  
    preprocessed_subtitles = json.load(preprocessed_subtitles_json)

def normalise_matrix(array):
    """This function efficiently normalises a numpy array by subtracting each element by 
    the minimum value along its row and then dividing this value by the difference between 
    the maximum and minimum values along its row. 
    i.e: X_i,j = ( X_i,j - min_i(X_i,j) ) / ( max_i(X_i,j) - min_i(X_i,j) ) 
    Input:
    array (np.array)
    Output:
    normalised_array (np.array)"""

    minimum_values = np.min(array, axis=1, keepdims=True)
    maximum_values = np.max(array, axis=1, keepdims=True)
    normalised_array = (array - minimum_values) / (maximum_values - minimum_values)
    return normalised_array

def create_word_histogram(word_vectors, s=25):
    histograms = []
    for dimension in range(word_vectors.shape[1]):
        hist, bin_edges = np.histogram(word_vectors[:, dimension], bins=s, range=(0, 1))
        l1_norm = np.sum(hist)
        normalized_hist = hist / l1_norm
        z_scores = (normalized_hist - np.mean(normalized_hist)) / np.std(normalized_hist) #Paper states the mean and std.dv are the mean and std.dv of all values in the histogram overall.
        histograms.append(z_scores)
    return histograms

# s = 25 #as defined in paper, 25 bins used for MLP-Histogram
i=0
for subtitle in tqdm(experimental_dataset_df['subtitle']):
    #Create word embeddings matrix (X in the paper) of dims nxk where n = number of words in subtitle track, k = dim of embedding
    embeddings_matrix = np.stack([embeddings[vocab_to_int[token],:] for token in subtitle]) 
    # Normalise the matrix X by: (X_i,j - min_i (X_i,j)) / max_i (X_i,j) - min_i (X_i,j)
    normalised_embeddings_matrix = normalise_matrix(embeddings_matrix)
    z_score_embeddings_matrix = np.transpose(np.array(create_word_histogram(normalised_embeddings_matrix, s=25)))
    
    experimental_dataset_df.iloc[i,0] = z_score_embeddings_matrix
    i+=1
    

# APPLY OVER/UNDERSAMPLING
# experimental_dataset_df = over_sampler(experimental_dataset_df, 'Thriller', isolate_col=True, dont_duplicate_these_genres=['Drama'], duplication_ratio=1.3)
# experimental_dataset_df = over_sampler(experimental_dataset_df, 'Action', isolate_col=True, dont_duplicate_these_genres=['Drama'], duplication_ratio=2.0)
# experimental_dataset_df = over_sampler(experimental_dataset_df, 'Romance', isolate_col=True, dont_duplicate_these_genres=['Drama','Action','Thriller'], duplication_ratio=4.0)

# experimental_dataset_df = under_sampler(experimental_dataset_df, 'Drama', isolate_col=True, dont_remove_these_genres=['Romance'], removal_ratio=0.5)
# experimental_dataset_df = under_sampler(experimental_dataset_df, 'Comedy', isolate_col=True, dont_remove_these_genres=['Romance'], removal_ratio=0.3)


#%% SPLIT DATASET
training_dataset, test_dataset = train_test_split(experimental_dataset_df, test_size=0.2, random_state=22)


#%% COMPUTE METRICS FUNCTION FOR USE WITH MODELS
def compute_metrics(y_true_all, y_pred_all, genres=None):
    """
    Calculate weighted average precision, recall, f1-score and accuracy across all folds. They are weighted by the supports. 

    Inputs:
    y_true_all (list of ndarray): A list of ground truth (correct) target values for each fold.
    y_pred_all (list of ndarray): A list of estimated targets as returned by a classifier for each fold.

    Outputs:
    report (dict): Classification report of precision, recall, F1 score and support for each label and average overall.
    """

    # Generate classification reports for each k-fold iteration
    for y_true, y_pred in zip(y_true_all, y_pred_all):
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True, target_names=genres)

    return report

#%% MLP MODEL

class MLP_Histogram(nn.Module):
    def __init__(self):
        super(MLP_Histogram, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(7500, 128),  # Input_size = 25 (bins) * 300 (embedding dimension)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),  
            nn.Dropout(0.5),
            nn.Linear(64, 5),  # Output dimension 5 (genre labels)
            nn.Sigmoid(),
        )   

        # Noise parameters
        self.noise_mean = 0
        self.noise_stddev = 0.02

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # self.layers[0] = nn.Linear(x.shape[1], 128).to(self.device) #input_size

        # Add Gaussian noise - as specified in paper
        noise = torch.randn_like(x) * self.noise_stddev + self.noise_mean
        x = x + noise

        # Flatten the input
        x = x.view(x.shape[0], -1)
        
        return self.layers(x)

def train_model(model, train_loader, criterion, optimizer, device, epochs):
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}")

def evaluate_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        all_outputs = []
        all_targets = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)
        y_pred = (all_outputs > 0.5).astype(int)
        return y_pred, all_targets.astype(int)
    
def pytorch_mlp(dataset, targets, genres):
    algorithm = "MLP_histogram"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    k=7 #kfold k=7
    kf = KFold(n_splits=k, shuffle=True) 

    #Store all predictions and truths for calculate metrics
    y_true_all = []
    y_pred_all = []

    # For storing each classification report for each fold
    report_each_fold = []

    model = MLP_Histogram().to(device)
    criterion = nn.BCELoss()
    epochs = 6 
    # decay_rate = 0.97  # r in paper as Adadelta used over Adam
    # decay_step = 100  # k in paper
    lr = 0.001 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for train_index, test_index in tqdm(kf.split(dataset)):
        X_train, X_test = dataset.iloc[train_index], dataset.iloc[test_index]
        y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]

        genre_counts =  np.sum(y_train, axis=0)
        print("Genre Counts:", genre_counts)
        
        flattened_arrays_list = [arr.flatten() for arr in X_train]
        flattened_test_arrays_list = [arr.flatten() for arr in X_test]

        X_train = np.vstack(flattened_arrays_list)
        X_test = np.vstack(flattened_test_arrays_list)
        y_train = y_train.values.astype('float32')
        y_test = y_test.values.astype('float32')

        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        testing_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
        train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
        test_loader = DataLoader(testing_dataset, batch_size=20)

        train_model(model, train_loader, criterion, optimizer, device, epochs)
        
        y_pred, y_test = evaluate_model(model, test_loader, device)
        current_fold_report = pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True, target_names=genres))
        report_each_fold.append(current_fold_report)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    report_all_folds = pd.concat(report_each_fold)
    report_folds_average = compute_metrics(y_true_all, y_pred_all, genres)
    report_folds_average_df = pd.DataFrame(report_folds_average)    
    report_folds_average_df.index = "{}_".format(algorithm) + report_folds_average_df.index.astype(str)    

    torch.save(model.state_dict(), 'mlp_histogram_trained_model.pth')

    return report_all_folds, report_folds_average_df

# # Unsave for quicker training when trialling
# training_subtitles_dataset_df = training_subtitles_dataset_df.iloc[:100]
# training_targets = training_targets.iloc[:100]

#%% RUN MLP FUNCTION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
genres = ['Romance', 'Action', 'Thriller', 'Comedy', 'Drama']
training_data, training_targets = training_dataset.iloc[:,0], training_dataset.iloc[:,1:]
test_data, test_targets = test_dataset.iloc[:,0], test_dataset.iloc[:,1:]

mlp_histogram_report_all_folds, mlp_histogram_report_folds_average_df = pytorch_mlp(training_data, training_targets, genres=genres)
model = MLP_Histogram().to(device)
model.load_state_dict(torch.load('mlp_histogram_trained_model.pth'))
model.eval()
mlp_histogram_test_report_all_folds, mlp_histogram_test_report_folds_average_df = pytorch_mlp(test_data, test_targets, genres=genres)

mlp_histogram_test_report_all_folds.to_csv('mlp_histogram_test_report_all_folds_osus.csv')
mlp_histogram_test_report_folds_average_df.to_csv('mlp_histogram_test_report_folds_average_osus.csv')

train_genre_counts =  np.sum(training_targets, axis=0)
test_genre_counts =  np.sum(test_targets, axis=0)
full_genre_counts = np.stack((train_genre_counts, test_genre_counts))
full_genre_counts = pd.DataFrame(full_genre_counts)
full_genre_counts.to_csv("full_genre_counts_mlp_histogram_osus.csv")

print("Finished")


#%%
"""
#Genres: Romance, Thriller, Action (,Drama, Comedy - added)

Embeddings size of 300
"""