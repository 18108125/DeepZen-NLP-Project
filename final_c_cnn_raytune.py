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
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
import random
import os
import csv
import gc
import ray
from ray import air, tune
from ray.air.config import RunConfig, CheckpointConfig
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from filelock import FileLock
from ray.tune.utils import validate_save_restore

os.environ['PYTHONHASHSEED'] = '22'
current_directory = os.getcwd()
print("Current directory:", current_directory)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% PREPARE EXPERIMENTAL DATASET 
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


data = np.load('data_embeddings.npy', allow_pickle=True) #Array shape = (10358496, 22) cols: subtitle line, imdb_id, genres (20 genres 1-hot encoded) 
embeddings = np.load('embeddings.npy', allow_pickle=True)
genre_to_idx = np.load('genre_to_idx.npy', allow_pickle=True).item()

with open("corpus.json", "r") as corpus_json: #UNSAVE 
    corpus = json.load(corpus_json)

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
del corpus
gc.collect()

genres_to_keep = ['Romance', 'Action', 'Thriller', 'Comedy', 'Drama']
romance_idx = genre_to_idx['Romance']
action_idx = genre_to_idx['Action']
thriller_idx = genre_to_idx['Thriller']
comedy_idx = genre_to_idx['Comedy']
drama_idx = genre_to_idx['Drama']

genres_to_keep_idx = [romance_idx, action_idx, thriller_idx, comedy_idx, drama_idx]

experimental_dataset_df = pd.DataFrame(data)
# assign names to columns
all_genres_list = [str(key) for key in genre_to_idx.keys()]
experimental_dataset_df.columns = ['subtitle', 'movie_id'] + all_genres_list
#Filter the training set, so only the movies which are at least one of the five genres are kept
experimental_dataset_df = experimental_dataset_df.drop(list(set(all_genres_list) - set(genres_to_keep)), axis='columns')

experimental_dataset_df = experimental_dataset_df[experimental_dataset_df['subtitle'].apply(lambda x: isinstance(x, str))] #remove nan/float values in the subtitle column
# group by 'movie_id', combine 'subtitle' and take the first row for the rest
experimental_dataset_df = experimental_dataset_df.groupby('movie_id').agg({
    'subtitle': ' '.join,
    **{column: 'first' for column in experimental_dataset_df.columns[2:]}
}).reset_index().set_index('movie_id')


experimental_dataset_df.to_pickle("experimental_dataset_df.pkl") #UNSAVE IF TESTING
experimental_dataset_df = pd.read_pickle("experimental_dataset_df.pkl")

#Create separate labels dataframe
cnn_labels_df = experimental_dataset_df.loc[:, genres_to_keep] #dataframe of (4664,5). index = movie_id, rows = unique movies, columns = the 5 genres one-hot encoded: Action, Romance, Thriller, Comedy, Drama


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


def pad_embeddings_matrix(embeddings_matrix):
    max_length = 8000
    words_length = embeddings_matrix.shape[0]
    padding_size = max_length - words_length
    top_padding = padding_size // 2
    bot_padding = padding_size - top_padding
    padded_embeddings_matrix = np.pad(embeddings_matrix, ((top_padding, bot_padding), (0, 0)), mode='constant', constant_values=0)
    return padded_embeddings_matrix.astype(np.float16)

# Preprocess the subtitle tracks and remove any empty subtitles
experimental_dataset_df['subtitle'] = experimental_dataset_df['subtitle'].apply(preprocessing)
experimental_dataset_df = experimental_dataset_df[experimental_dataset_df['subtitle'].apply(len) > 0]

subtitles_embeddings = []

for i, subtitle in tqdm(enumerate(experimental_dataset_df['subtitle'])):
    #Create word embeddings matrix (X in the paper) of dims nxk where n = number of words in subtitle track, k = dim of embedding
    embeddings_matrix = np.stack([embeddings[vocab_to_int[token],:] for token in subtitle])
    embeddings_matrix = embeddings_matrix[:8000,:]
    embeddings_matrix = pad_embeddings_matrix(embeddings_matrix)
    movie_id_value = experimental_dataset_df.index[i]
    experimental_dataset_df.at[movie_id_value, 'subtitle'] = embeddings_matrix.astype(np.float16)

del data
del embeddings
del embeddings_matrix
gc.collect()

#%% PREPARE DATASETS
cnn_index = np.arange(0, len(experimental_dataset_df), 1)

cnn_train_idx, cnn_test_idx = train_test_split(cnn_index, test_size=0.2, random_state=22) #60/20/20 train,test,validation split.
cnn_train_idx, cnn_validation_idx = train_test_split(cnn_train_idx, test_size=0.25, random_state=22) #60/20/20 train,test,validation split.

train_dataset = experimental_dataset_df.iloc[cnn_train_idx,0] 
flatten_train = np.array([array.flatten() for array in train_dataset.values])
train_dataset = torch.from_numpy(flatten_train)
train_labels = experimental_dataset_df.iloc[cnn_train_idx, 1:]
test_dataset = experimental_dataset_df.iloc[cnn_test_idx,0]
flatten_test = np.array([array.flatten() for array in test_dataset.values])
test_dataset = torch.from_numpy(flatten_test)
test_labels = experimental_dataset_df.iloc[cnn_test_idx, 1:]
validation_dataset = experimental_dataset_df.iloc[cnn_validation_idx,0]
flatten_validation = np.array([array.flatten() for array in validation_dataset.values])
validation_dataset = torch.from_numpy(flatten_validation)
validation_labels = experimental_dataset_df.iloc[cnn_validation_idx, 1:]

del experimental_dataset_df
gc.collect()

#Count genre distribution in train/validation/test sets
train_genre_counts =  np.sum(train_labels, axis=0)
validation_genre_counts =  np.sum(validation_labels, axis=0)
test_genre_counts =  np.sum(test_labels, axis=0)
full_genre_counts = np.stack((train_genre_counts,validation_genre_counts,test_genre_counts))

data_dir = "\\data_cnn"  # Path to the data subfolder
data_dir = current_directory + data_dir
# Create the data subfolder if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

np.save(os.path.join(data_dir, "train_dataset_cnn.npy"), train_dataset)
np.save(os.path.join(data_dir, "train_labels_cnn.npy"), train_labels)
np.save(os.path.join(data_dir, "test_dataset_cnn.npy"), test_dataset)
np.save(os.path.join(data_dir, "test_labels_cnn.npy"), test_labels)
np.save(os.path.join(data_dir, "validation_dataset_cnn.npy"), validation_dataset)
np.save(os.path.join(data_dir, "validation_labels_cnn.npy"), validation_labels)

del flatten_train
del flatten_test
del flatten_validation
del train_dataset
del test_dataset
del validation_dataset
del train_labels
del test_labels
del validation_labels
gc.collect()

#%% RAY TUNE

from torch.utils.data import Dataset
import datetime

class CustomDataset(Dataset):
    def __init__(self, full_df, index):
        self.full_df = full_df
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        full_idx = self.index[idx]
        data = self.full_df.iloc[full_idx]
        data2 = data.values
        return data2

def load_data(data_dir="\\data_cnn",):
    """The dataloaders are wrapping in their own function and a global data directory is passed so we can share the data between different trials
    (As suggested in the ray documentation). Filelock prevents multiple threads/processes from facing conflicts"""
    data_dir = os.path.join(current_directory, data_dir)
    lock_file_path = os.path.join(data_dir, ".lock")
        # FileLock added here to prevent concurrent access
    with FileLock(lock_file_path):                                                                                    
        train_data = np.load(os.path.join(data_dir, "train_dataset_cnn.npy"), allow_pickle=True)
        test_data = np.load(os.path.join(data_dir, "test_dataset_cnn.npy"), allow_pickle=True)
        validation_data = np.load(os.path.join(data_dir, "validation_dataset_cnn.npy"), allow_pickle=True)

        # Assuming you have labels as well, load them similarly
        train_labels = np.load(os.path.join(data_dir, "train_labels_cnn.npy"), allow_pickle=True)
        test_labels = np.load(os.path.join(data_dir, "test_labels_cnn.npy"), allow_pickle=True)
        validation_labels = np.load(os.path.join(data_dir, "validation_labels_cnn.npy"), allow_pickle=True)
    
    return TensorDataset(
        torch.tensor(train_data.astype('float16')).float(), 
        torch.tensor(train_labels.astype('float16')).float()), TensorDataset(
        torch.tensor(test_data.astype('float16')).float(), 
        torch.tensor(test_labels.astype('float16')).float()), TensorDataset(
        torch.tensor(validation_data.astype('float16')).float(), 
        torch.tensor(validation_labels.astype('float16')).float()
        )
 
class CCNN(nn.Module): 
    def __init__(self, n_filters, filter_sizes):
        super(CCNN, self).__init__()

        self.output_dim = 5
        self.dropout_rate = 0.5

        # Convolutional Layers
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, 300))
            for fs in filter_sizes
        ])

        # Fully Connected Layer
        self.fc = nn.Linear(len(filter_sizes) * n_filters, self.output_dim)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # x -> [batch size, 1, sent len, emb dim]

        #infer sentence length
        batch_siz = x.size(dim=0)
        sentlen_dim = x.size(dim=2)
        dim = 300
        sentlen = int(sentlen_dim / dim)

        #change shape of tensor from flattened back to matrix of dim
        x = x.view(batch_siz, 1, sentlen, dim).to(torch.float32)

        # Convolutional Layers
        conved = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # conved_n -> [batch size, n_filters, sent len - filter_sizes[n] + 1]

        # Max-over-time pooling
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n -> [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat -> [batch size, n_filters * len(filter_sizes)]

        return torch.sigmoid(self.fc(cat))


class TrainCNN(tune.Trainable):

    def setup(self, config):
        # Hyperparameters
        self.lr = config["lr"]
        self.decay_rate = config["decay_rate"]
        self.batch_size = int(config["batch_size"])
        self.n_filters = config['n_filters']
        self.filter_size1 = config['filter_size1']
        self.filter_size2 = self.filter_size1 + config['filter_size2']
        self.filter_size3 = self.filter_size2 + config['filter_size3']
        self.filter_sizes = [self.filter_size1, self.filter_size2, self.filter_size3]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data, self.test_data, self.validation_data = load_data(data_dir)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=True)
        self.model = CCNN(self.n_filters, self.filter_sizes ).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.lr, rho=self.decay_rate, eps=1e-6, weight_decay=1e-4) 
        self.t = 0
        self.decay_step = 100
        self.threshold = 0.5
        self.best_f1 = 0
        self.best_model = None
        self.checkpoint = None

    def step(self):
        train_model(
            self.model, self.train_loader, self.optimizer, self.criterion, self.device, self.t, self.lr, self.decay_step, self.decay_rate
        )
        f1 = evaluate_model(self.model, self.validation_loader, self.device, self.threshold)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_model_state = self.model.state_dict()
            self.checkpoint = self.save_checkpoint(data_dir + "/temp")
        return {"f1" : f1}

    #TEST NEW
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_dir = self._create_checkpoint_dir(checkpoint_dir=checkpoint_dir)
    # Save the model, and any additional state
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        with open(checkpoint_path, "wb") as checkpoint_file:
            torch.save(self.model.state_dict(), checkpoint_file)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        # Restore from saved state
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        with open(checkpoint_path, "rb") as checkpoint_file:
            self.model.load_state_dict(torch.load(checkpoint_file))

    def reset_config(self, new_config):
        # Reset any hyperparameters or configurations
        return True 
              
        
    def forward(self, x):
        # x -> [batch size, 1, sent len, emb dim]

        #infer sentence length
        batch_siz = x.size(dim=0)
        sentlen_dim = x.size(dim=2)
        dim = 300
        sentlen = int(sentlen_dim / dim)

        #change shape of tensor from flattened back to matrix
        x = x.view(batch_siz, 1, sentlen, dim).to(torch.float32)

        # Convolutional Layers
        conved = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # conved_n -> [batch size, n_filters, sent len - filter_sizes[n] + 1]

        # Max-over-time pooling
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n -> [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat -> [batch size, n_filters * len(filter_sizes)]

        return torch.sigmoid(self.fc(cat))


t=0
def train_model(model, iterator, optimizer, criterion, device, t, lr, decay_step, decay_rate, num_epochs=6):
    model.train()
    model.to(device)
    running_loss = 0
    for epoch in range(num_epochs):
        for data, labels in iterator:

            data = data.unsqueeze(1)  # Adding channel dimension
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(data)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if t % 500 == 499:    # print every 500 mini-batches
                print(f'[{epoch + 1}, {t + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
            t += 1 #Iteration step

            #decay the lr
            if t % decay_step == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * (decay_rate ** (t / decay_step))
        
    return running_loss / len(iterator)

def evaluate_model(model, validation_loader, device, threshold):
    model.to(device)
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():    
        for data, labels in validation_loader:
  
            data = data.unsqueeze(1)  # Adding channel dimension
            data, labels = data.to(device), labels.to(device)
            predictions = model(data)
            all_outputs.append(predictions.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())
        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)
        y_pred = (all_outputs > threshold).astype(int)
        f1 = f1_score(all_targets, y_pred, average='weighted')
        return f1


#CONFIG
config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "decay_rate": tune.uniform(0.8, 0.99),
        "batch_size": tune.choice([512, 1024, 2048]), 
        # "batch_size": tune.choice([16, 32, 64, 128]), #Choice of smaller batch sizes if needed.
        "n_filters" : tune.choice([64, 128, 256]),
        "filter_size1" : tune.choice([3, 4, 5]),
        "filter_size2" : tune.choice([0, 1, 2]),
        "filter_size3" : tune.choice([0, 1, 2])
    }

hyperopt_search = HyperOptSearch(metric='f1', mode='max') 

scheduler=ASHAScheduler(
    time_attr='training_iteration',
        max_t=3,  #number of iterations per trial/sample
        grace_period=1,
        reduction_factor=2
    )

model = TrainCNN

#Use GPU
trainable_with_resources = tune.with_resources(TrainCNN, {"cpu":12, "gpu": 1})

tuner = tune.Tuner(
    trainable_with_resources,
    param_space=config,
    tune_config = tune.TuneConfig(
        metric='f1', 
        mode="max", 
        search_alg= hyperopt_search,
        scheduler=scheduler,
        num_samples=10, # Number of trials
        max_concurrent_trials=1,
        reuse_actors=True,
        chdir_to_trial_dir=False #False
    ),
    run_config=air.RunConfig(storage_path="./")
)
results = tuner.fit()

if results.errors:
    print("At least one trial failed")

# Get the best result
best_result = results.get_best_result("f1", "max", "last")
best_checkpoint_dir = best_result.checkpoint 

# And the best metrics
best_metric = best_result.metrics

# Extract the best hyperparameters
best_lr = best_result.config["lr"]
best_decay_rate = best_result.config["decay_rate"]
best_batch_size = best_result.config["batch_size"]
best_n_filter = best_result.config["n_filters"]
best_filter_size1 = best_result.config["filter_size1"]
best_filter_size2 = best_filter_size1 + best_result.config["filter_size2"]
best_filter_size3 = best_filter_size2 + best_result.config["filter_size3"]


#%% TRAIN CCNN FUNCTION WITH BEST HYPERPARAMETERS AND THRESHOLD

class CCNN(nn.Module): #This is the OG
    def __init__(self, n_filters, filter_sizes):
        super(CCNN, self).__init__()

        self.output_dim = 5
        self.dropout_rate = 0.5

        # Convolutional Layers
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, 300))
            for fs in filter_sizes
        ])

        # Fully Connected Layer
        self.fc = nn.Linear(len(filter_sizes) * n_filters, self.output_dim)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # x -> [batch size, 1, sent len, emb dim]

        #infer sentence length
        batch_siz = x.size(dim=0)
        sentlen_dim = x.size(dim=2)
        dim = 300
        sentlen = int(sentlen_dim / dim)

        #change shape of tensor from flattened back to matrix
        x = x.view(batch_siz, 1, sentlen, dim).to(torch.float32)

        # Convolutional Layers
        conved = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # conved_n -> [batch size, n_filters, sent len - filter_sizes[n] + 1]

        # Max-over-time pooling
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n -> [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat -> [batch size, n_filters * len(filter_sizes)]

        return torch.sigmoid(self.fc(cat))


def train_model(model, iterator, optimizer, criterion, device, t, num_epochs=10):
    model.train()
    model.to(device)
    running_loss = 0
    for epoch in range(num_epochs):
        for data, labels in iterator:

            data = data.unsqueeze(1)  # Adding channel dimension
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(data)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if t % 500 == 499:    # print every 500 mini-batches
                print(f'[{epoch + 1}, {t + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
            t += 1 #Iteration step

def test_model(model, test_loader, device, threshold):
    model.to(device)
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():    
        for data, labels in test_loader:
  
            data = data.unsqueeze(1)  # Adding channel dimension
            data, labels = data.to(device), labels.to(device)
            predictions = model(data)
            all_outputs.append(predictions.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())
        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)
        y_pred = (all_outputs > threshold).astype(int)
        return y_pred, all_targets
    
def pytorch_cnn_threshold(train_dataset, validation_dataset, genres, thresholds):
    algorithm = "CCNN_RayTune"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CCNN(best_n_filter, filter_sizes=[best_filter_size1, best_filter_size2, best_filter_size3]).to(device)
    criterion = nn.BCELoss()
    epochs = 6 
    optimizer = torch.optim.Adadelta(model.parameters(), lr=best_lr, rho=best_decay_rate, eps=1e-6)
    t=0

    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True) 
    validation_loader = DataLoader(validation_dataset, batch_size=best_batch_size)


    train_model(model, train_loader, optimizer, criterion, device, t, num_epochs=epochs)
    torch.save(model.state_dict(), 'c_cnn_model_weights.pth')

    validation_results = {}
    for threshold in thresholds:    
        y_pred, y_test = test_model(model, validation_loader, device, threshold)
        current_fold_report = pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True, target_names=genres))
        weightedavg_f1_score_threshold = current_fold_report['weighted avg']['f1-score']
        print(weightedavg_f1_score_threshold)
        validation_results['{}'.format(threshold)] = weightedavg_f1_score_threshold
        best_threshold = max(validation_results, key=validation_results.get)
        best_threshold = float(best_threshold)
  
    return current_fold_report, best_threshold


def pytorch_cnn(train_dataset, test_dataset, genres):
    algorithm = "CCNN_RayTune"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CCNN(best_n_filter, filter_sizes=[best_filter_size1, best_filter_size2, best_filter_size3]).to(device)
    criterion = nn.BCELoss()
    epochs = 6 
    optimizer = torch.optim.Adadelta(model.parameters(), lr=best_lr, rho=best_decay_rate, eps=1e-6)
    t=0

    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size)


    train_model(model, train_loader, optimizer, criterion, device, t, num_epochs=epochs)
    torch.save(model.state_dict(), 'c_cnn_model_weights.pth')
        
    y_pred, y_test = test_model(model, test_loader, device, best_threshold)
    current_fold_report = pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True, target_names=genres))
  
    return current_fold_report

genres = ['Romance', 'Action', 'Thriller', 'Comedy', 'Drama']
cnn_raytune_best_hyperparameters = {'best_lr': best_lr,
                        'best_decay_rate': best_decay_rate, 
                        'best_batch_size': best_batch_size, 
                        'best_n_filter': best_n_filter, 
                        'best_filter_size1': best_filter_size1, 
                        'best_filter_size2': best_filter_size2, 
                        'best_filter_size3': best_filter_size3, 
                        }
with open('best_hyperparameters.json', 'w') as f:
    json.dump(cnn_raytune_best_hyperparameters, f)
t=0
thresholds = np.linspace(0.2, 0.6, num=5)
train_dataset, test_dataset, validation_dataset = load_data(data_dir)
report, best_threshold = pytorch_cnn_threshold(train_dataset, validation_dataset, genres, thresholds)
print(best_threshold)
train_dataset = ConcatDataset([train_dataset, validation_dataset])
report = pytorch_cnn(train_dataset, test_dataset, genres)
report.to_csv('cnn_raytune_report.csv')
full_genre_counts = pd.DataFrame(full_genre_counts)
full_genre_counts.to_csv("full_genre_counts_cnn_raytune.csv")

print("Finished")
