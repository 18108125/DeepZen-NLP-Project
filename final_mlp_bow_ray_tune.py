import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
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
from sklearn.feature_extraction.text import CountVectorizer
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
import ray
from ray import tune
from ray.air.config import RunConfig, CheckpointConfig
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from filelock import FileLock


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


#%% PREPARE DATA
# full_movies_subtitles = pd.read_csv("movies_subtitles.csv", sep=',', header = 0)
# movies_meta = pd.read_csv("movies_meta.csv", sep=',', header = 0)
# full_movies_subtitles_np = full_movies_subtitles.to_numpy() #Array shape = (10358496,4) columns = start time of passage, end time, text, imdb movie ID
# movies_meta_np = movies_meta.to_numpy() #Array shape = (4690,24) columns = saved as movies_meta_headers variable below. Noteworthy cols: 3 = genres, 6 = imdb_id
# movies_meta_headers = movies_meta.columns.tolist()
# data = full_movies_subtitles_np[:,-2:]
# extra_cols = np.zeros((data.shape[0], 20) ) #attempting for embeddings_v3.py
# data = np.hstack((data,extra_cols))

# #UNIQUE_GENRES - index for 1-hot encoding the genres into data
# genres = []
# for i in tqdm(range(movies_meta_np.shape[0])):
#     genres_col = movies_meta['genres']
#     genres_rc = genres_col.iloc[i]
#     list_of_dicts = json.loads(genres_rc.replace("'", "\""))
#     for dictionary in list_of_dicts:
#         genres.append(dictionary['name'])

# genres = set(genres)
# genre_to_idx = dict(zip(genres, range(-20,0)))
# print(genre_to_idx)
# np.save('genre_to_idx.npy', genre_to_idx)
genre_to_idx = np.load('genre_to_idx.npy', allow_pickle=True).item()

# #DATA
# for i in tqdm(range(data.shape[0])):
#     genres = movies_meta.iloc[data[i,1] == movies_meta_np[:,6]]['genres']
#     genres = genres.apply(eval)
#     # indices = np.where(data[i,1] == movies_meta_np[:,6]) #Delete?
#     # genres = movies_meta[indices,3] #Delete?
#     # data[i,2] = []
#     for item in genres:
#         for dictionary in item:
#                 for key,value in dictionary.items():
#                     if key == 'name':
#                         idx = genre_to_idx[value]
#                         data[i,idx] = 1
                
# np.save('data_embeddings.npy', data)
data = np.load('data_embeddings.npy', allow_pickle=True) #Array shape = (10358496, 22) cols: subtitle line, imdb_id, genres (20 genres 1-hot encoded)


#%% CREATE BoW input for MLP
genres_to_keep = ['Romance', 'Action', 'Thriller', 'Comedy', 'Drama']
romance_idx = genre_to_idx['Romance']
action_idx = genre_to_idx['Action']
thriller_idx = genre_to_idx['Thriller']
comedy_idx = genre_to_idx['Comedy']
drama_idx = genre_to_idx['Drama']

genres_to_keep_idx = [romance_idx, action_idx, thriller_idx, comedy_idx, drama_idx]

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


#%% DATA SPLIT
# Prep the subtitle col for CountVectorizer MLP input
def mlp_input_preprocess(passage):
    """This function takes list of tokens (i.e. each row of the subtitle column of data) and coverts them to a string of tokens so CountVectorizer can be used.
    It also removes all tokens that aren't in vocabulary_bow - the 50,000 most frequent words in the corpus"""
    # tokens = passage.split()
    filtered_tokens = [token for token in passage if token in vocabulary_bow]
    return ' '.join(filtered_tokens)

data['subtitle'] = data['subtitle'].apply(preprocessing)
df_filtered = data[data['subtitle'].apply(len) > 0]
data = df_filtered


# APPLY OVER/UNDERSAMPLING
# data = over_sampler(data, 'Thriller', isolate_col=True, dont_duplicate_these_genres=['Drama'], duplication_ratio=1.2)
# data = over_sampler(data, 'Action', isolate_col=True, dont_duplicate_these_genres=['Drama'], duplication_ratio=1.5)
# data = over_sampler(data, 'Romance', isolate_col=True, dont_duplicate_these_genres=['Drama','Action','Thriller'], duplication_ratio=4.0)

# data = under_sampler(data, 'Drama', isolate_col=True, dont_remove_these_genres=['Romance'], removal_ratio=0.5)
# data = under_sampler(data, 'Comedy', isolate_col=True, dont_remove_these_genres=['Romance'], removal_ratio=0.3)

# The data array(s) below consist of one line from the subtitle track, the imdb_id of the movie, then 1-hot encoding of which genre(s) it belongs to.
train_data_bow, test_data_bow = train_test_split(data, test_size=0.2, random_state=22) 

# Create vocabulary of 50,000 most frequent words
corpus = train_data_bow['subtitle'] 
vocabulary_bow = Counter()
vocabulary_bow_dict = {vocabulary_bow.update(Counter(subtitle)) for subtitle in corpus}
vocabulary_bow = dict(vocabulary_bow.most_common(50000))

# Create BoW datasets of using 50000 most frequent words
#Trainset
vectorizer = CountVectorizer()
train_data_bow['subtitle'] = train_data_bow['subtitle'].apply(mlp_input_preprocess)
mlp_bow_input_array = vectorizer.fit_transform(train_data_bow['subtitle'])
mlp_bow_input_array = mlp_bow_input_array.toarray()
np.save('mlp_bow_input_array.npy', mlp_bow_input_array)
mlp_bow_input_array = np.load('mlp_bow_input_array.npy', allow_pickle=True) #array shape = (4665, 49980) rows = movies, cols: top 50,000 words

#Testset
test_data_bow['subtitle'] = test_data_bow['subtitle'].apply(mlp_input_preprocess)
test_mlp_bow_input_array = vectorizer.fit_transform(test_data_bow['subtitle'])
test_mlp_bow_input_array = test_mlp_bow_input_array.toarray()
np.save('test_mlp_bow_input_array.npy', test_mlp_bow_input_array)
test_mlp_bow_input_array = np.load('test_mlp_bow_input_array.npy', allow_pickle=True) #array shape = (4665, 49980) rows = movies, cols: top 50,000 words
testing_targets = test_data_bow.iloc[:,1:]

# Split the training data into 80% training and 20% testing (validation)
training_targets = train_data_bow.iloc[:,1:]
n = mlp_bow_input_array.shape[0]
train_idx = np.random.choice(n, int(n * 0.8), replace=False)
test_idx = np.setdiff1d(np.arange(n), train_idx)

X_train_full, X_test_full = mlp_bow_input_array[train_idx], mlp_bow_input_array[test_idx]
y_train_full, y_test_full = training_targets.iloc[train_idx], training_targets.iloc[test_idx]

#Count genre distribution in train/validation/test sets
train_genre_counts =  np.sum(y_train_full, axis=0)
validation_genre_counts =  np.sum(y_test_full, axis=0)
test_genre_counts =  np.sum(testing_targets, axis=0)
full_genre_counts = np.stack((train_genre_counts,validation_genre_counts,test_genre_counts))

data_dir = "\\data"  # Path to the data subfolder
# Create the data subfolder if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

np.save(os.path.join(data_dir, "train_data_mlpbow.npy"), X_train_full)
np.save(os.path.join(data_dir, "test_data_mlpbow.npy"), X_test_full)
np.save(os.path.join(data_dir, "train_labels_mlpbow.npy"), y_train_full)
np.save(os.path.join(data_dir, "test_labels_mlpbow.npy"), y_test_full)

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

def load_data(data_dir="\\data"):
    """The dataloaders are wrapping in their own function and a global data directory is passed so we can share the data between different trials
    (As suggested in the ray documentation). Filelock prevents multiple threads/processes from facing conflicts"""
    data_dir = os.path.join(current_directory, data_dir)
    lock_file_path = os.path.join(data_dir, ".lock")
    # We add FileLock here to prevent concurrent access
    with FileLock(lock_file_path):
        train_data = np.load(os.path.join(data_dir, "train_data_mlpbow.npy"), allow_pickle=True)
        test_data = np.load(os.path.join(data_dir, "test_data_mlpbow.npy"), allow_pickle=True)

        train_labels = np.load(os.path.join(data_dir, "train_labels_mlpbow.npy"), allow_pickle=True)
        test_labels = np.load(os.path.join(data_dir, "test_labels_mlpbow.npy"), allow_pickle=True)

    return TensorDataset(torch.from_numpy(train_data.astype('float32')).float(), torch.from_numpy(train_labels.astype('float32')).float()), TensorDataset(torch.from_numpy(test_data.astype('float32')).float(), torch.from_numpy(test_labels.astype('float32')).float())

#%% MLP MODEL
class MLP_BoW(nn.Module):
    def __init__(self):
        super(MLP_BoW, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(0, 512),  # Input_size = XXX
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),  
            nn.Dropout(0.5),
            nn.Linear(256, 5),  # Output dimension 5 (genre labels)
            nn.Sigmoid(),
        )
           
    def forward(self, x):
        self.layers[0] = nn.Linear(x.shape[1], 512).to(device)

        # Flatten the input
        x = x.view(x.shape[0], -1).to(device)
        
        return self.layers(x)

class TrainMLP_BoW(tune.Trainable):

    def setup(self, config):
        # Hyperparameters
        self.lr = config["lr"]
        self.decay_rate = config["decay_rate"]
        self.batch_size = int(config["batch_size"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data, self.test_data = load_data()
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        self.model = MLP_BoW().to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.lr, rho=self.decay_rate, eps=1e-6)
        self.t = 0
        self.decay_step = 100
        self.threshold = 0.5
        self.best_f1 = 0
        self.best_model = None
        self.checkpoint = 0

    def step(self):
        train_model(
            self.model, self.train_loader, self.criterion, self.optimizer, self.device, self.t, self.lr, self.decay_step, self.decay_rate
        )
        f1 = validate_model(self.model, self.test_loader, self.device, self.threshold)      
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_model_state = self.model.state_dict()
            self.checkpoint = self.save(prevent_upload=True)
        return {"f1" : f1}
    
    def save_checkpoint(self, checkpoint_dir):
        # Save the model, and any additional state
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.best_model_state, checkpoint_path)
        return 

    def load_checkpoint(self, checkpoint_path):
        # Restore from saved state
        checkpoint_path = os.path.join(checkpoint_path, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))
           
        
    def forward(self, x):
        # Flatten the input
        x = x.view(x.shape[0], -1)
        
        return self.layers(x)
    


def train_model(model, train_loader, criterion, optimizer, device, t, lr, decay_step, decay_rate, num_epochs=5):
        model.to(device)
        model.train()
        for epoch in range(num_epochs):
            # running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                t += 1 #Iteration step

                #decay the lr
                if t % decay_step == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr * (decay_rate ** (t / decay_step))


def validate_model(model, test_loader, device, threshold):
    model.to(device)
    model.eval()
    running_val_loss = 0.0
    all_outputs = []
    y_test = []
    with torch.no_grad():
          for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # input_size = input.shape[1]
            outputs = model(inputs)
            all_outputs.append(outputs.detach().cpu().numpy())
            y_test.append(targets.detach().cpu().numpy())
            # val_loss = criteron(outputs, targets)
            # running_val_loss += val_loss.item() * inputs.size(0) #CHANGE TO CALCULATE F1
    all_outputs = np.vstack(all_outputs)
    y_test = np.vstack(y_test)
    y_pred = (all_outputs > threshold).astype(int)
    # val_loss = running_val_loss / len(test_loader.dataset)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return f1


data_dir = os.path.abspath("./data")
trainset, testset = load_data(data_dir)

config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "decay_rate": tune.uniform(0.8, 0.99),
        "batch_size": tune.choice([16, 32, 64, 128]),
    }

hyperopt_search = HyperOptSearch(metric='f1', mode='max') #CHange to MAX & F1

scheduler=ASHAScheduler(
    time_attr='training_iteration',
        max_t=3, #5 #number of iterations per trial/sample
        grace_period=1,
        reduction_factor=2
    )

model = TrainMLP_BoW

#Use GPU
trainable_with_resources = tune.with_resources(TrainMLP_BoW, {"cpu":12, "gpu": 1})

tuner = tune.Tuner(
    trainable_with_resources,
    param_space=config,
    tune_config = tune.TuneConfig(
        metric='f1', #change to F1
        mode="max", #change to MAX
        search_alg= hyperopt_search,
        scheduler=scheduler,
        num_samples=10,  #20,  # Number of trials
        max_concurrent_trials=5,
        chdir_to_trial_dir=True #False
    ),
)
results = tuner.fit()

if results.errors:
    print("At least one trial failed")

# Get the best result
best_result = results.get_best_result("f1", "max", "last")
best_checkpoint_dir = best_result.checkpoint 
best_checkpoint_dir_path = best_result.checkpoint.path

# And the best metrics
best_metric = best_result.metrics

# Extract the best hyperparameters
best_lr = best_result.config["lr"]
best_decay_rate = best_result.config["decay_rate"]
best_batch_size = best_result.config["batch_size"]

#%% FIND OPTIMAL THRESHOLD BY MAXIMISING F1 SCORE FROM A RANGE OF THRESHOLDS
class TuneThreshold(tune.Trainable):
    def setup(self, config):
        # Initialize with the best hyperparameters
        self.lr = best_lr
        self.decay_rate = best_decay_rate
        self.batch_size = best_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data, self.test_data = load_data()
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        self.model = MLP_BoW().to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.lr, rho=self.decay_rate, eps=1e-6)
        self.t = 0
        self.decay_step = 100
        self.threshold = config["threshold"]

    def step(self):
        train_model(
            self.model, self.train_loader, self.criterion, self.optimizer, self.device, self.t, self.lr, self.decay_step, self.decay_rate
        )
        # val_loss = validate_model(self.model, self.test_loader, self.device, self.criterion)
        f1 = validate_model(self.model, self.test_loader, self.device, self.threshold)
        return {"f1" : f1}
    
    def save_checkpoint(self, checkpoint_dir):
        # Save the model, and any additional state
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        # Restore from saved state
        self.model.load_state_dict(torch.load(checkpoint_path))
           
        
    def forward(self, x):
        # Flatten the input
        x = x.view(x.shape[0], -1)
        
        return self.layers(x)

model = TuneThreshold
trainable_with_resources = tune.with_resources(model, {"cpu":12, "gpu": 1})

# Create a new config for just the threshold
threshold_config = {
    "threshold": tune.choice([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
}

thresh_hyperopt_search = HyperOptSearch(metric='f1', mode='max') 

thresh_scheduler=ASHAScheduler(
    time_attr='training_iteration',
        max_t=2, #5 #number of iterations per trial/sample
        grace_period=1,
        reduction_factor=2
    )

threshold_tuner = tune.Tuner(
    trainable_with_resources,
    param_space=threshold_config,
    tune_config = tune.TuneConfig(
        metric='f1', 
        mode="max", 
        search_alg= thresh_hyperopt_search,
        scheduler=thresh_scheduler,
        num_samples=10,  # Number of trials
        max_concurrent_trials=5,
        chdir_to_trial_dir=False
    )
)

threshold_results = threshold_tuner.fit()

threshold_best_result = threshold_results.get_best_result("f1", "max", "last")
best_threshold = threshold_best_result.config["threshold"]
best_checkpoint_dir = best_result.checkpoint 
best_checkpoint_dir_path = best_result.checkpoint.path



#%% TRAIN MLP FUNCTION WITH BEST HYPERPARAMETERS AND THRESHOLD


class MLP_BoW(nn.Module):
    def __init__(self):
        super(MLP_BoW, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(0, 512),  # Input_size = XXX
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),  
            nn.Dropout(0.5),
            nn.Linear(256, 5),  # Output dimension 5 (genre labels)
            nn.Sigmoid(),
        )
                   
    def forward(self, x):
        self.layers[0] = nn.Linear(x.shape[1], 512).to(device)
        
        # Flatten the input
        x = x.view(x.shape[0], -1).to(device)
        
        return self.layers(x)


def train_model(model, train_loader, criterion, optimizer, device, t, num_epochs=10):
        model.train()
        running_loss = 0
        for epoch in tqdm(range(num_epochs)):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                t += 1 #Iteration step

                running_loss += loss.item()

                if t % 500 == 499:    # print every 500 mini-batches
                    print(f'[{epoch + 1}, {t + 1:5d}] loss: {running_loss / 500:.3f}')
                    running_loss = 0.0

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
        y_pred = (all_outputs > best_threshold).astype(int)
        return y_pred, all_targets.astype(int)
    
def pytorch_mlp(training_dataset, training_targets, test_dataset, testing_targets, genres):
    algorithm = "MLP_BoW_RayTune"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP_BoW().to(device)
    criterion = nn.BCELoss()
    epochs = 6 
    optimizer = torch.optim.Adadelta(model.parameters(), lr=best_lr, rho=best_decay_rate, eps=1e-6)
    t=0

    
    train_dataset = TensorDataset(torch.from_numpy(training_dataset.astype('float32')).float(), torch.from_numpy(training_targets.values.astype('float32')).float())
    testing_dataset = TensorDataset(torch.from_numpy(test_dataset.astype('float32')).float(), torch.from_numpy(testing_targets.values.astype('float32')).float())
    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True) 
    test_loader = DataLoader(testing_dataset, batch_size=best_batch_size)


    train_model(model, train_loader, criterion, optimizer, device, t, num_epochs=epochs)

        
    y_pred, y_test = evaluate_model(model, test_loader, device)
    current_fold_report = pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True, target_names=genres))
    
    return current_fold_report


genres = ['Romance', 'Action', 'Thriller', 'Comedy', 'Drama']
best_hyperparameters = pd.DataFrame([best_lr, best_decay_rate, best_batch_size, best_threshold])
best_hyperparameters.to_csv("mlp_bow_raytune_hyperparameters_osus.csv")
t=0
training_dataset = mlp_bow_input_array
testing_dataset = test_mlp_bow_input_array
report = pytorch_mlp(training_dataset, training_targets, testing_dataset, testing_targets, genres)
report.to_csv('mlp_bow_raytune_report_osus.csv')
full_genre_counts = pd.DataFrame(full_genre_counts)
full_genre_counts.to_csv("full_genre_counts_mlp_bow_raytune_osus.csv")

print("Finished")

