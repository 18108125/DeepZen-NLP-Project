import torch
import numpy as np
import pandas as pd
import math
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter, defaultdict, OrderedDict
from tqdm import tqdm
import re
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  KFold
# from sklearn.metrics import precision_score, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from skmultilearn.adapt import MLkNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import csv
import json
import ast
# print(os.getcwd())


def preprocessing(passage, stop_words=True):
    """Function that takes sentences as input and preprocesses them to output tokens. The paper removes numbers/timestamps, converts to lowercase & removes stopwords.
    There is the option to remove stopwords, which the default is set to True.

    NOTE: Some of the subtitles are floats (NaN values), so it checks if the 'passage' is a string first and if its not makes it a string before preprocessing it.
    
    Input: Passage of sentences/words from subtitle file.
    Output: List of preprocessed tokens"""
   
    if not isinstance(passage, str):
            passage = str(passage)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(passage)
    tokens = [re.sub(r'/', '', token) for token in tokens if not token.isdigit() and not re.match(r'\d+:\d+:\d+', token)] 
    tokens = [token.lower() for token in tokens] 
    tokens = [tok for tok in tokens if tok.isalpha()] 
    # passage = passage.lower()
    if stop_words == True:
        stop_words = set(stopwords.words('english'))
        tokens = [tok for tok in tokens if tok not in stop_words]
    else:
        tokens = tokens
    return tokens
    
def under_sampler(train_df, label_df, genre_to_undersample, isolate_col=False, dont_remove_these_genres=[], removal_ratio=0.5):
    """
    Multi-label under sampler for cases of label inbalance. Removes rows based on the genre specified.
    
    Args:
        - train_df (pandas.DataFrame): The input DataFrame.
        - label_df (pandas.DataFrame): The associated labels of the input DataFrame.
        - genre_to_undersample (str): The column name to check for the value to remove rows.
        - isolate_col (bool): Toggle flag. If true and dont_remove_these_genres has at least one other genre, the movies containing 
        the specified genre(s) are not removed, thus ensuring genres with low count values are not removed during under sampling.
        - dont_remove_these_genres (list): List of column names (genres) to protect when performing undersampling (if isolate_col=True).
        - removal_ratio (float): The ratio of the rows to be removed.
    
    Returns:
        train_df (pandas.DataFrame): The DataFrame after adding rows.
        label_df (pandas.DataFrame): The label DataFrame after adding rows.
    """
    mask = label_df[genre_to_undersample] == 1  #Initial mask based on the specified column
    
    if isolate_col and dont_remove_these_genres:  #Check additional conditions if toggle=True and dont_remove_these_genres is specified
        for col in dont_remove_these_genres:
            mask &= label_df[col] == 0  #Additional condition to check the value in other specified columns

    label_rows_to_remove = label_df[mask]  #DataFrame with the rows to remove
    label_num_rows = int(removal_ratio * len(label_rows_to_remove))
    random_rows = np.random.choice(label_rows_to_remove.index, label_num_rows, replace=False)
    label_df = label_df.drop(random_rows)
    train_df = train_df.drop(random_rows)
    return train_df, label_df

def over_sampler(train_df, label_df, genre_to_oversample, isolate_col=False, dont_duplicate_these_genres=[], duplication_ratio=2.0):
    """
    Multi-label over sampler for cases of label inbalance. Duplicates rows based on the genre specified.
    
    Args:
        - train_df (pandas.DataFrame): The input DataFrame.
        - label_df (pandas.DataFrame): The associated labels of the input DataFrame.
        - genre_to_oversample (str): The column name to check for the value to duplicate rows.
        - isolate_col (bool): Toggle flag. If true and dont_duplicate_these_genres has at least one other genre, the movies containing 
          the specified genre(s) are not duplicated, thus ensuring genres with high count values are not overly represented during over sampling.
        - dont_duplicate_these_genres (list): List of column names (genres) to protect from being duplicated (if isolate_col=True).
        - duplication_ratio (float): The ratio of the rows to be added.
    
    Returns:
        train_df (pandas.DataFrame): The DataFrame after adding rows.
        label_df (pandas.DataFrame): The label DataFrame after adding rows.
    """
    mask = label_df[genre_to_oversample] == 1  #Initial mask based on the specified column

    
    if isolate_col and dont_duplicate_these_genres:  #Check additional conditions if toggle=True and dont_duplicate_these_genres is specified
        for col in dont_duplicate_these_genres:
            mask &= label_df[col] == 0  #Additional condition to check the value in other specified columns

    label_rows_to_duplicate = label_df[mask]  #All potential label_df rows to duplicate
    label_num_rows = int(duplication_ratio * len(label_rows_to_duplicate))
    duplicated_label_rows = label_rows_to_duplicate.sample(n=label_num_rows, replace=True) #Duplicate the 
    duplicated_train_rows = train_df.loc[duplicated_label_rows.index] #As they share the same index, duplicate the train_df rows to match the duplicated label rows
    train_df = pd.concat([train_df, duplicated_train_rows])
    label_df = pd.concat([label_df, duplicated_label_rows])

    return train_df, label_df

full_movies_subtitles = pd.read_csv("movies_subtitles.csv", sep=',', header = 0)
movies_meta = pd.read_csv("movies_meta.csv", sep=',', header = 0)

#SPLIT DATA
train_movies_subtitles, test_movies_subtitles = train_test_split(full_movies_subtitles, test_size=0.2, random_state=22)

movies_meta = movies_meta[movies_meta['imdb_id'].isin(train_movies_subtitles['imdb_id'])]
test_movies_meta = movies_meta[movies_meta['imdb_id'].isin(test_movies_subtitles['imdb_id'])]

full_movies_subtitles = train_movies_subtitles

full_movies_subtitles_np = full_movies_subtitles.to_numpy() #Array shape = (n,4) columns = start time of passage, end time, text, imdb movie ID
movies_meta_np = movies_meta.to_numpy() #Array shape = (n,24) columns = saved as movies_meta_headers variable below. Noteworthy cols: 3 = genres, 6 = imdb_id
movies_meta_headers = movies_meta.columns.tolist()

test_movies_subtitles_np = test_movies_subtitles.to_numpy() #Array shape = (n,4) columns = start time of passage, end time, text, imdb movie ID
test_movies_meta_np = test_movies_meta.to_numpy() #Array shape = (n,24) columns = saved as movies_meta_headers variable below. Noteworthy cols: 3 = genres, 6 = imdb_id

#Counter object below len = (4667) of all the unique imdb movie ID's
number_of_unique_movies = Counter(full_movies_subtitles_np[:,-1])

#Unique genres
number_of_unique_genres = Counter(movies_meta_np[:,3])

#passage = the individual line from the related subtitle file
passages = full_movies_subtitles_np[:,-2]

#%% CREATING DATA, DATA_DICT, DATA_DICT2 AND UNIQUE_GENRES
data_train_split = full_movies_subtitles_np[:,-2:]
extra_cols = np.zeros((data_train_split.shape[0], 1) )
data_train_split = np.hstack((data_train_split,extra_cols))

data_test_split = test_movies_subtitles_np[:,-2:]
extra_cols = np.zeros((data_test_split.shape[0], 1) )
data_test_split = np.hstack((data_test_split,extra_cols))

genres = []
for i in tqdm(range(movies_meta_np.shape[0])):
    genres_col = movies_meta['genres']
    genres_rc = genres_col.iloc[i]
    list_of_dicts = json.loads(genres_rc.replace("'", "\""))
    for dictionary in list_of_dicts:
        genres.append(dictionary['name'])

unique_genres_freq = Counter(genres)
genres = set(genres)
genre_to_idx = dict(zip(genres, range(-20,0)))

#DATA
for i in tqdm(range(data_train_split.shape[0])):
    test = data_train_split[i,1]
    genres = movies_meta.iloc[data_train_split[i,1] == movies_meta_np[:,6]]['genres']
    genres = genres.apply(eval)
    data_train_split[i,2] = []
    for item in genres:
        for dictionary in item:
                for key,value in dictionary.items():
                    if key == 'name':
                        genre = value
                        data_train_split[i,2].append(genre)

np.save('data_train_split.npy', data_train_split)
data_train_split = np.load('data_train_split.npy', allow_pickle=True) #Array shape = (n,3) cols: subtitle line, imdb_id, genres

#DATA DICT2 (/data dict)
data_dict2 = {}
for key, value in tqdm(number_of_unique_movies.items()): #COMMENT OUT WHEN THE FILES HAVE BEEN RUN BEFORE
    data_dict2.setdefault(key, )
    subtitle_lines = data_train_split[:, 0][data_train_split[:,1] == key].tolist()
    genres = data_train_split[:,2][np.argmax(data_train_split[:,1] == key)]
    data_dict2[key] = genres,subtitle_lines #remove genres to get data_dict

np.save('train_data_dict2.npy', data_dict2)
data_dict2 = np.load('train_data_dict2.npy', allow_pickle=True).item()
#Preventing errors later on by removing the nan key(s)
keys_to_delete = []
for key in tqdm(data_dict2):
    if key == "nan" or (isinstance(key, float) and math.isnan(key)):
        keys_to_delete.append(key)

for key in keys_to_delete:
    del data_dict2[key]

#TEST DATA DICT2 (/data dict)
test_data_dict2 = {}
for key, value in tqdm(number_of_unique_movies.items()): #COMMENT OUT WHEN THE FILES HAVE BEEN RUN BEFORE
    test_data_dict2.setdefault(key, )
    subtitle_lines = data_test_split[:, 0][data_test_split[:,1] == key].tolist()
    genres = data_test_split[:,2][np.argmax(data_test_split[:,1] == key)]
    test_data_dict2[key] = genres,subtitle_lines #remove genres to get data_dict

np.save('test_data_dict2.npy', test_data_dict2)
test_data_dict2 = np.load('test_data_dict2.npy', allow_pickle=True).item()
#Preventing errors later on by removing the nan key(s)
keys_to_delete = []
for key in tqdm(test_data_dict2):
    if key == "nan" or (isinstance(key, float) and math.isnan(key)):
        keys_to_delete.append(key)

for key in keys_to_delete:
    del test_data_dict2[key]

# #UNIQUE_GENRES
genres = np.concatenate(data_train_split[:, 2])  # Flatten the genres column into a 1D array
unique_genres = set(genres)  # Get unique genres using set()
np.save('unique_genres.npy', unique_genres)
#UNCOMMENT ABOVE WHEN NEEDED
unique_genres = np.load('unique_genres.npy', allow_pickle=True).item()

# unique_genres_dict = dict(zip(list(unique_genres), range(-20,0))) #Use this to reference the column of what genre is required in 'dataset'


#%% START OF ALGORITHM

#3. unique_genres = list of genres
# print(unique_genres) 

#4. Set thresh_low and thresh_upp
thresh_low_vals = [100,500,1000]
thresh_upp = 10000
full_reports_list = [] #Used for storing results
for thresh_low in tqdm(thresh_low_vals):

    #5. Preprocess subtitle files (they've been tokenised here too, even though the paper does it in 6.)
    preprocessed_data_dict2 = {} #COMMENT OUT WHEN THE FILES HAVE BEEN RUN BEFORE
    for key in tqdm(data_dict2.keys()):
        subtitles = data_dict2[key][1]
        tokens = preprocessing(subtitles)
        preprocessed_data_dict2[key] = data_dict2[key][0], tokens
    preprocessed_data_dict2_npy = "train_preprocessed_data_dict2.npy"
    np.save(preprocessed_data_dict2_npy, preprocessed_data_dict2) #COMMENT OUT WHEN THE FILES HAVE BEEN RUN BEFORE
    preprocessed_data_dict2 = np.load(preprocessed_data_dict2_npy, allow_pickle=True).item()

    #5.2 - TEST VERSION
    test_preprocessed_data_dict2 = {} #COMMENT OUT WHEN THE FILES HAVE BEEN RUN BEFORE
    for key in tqdm(test_data_dict2.keys()):
        subtitles = test_data_dict2[key][1]
        tokens = preprocessing(subtitles)
        test_preprocessed_data_dict2[key] = test_data_dict2[key][0], tokens
    test_preprocessed_data_dict2_npy = "test_preprocessed_data_dict2.npy"
    np.save(test_preprocessed_data_dict2_npy, preprocessed_data_dict2) #COMMENT OUT WHEN THE FILES HAVE BEEN RUN BEFORE
    preprocessed_data_dict2 = np.load(test_preprocessed_data_dict2_npy, allow_pickle=True).item()


    #6. Create dictionary D of word/frequencies
    D_dict = defaultdict(int)
    for key, items in tqdm(preprocessed_data_dict2.items()): #COMMENT OUT WHEN THE FILES HAVE BEEN RUN BEFORE
        tokens = items[1]
        for token in tokens:
            D_dict[token] += 1
    D_dict = dict(D_dict)

    # #7. Sort dict in order of word frequencies
    D_dict = OrderedDict(sorted(D_dict.items(), key=lambda item: item[1], reverse=True)) #COMMENT OUT WHEN THE FILES HAVE BEEN RUN BEFORE
    D_dict_npy = "test_D_dict.npy"
    # np.save(D_dict_npy, D_dict) #COMMENT OUT WHEN THE FILES HAVE BEEN RUN BEFORE
    D_dict = np.load(D_dict_npy, allow_pickle=True).item()

    #8. Create feature vector, F
    F = []
    for key, value in tqdm(D_dict.items()):
        if value <= thresh_upp:
            if value >= thresh_low:
                F.append(key)
    F_array = np.array(F)
                
    #Create feature matrix for selectbestK / dataset
    # print("number of words/features", len(F))
    # print("number of unique genres", len(unique_genres))

    #Array needs to be size(4666, 5537[5517+20]) = (len(f)[=number of words in feature set] + len(unique_genres)[=number of genres]
    #%%
    #BUILD DATASET - ALL GENRES
    #WHERE TEST DATA CAN BE INSERTED, NEED TO MAKE A DATA_DICT2 FOR THE TEST DATA, THEN REINSERT HERE.

    cols = F + list(unique_genres)
 
    # Prepare a list to collect data rows
    rows_list = []
    #The following method is more efficient than using .iloc to populate the dataset DataFrame as it
    #populates it in a batch.
    for imdb_id in tqdm(test_data_dict2.keys()):
        tokens = preprocessed_data_dict2[imdb_id][1]
        genres = preprocessed_data_dict2[imdb_id][0]
        frequencies = Counter(tokens)

        # Prepare a dictionary to collect row data - also using a dictionary means when it is converted
        #to a DataFrame the keys will become the columns so we can reindex it to match dataset and maintain the order
        row_data = defaultdict(int)  # Default frequency is 0 when using a defaultdict(int)

        for feature in F:
            # Set the frequency value directly in the right column so it maintains the same order each iteration
            row_data[feature] = frequencies.get(feature, 0)

        for genre in unique_genres:
            # Set the genre value directly in the right column so it maintains the same order each iteration
            row_data[genre] = 1 if genre in genres else 0

        rows_list.append(row_data)

    # Convert rows_list to a dataframe with the correct columns
    dataset = pd.DataFrame(rows_list, index=test_data_dict2.keys()).reindex(columns=cols).fillna(0)
                
    dataset_pkl = "test_datasetpkl_{}.pkl".format(thresh_low)
    dataset.to_pickle(dataset_pkl) #COMMENT OUT WHEN THE FILES HAVE BEEN RUN BEFORE
    dataset = pd.read_pickle(dataset_pkl)

    #%% REMOVE ONCE TESTING COMPLETE
    #TEMPORARILY SLICE DATASET FOR TESTING PURPOSES:
    dataset = dataset.iloc[:200]
    #%%
    #9. Create final feature vector, Final_F
    features = dataset.iloc[:, :-20].astype('int32')  #Slice all columns except the last 20 (features)
    targets = dataset.iloc[:, -20:].astype('int32')  #Slice the last 20 columns (target)

    best_k_features = [50,100,200,300,500]

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

    #%% LOGISTIC REGRESSION function
    #BUILD DATASET & MODEL TRAINING - Action, Fantasy, Horror, Romance, War (same as paper except no Sports)
    def logisticregression(Final_F_df, targets, genres):
        algorithm = "LR"
        model = LogisticRegression(penalty="l2", max_iter=10000)
        k=10
        kf = KFold(n_splits=k)

        Final_F_df = Final_F_df.reset_index().rename(columns={'index':'original_index'})
        targets = targets.reset_index().rename(columns={'index':'original_index'})

        # Shuffle the dataframes
        Final_F_df_shuffled = Final_F_df.sample(frac=1.0, random_state=22)  # Shuffle dataset before split

        # Reshuffle the targets to match the features
        targets_shuffled = targets.set_index('original_index').loc[Final_F_df_shuffled['original_index']] 

        # Reset the index of the shuffled dataframe, and remove 'original_index' column after using it for alignment
        Final_F_df_shuffled.set_index('original_index', inplace=True) 
        
        #Store all predictions and truths for calculate metrics
        y_true_all = []
        y_pred_all = []

        # For storing each classification report for each fold
        report_each_fold = []

        for train_index, test_index in tqdm(kf.split(Final_F_df_shuffled)):
            X_train, X_test = Final_F_df_shuffled.iloc[train_index], Final_F_df_shuffled.iloc[test_index]
            y_train, y_test = targets_shuffled.iloc[train_index], targets_shuffled.iloc[test_index]

            #OVERSAMPLING
            # X_train, y_train = over_sampler(X_train, y_train,'War', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=4.0)
            # X_train, y_train = over_sampler(X_train, y_train,'Fantasy', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=3.0)
            #UNDERSAMPLING
            # X_train, y_train = under_sampler(X_train, y_train, 'Action', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            # X_train, y_train = under_sampler(X_train, y_train, 'Romance', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)

            #OVERSAMPLING & UNDERSAMPLING (WORKS) - Only possible because I manually implement the k-fold cross val rather than using cross_val_predict
            # X_train, y_train = over_sampler(X_train, y_train,'War', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=2.5)
            # X_train, y_train = over_sampler(X_train, y_train,'Fantasy', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=2.0)
            # X_train, y_train = under_sampler(X_train, y_train, 'Action', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            # X_train, y_train = under_sampler(X_train, y_train, 'Romance', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)

            clf = MultiOutputClassifier(model)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            current_fold_report = pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True, target_names=genres))
            report_each_fold.append(current_fold_report)

            y_true_all.append(y_test.to_numpy())
            y_pred_all.append(y_pred)

        report_all_folds = pd.concat(report_each_fold)
        report_folds_average = compute_metrics(y_true_all, y_pred_all, genres)
        report_folds_average_df = pd.DataFrame(report_folds_average)  
        report_folds_average_df.index = "{}_".format(algorithm) + report_folds_average_df.index.astype(str) + "_{}_{}".format(thresh_low, best_k)    
        full_reports_list.append(report_folds_average_df)

    #%% DECISION TREE function
    def decisiontree(Final_F_df, targets, genres):
        algorithm = "DT"
        model = DecisionTreeClassifier(criterion='gini', splitter='best',max_depth=None, min_samples_split=2) #All the default values used for now
        k=10
        kf = KFold(n_splits=k)
        
        Final_F_df = Final_F_df.reset_index().rename(columns={'index':'original_index'})
        targets = targets.reset_index().rename(columns={'index':'original_index'})

        # Shuffle the dataframes
        Final_F_df_shuffled = Final_F_df.sample(frac=1.0, random_state=22)  # Shuffle dataset before split

        # Reshuffle the targets to match the features
        targets_shuffled = targets.set_index('original_index').loc[Final_F_df_shuffled['original_index']] 

        # Reset the index of the shuffled dataframe, and remove 'original_index' column after using it for alignment
        Final_F_df_shuffled.set_index('original_index', inplace=True) 

        #Store all predictions and truths for calculate metrics
        y_true_all = []
        y_pred_all = []

        # For storing each classification report for each fold
        report_each_fold = []

        for train_index, test_index in tqdm(kf.split(Final_F_df_shuffled)):
            X_train, X_test = Final_F_df_shuffled.iloc[train_index], Final_F_df_shuffled.iloc[test_index]
            y_train, y_test = targets_shuffled.iloc[train_index], targets_shuffled.iloc[test_index]

            #OVERSAMPLING
            # X_train, y_train = over_sampler(X_train,y_train,'War',isolate_col=True, dont_duplicate_these_genres=['Action','Romance'],duplication_ratio=4.0)
            # X_train, y_train = over_sampler(X_train,y_train,'Fantasy',isolate_col=True, dont_duplicate_these_genres=['Action','Romance'],duplication_ratio=3.0)
            #UNDERSAMPLING
            # X_train, y_train = under_sampler(X_train, y_train, 'Action', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            # X_train, y_train = under_sampler(X_train, y_train, 'Romance', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)

            #OVERSAMPLING & UNDERSAMPLING
            # X_train, y_train = over_sampler(X_train, y_train,'War', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=2.5)
            # X_train, y_train = over_sampler(X_train, y_train,'Fantasy', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=2.0)
            # X_train, y_train = under_sampler(X_train, y_train, 'Action', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            # X_train, y_train = under_sampler(X_train, y_train, 'Romance', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)

            clf = MultiOutputClassifier(model)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            current_fold_report = pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True, target_names=genres))
            report_each_fold.append(current_fold_report)

            y_true_all.append(y_test.to_numpy())
            y_pred_all.append(y_pred)

        report_all_folds = pd.concat(report_each_fold)
        report_folds_average = compute_metrics(y_true_all, y_pred_all, genres)
        report_folds_average_df = pd.DataFrame(report_folds_average)    
        report_folds_average_df.index = "{}_".format(algorithm) + report_folds_average_df.index.astype(str) + "_{}_{}".format(thresh_low, best_k)    
        full_reports_list.append(report_folds_average_df)

    #%% KNN function
    def knn(Final_F_df, targets, genres):
        algorithm = "KNN"
        k=10 #kNN k=10 and kfold k=10
        model = MLkNN(k=k)
        kf = KFold(n_splits=k)

        Final_F_df = Final_F_df.reset_index().rename(columns={'index':'original_index'})
        targets = targets.reset_index().rename(columns={'index':'original_index'})

        # Shuffle the dataframes
        Final_F_df_shuffled = Final_F_df.sample(frac=1.0, random_state=22)  # Shuffle dataset before split

        # Reshuffle the targets to match the features
        targets_shuffled = targets.set_index('original_index').loc[Final_F_df_shuffled['original_index']] 

        # Reset the index of the shuffled dataframe, and remove 'original_index' column after using it for alignment
        Final_F_df_shuffled.set_index('original_index', inplace=True) 

        #Store all predictions and truths for calculate metrics
        y_true_all = []
        y_pred_all = []

        # For storing each classification report for each fold
        report_each_fold = []

        for train_index, test_index in tqdm(kf.split(Final_F_df_shuffled)):
            X_train, X_test = Final_F_df_shuffled.iloc[train_index], Final_F_df_shuffled.iloc[test_index].to_numpy()
            y_train, y_test = targets_shuffled.iloc[train_index], targets_shuffled.iloc[test_index].to_numpy()

            #OVERSAMPLING
            # X_train, y_train = over_sampler(X_train,y_train,'War',isolate_col=True, dont_duplicate_these_genres=['Action','Romance'],duplication_ratio=4.0)
            # X_train, y_train = over_sampler(X_train,y_train,'Fantasy',isolate_col=True, dont_duplicate_these_genres=['Action','Romance'],duplication_ratio=3.0)
            #UNDERSAMPLING
            # X_train, y_train = under_sampler(X_train, y_train, 'Action', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            # X_train, y_train = under_sampler(X_train, y_train, 'Romance', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)

            #OVERSAMPLING & UNDERSAMPLING
            # X_train, y_train = over_sampler(X_train, y_train,'War', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=2.5)
            # X_train, y_train = over_sampler(X_train, y_train,'Fantasy', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=2.0)
            # X_train, y_train = under_sampler(X_train, y_train, 'Action', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            # X_train, y_train = under_sampler(X_train, y_train, 'Romance', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)

            X_train, y_train = X_train.to_numpy(), y_train.to_numpy() #Needs to be a numpy array for knn

            clf = model
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            current_fold_report = pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True, target_names=genres))
            report_each_fold.append(current_fold_report)

            y_true_all.append(y_test)
            y_pred_all.append(y_pred)

        report_all_folds = pd.concat(report_each_fold)
        report_folds_average = compute_metrics(y_true_all, y_pred_all, genres)
        report_folds_average_df = pd.DataFrame(report_folds_average)    
        report_folds_average_df.index = "{}_".format(algorithm) + report_folds_average_df.index.astype(str) + "_{}_{}".format(thresh_low, best_k)    
        full_reports_list.append(report_folds_average_df)


    #%% MULTINOMIAL NAIVE BAYES function
    def multinomial_naive_bayes(Final_F_df, targets, genres):
        algorithm = "MNB"
        model = MultinomialNB()
        k=10 #kfold k=10
        kf = KFold(n_splits=k)

        Final_F_df = Final_F_df.reset_index().rename(columns={'index':'original_index'})
        targets = targets.reset_index().rename(columns={'index':'original_index'})

        # Shuffle the dataframes
        Final_F_df_shuffled = Final_F_df.sample(frac=1.0, random_state=22)  # Shuffle dataset before split

        # Reshuffle the targets to match the features
        targets_shuffled = targets.set_index('original_index').loc[Final_F_df_shuffled['original_index']] 

        # Reset the index of the shuffled dataframe, and remove 'original_index' column after using it for alignment
        Final_F_df_shuffled.set_index('original_index', inplace=True) 
        
        #Store all predictions and truths for calculate metrics
        y_true_all = []
        y_pred_all = []

        # For storing each classification report for each fold
        report_each_fold = []

        for train_index, test_index in tqdm(kf.split(Final_F_df_shuffled)):
            X_train, X_test = Final_F_df_shuffled.iloc[train_index], Final_F_df_shuffled.iloc[test_index]
            y_train, y_test = targets_shuffled.iloc[train_index], targets_shuffled.iloc[test_index]

            #OVERSAMPLING
            # X_train, y_train = over_sampler(X_train,y_train,'War',isolate_col=True, dont_duplicate_these_genres=['Action','Romance'],duplication_ratio=4.0)
            # X_train, y_train = over_sampler(X_train,y_train,'Fantasy',isolate_col=True, dont_duplicate_these_genres=['Action','Romance'],duplication_ratio=3.0)
            #UNDERSAMPLING
            # X_train, y_train = under_sampler(X_train, y_train, 'Action', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            # X_train, y_train = under_sampler(X_train, y_train, 'Romance', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)

            #OVERSAMPLING & UNDERSAMPLING
            # X_train, y_train = over_sampler(X_train, y_train,'War', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=2.5)
            # X_train, y_train = over_sampler(X_train, y_train,'Fantasy', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=2.0)
            # X_train, y_train = under_sampler(X_train, y_train, 'Action', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            # X_train, y_train = under_sampler(X_train, y_train, 'Romance', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)

            clf = MultiOutputClassifier(model)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            current_fold_report = pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True, target_names=genres))
            report_each_fold.append(current_fold_report)

            y_true_all.append(y_test.to_numpy())
            y_pred_all.append(y_pred)

        report_all_folds = pd.concat(report_each_fold)
        report_folds_average = compute_metrics(y_true_all, y_pred_all, genres)
        report_folds_average_df = pd.DataFrame(report_folds_average)    
        report_folds_average_df.index = "{}_".format(algorithm) + report_folds_average_df.index.astype(str) + "_{}_{}".format(thresh_low, best_k)    
        full_reports_list.append(report_folds_average_df)

    #%% SVM function
    def svm(Final_F_df, targets, genres):
        algorithm = "SVM"
        model = SVC()
        k=10 #kfold k=10
        kf = KFold(n_splits=k)

        Final_F_df = Final_F_df.reset_index().rename(columns={'index':'original_index'})
        targets = targets.reset_index().rename(columns={'index':'original_index'})

        # Shuffle the dataframes
        Final_F_df_shuffled = Final_F_df.sample(frac=1.0, random_state=22)  # Shuffle dataset before split

        # Reshuffle the targets to match the features
        targets_shuffled = targets.set_index('original_index').loc[Final_F_df_shuffled['original_index']] 

        # Reset the index of the shuffled dataframe, and remove 'original_index' column after using it for alignment
        Final_F_df_shuffled.set_index('original_index', inplace=True) 

        #Store all predictions and truths for calculate metrics
        y_true_all = []
        y_pred_all = []

        # For storing each classification report for each fold
        report_each_fold = []

        for train_index, test_index in tqdm(kf.split(Final_F_df_shuffled)):
            X_train, X_test = Final_F_df_shuffled.iloc[train_index], Final_F_df_shuffled.iloc[test_index]
            y_train, y_test = targets_shuffled.iloc[train_index], targets_shuffled.iloc[test_index]

            #OVERSAMPLING
            # X_train, y_train = over_sampler(X_train,y_train,'War',isolate_col=True, dont_duplicate_these_genres=['Action','Romance'],duplication_ratio=4.0)
            # X_train, y_train = over_sampler(X_train,y_train,'Fantasy',isolate_col=True, dont_duplicate_these_genres=['Action','Romance'],duplication_ratio=3.0)
            #UNDERSAMPLING
            # X_train, y_train = under_sampler(X_train, y_train, 'Action', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            # X_train, y_train = under_sampler(X_train, y_train, 'Romance', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)

            #OVERSAMPLING & UNDERSAMPLING
            # X_train, y_train = over_sampler(X_train, y_train,'War', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=2.5)
            # X_train, y_train = over_sampler(X_train, y_train,'Fantasy', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=2.0)
            # X_train, y_train = under_sampler(X_train, y_train, 'Action', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            # X_train, y_train = under_sampler(X_train, y_train, 'Romance', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)

            clf = clf = MultiOutputClassifier(model)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            current_fold_report = pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True, target_names=genres))
            report_each_fold.append(current_fold_report)

            y_true_all.append(y_test.to_numpy())
            y_pred_all.append(y_pred)

        report_all_folds = pd.concat(report_each_fold)
        report_folds_average = compute_metrics(y_true_all, y_pred_all, genres)
        report_folds_average_df = pd.DataFrame(report_folds_average)    
        report_folds_average_df.index = "{}_".format(algorithm) + report_folds_average_df.index.astype(str) + "_{}_{}".format(thresh_low, best_k)    
        full_reports_list.append(report_folds_average_df)


    #%% NEURAL NET - MLP functions
    class MLP(nn.Module):
        def __init__(self, input_size, num_classes):
            super(MLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, num_classes),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.layers(x)

    def train_model(model, train_loader, criterion, optimizer, device):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

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
        
    def pytorch_mlp(Final_F_df, targets, genres):
        algorithm = "MLP"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        scaler = StandardScaler()
        Final_F_df = pd.DataFrame(scaler.fit_transform(Final_F_df.values), columns=Final_F_df.columns, index=Final_F_df.index)

        k=10 #kfold k=10
        kf = KFold(n_splits=k)
        Final_F_df = Final_F_df.reset_index().rename(columns={'index':'original_index'})
        targets = targets.reset_index().rename(columns={'index':'original_index'})

        # Shuffle the dataframes
        Final_F_df_shuffled = Final_F_df.sample(frac=1.0, random_state=22)  # Shuffle dataset before split

        # Reshuffle the targets to match the features
        targets_shuffled = targets.set_index('original_index').loc[Final_F_df_shuffled['original_index']] 

        # Reset the index of the shuffled dataframe, and remove 'original_index' column after using it for alignment
        Final_F_df_shuffled.set_index('original_index', inplace=True) 

        #Store all predictions and truths for calculate metrics
        y_true_all = []
        y_pred_all = []

        # For storing each classification report for each fold
        report_each_fold = []

        input_size = Final_F_df_shuffled.shape[1]
        num_classes = targets_shuffled.shape[1]

        model = MLP(input_size, num_classes).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epochs = 10

        for train_index, test_index in tqdm(kf.split(Final_F_df_shuffled)):
            X_train, X_test = Final_F_df_shuffled.iloc[train_index], Final_F_df_shuffled.iloc[test_index]
            y_train, y_test = targets_shuffled.iloc[train_index], targets_shuffled.iloc[test_index]

            #OVERSAMPLING
            # X_train, y_train = over_sampler(X_train,y_train,'War',isolate_col=True, dont_duplicate_these_genres=['Action','Romance'],duplication_ratio=4.0)
            # X_train, y_train = over_sampler(X_train,y_train,'Fantasy',isolate_col=True, dont_duplicate_these_genres=['Action','Romance'],duplication_ratio=3.0)
            #UNDERSAMPLING
            # X_train, y_train = under_sampler(X_train, y_train, 'Action', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            # X_train, y_train = under_sampler(X_train, y_train, 'Romance', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)

            #OVERSAMPLING & UNDERSAMPLING
            # X_train, y_train = over_sampler(X_train, y_train,'War', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=2.5)
            # X_train, y_train = over_sampler(X_train, y_train,'Fantasy', isolate_col=True, dont_duplicate_these_genres=['Action','Romance'], duplication_ratio=2.0)
            # X_train, y_train = under_sampler(X_train, y_train, 'Action', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            # X_train, y_train = under_sampler(X_train, y_train, 'Romance', isolate_col=False, dont_remove_these_genres=['Fantasy', 'War'], removal_ratio=0.5)
            
            train_dataset = TensorDataset(torch.from_numpy(X_train.values).float(), torch.from_numpy(y_train.values).float())
            test_dataset = TensorDataset(torch.from_numpy(X_test.values).float(), torch.from_numpy(y_test.values).float())
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32)

            for epoch in range(epochs):
                train_model(model, train_loader, criterion, optimizer, device)
            
            y_pred, y_test = evaluate_model(model, test_loader, device)
            current_fold_report = pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True, target_names=genres))
            report_each_fold.append(current_fold_report)

            y_true_all.append(y_test)
            y_pred_all.append(y_pred)


        report_all_folds = pd.concat(report_each_fold)
        report_folds_average = compute_metrics(y_true_all, y_pred_all, genres)
        report_folds_average_df = pd.DataFrame(report_folds_average)    
        report_folds_average_df.index = "{}_".format(algorithm) + report_folds_average_df.index.astype(str) + "_{}_{}".format(thresh_low, best_k)    
        full_reports_list.append(report_folds_average_df)

    #%% Start loops

    dataset = pd.read_pickle(dataset_pkl)

    genres_kept = ['Action','Fantasy','Horror','Romance','War']

    #Create final feature vector, Final_F
    filtered_dataset = dataset[(dataset['Action'] ==1) | (dataset['Fantasy'] ==1)| (dataset['Horror'] ==1) | (dataset['Romance'] ==1) | (dataset['War'] ==1)]
    features = dataset.iloc[:, :-20].astype('int32')
    features = features[features.index.isin(filtered_dataset.index)]  #Slice to obtain rows that are present in the filtered dataset
    targets = filtered_dataset.loc[:, genres_kept].astype('int32')  #Slice the columns of the genres kept (target) 

    #Start best_k loop
    for best_k in tqdm(best_k_features):
        selectkbest = SelectKBest(chi2, k=best_k)
        Final_F = selectkbest.fit_transform(features, targets)
        selected_features = features.columns[selectkbest.get_support()]
        Final_F_df = pd.DataFrame(Final_F, columns=selected_features, index=features.index)

        Final_F_df_pkl = "test_Final_F_df_{}_{}.pkl".format(thresh_low,best_k)
        # Final_F_df.to_pickle(Final_F_df_pkl)
        Final_F_df = pd.read_pickle(Final_F_df_pkl)

        genre_counts =  np.sum(targets, axis=0)
        # print(genre_counts)

        #FUNCTIONS
        logisticregression(Final_F_df, targets, genres=genres_kept)
        decisiontree(Final_F_df, targets, genres=genres_kept)
        knn(Final_F_df, targets, genres=genres_kept)
        multinomial_naive_bayes(Final_F_df, targets, genres=genres_kept)
        svm(Final_F_df, targets, genres=genres_kept)
        pytorch_mlp(Final_F_df, targets, genres=genres_kept)

full_reports_df = pd.concat(full_reports_list)
full_reports_df.to_csv('test_full_reports.csv')
print("Finished")


# %%
