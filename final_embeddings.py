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
    

#%% PREPARE DATA
full_movies_subtitles = pd.read_csv("movies_subtitles.csv", sep=',', header = 0)
movies_meta = pd.read_csv("movies_meta.csv", sep=',', header = 0)
full_movies_subtitles_np = full_movies_subtitles.to_numpy() #Array shape = (10358496,4) columns = start time of passage, end time, text, imdb movie ID
movies_meta_np = movies_meta.to_numpy() #Array shape = (4690,24) columns = saved as movies_meta_headers variable below. Noteworthy cols: 3 = genres, 6 = imdb_id
movies_meta_headers = movies_meta.columns.tolist()
data = full_movies_subtitles_np[:,-2:]
extra_cols = np.zeros((data.shape[0], 20) ) 
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

#DATA
for i in tqdm(range(data.shape[0])):
    genres = movies_meta.iloc[data[i,1] == movies_meta_np[:,6]]['genres']
    genres = genres.apply(eval)
    for item in genres:
        for dictionary in item:
                for key,value in dictionary.items():
                    if key == 'name':
                        idx = genre_to_idx[value]
                        data[i,idx] = 1
                
np.save('data_embeddings.npy', data)
data = np.load('data_embeddings.npy', allow_pickle=True) #Array shape = (10358496, 22) cols: subtitle line, imdb_id, genres (20 genres 1-hot encoded)

#%% EMBEDDINGS
# The embeddings arrays below consist of one line from the subtitle track, the imdb_id of the movie, then 1-hot encoding of which genre(s) it belongs to.
corpus_raw = data[:,0]


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

for line in tqdm(corpus_raw): #UNSAVE IF STARTING FROM FRESH
    line = preprocessing(line)
    corpus.append(line)
# Save preprocessed corpus
with open("corpus.json", "w") as corpus_json: #UNSAVE IF STARTING FROM FRESH
    json.dump(corpus, corpus_json)
with open("corpus.json", "r") as corpus_json: #UNSAVE 
    corpus = json.load(corpus_json)

vocabulary =  create_vocabulary(corpus)
#Save preprocessed vocabulary
with open("vocabulary.json", "w") as vocabulary_json: #UNSAVE IF STARTING FROM FRESH
    json.dump(vocabulary, vocabulary_json)
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

int_words = [vocab_to_int[word] for words_list in corpus for word in words_list] # flattened_list = [item for sublist in nested_list for item in sublist]
print(int_words[:30])

threshold = 1e-3
word_counts = Counter(int_words)
#print(list(word_counts.items())[0])  # dictionary of int_words, how many times they appear

# total_count = len(int_words)
total_count = sum(word_counts.values())
freqs = {word: count/total_count for word, count in word_counts.items()}
p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts} # discard some frequent words, according to the subsampling equation

# create a new list of words for training
train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

print(train_words[:30])

def get_target(words, idx, window_size=5):
    """Get a list of words in a window around an index. """
    
    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx+1:stop+1]
    
    return list(target_words)

def get_batches(words, batch_size, window_size=5):
    """ Generate word batches as a tuple (inputs, targets) """
    
    n_batches = len(words)//batch_size
    
    # only full batches
    words = words[:n_batches*batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y

int_text = [i for i in range(20)]
x,y = next(get_batches(int_text, batch_size=4, window_size=5))
print('x\n', x)
print('y\n', y)


def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        Inputs:
        Embedding (PyTorch embedding)
        
    """
    
    # Here we're calculating the cosine similarity between some random words and 
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.
    
    # sim = (a . b) / |a||b|
    
    embed_vectors = embedding.weight
    
    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    
    # pick N words from our ranges (0,window) and (1000, 1000+window). lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)
    
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
        
    return valid_examples, similarities


class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        
        # define embedding layers for input and output words
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        
        # Initialize embedding tables with uniform distribution, this should help with convergence
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
        
    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors
    
    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors
    
    def forward_noise(self, batch_size, n_samples):
        """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        if self.noise_dist is None:
            # Sample words uniformly
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # Sample words from our noise distribution
        noise_words = torch.multinomial(noise_dist,
                                        batch_size * n_samples,
                                        replacement=True)
        
        device = "cuda" if model.out_embed.weight.is_cuda else "cpu"
        noise_words = noise_words.to(device)
        
        noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
        
        return noise_vectors

class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        
        batch_size, embed_size = input_vectors.shape
        
        # Input vectors should be a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        
        # Output vectors should be a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        
        # bmm = batch matrix multiplication
        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()
        
        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        # negate and sum correct and noisy log-sigmoid losses
        # return average batch loss
        return -(out_loss + noise_loss).mean()
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get our noise distribution
# Using word frequencies calculated earlier
word_freqs = np.array(sorted(freqs.values(), reverse=True))
unigram_dist = word_freqs/word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

# instantiating the model
embedding_dim = 300
model = SkipGramNeg(len(vocab_to_int), embedding_dim, noise_dist=noise_dist).to(device)
criterion = NegativeSamplingLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001) #LR = 0.1 in the paper

print_every = 1500
steps = 0
epochs = 12
batch_size = 1024 #16 used in paper
window_size = 5 #window size of 5 == context of 2

#%% SKIP-GRAM MODEL TRAINING
# train for 12 epochs
for e in tqdm(range(epochs)):
    
    # get our input, target batches
    for input_words, target_words in get_batches(train_words, batch_size=batch_size, window_size=window_size): 
        steps += 1
        inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # input, output, and noise vectors
        input_vectors = model.forward_input(inputs)
        output_vectors = model.forward_output(targets)
        noise_vectors = model.forward_noise(inputs.shape[0], 5)

        # negative sampling loss
        loss = criterion(input_vectors, output_vectors, noise_vectors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss stats
        if steps % print_every == 0:
            print("Epoch: {}/{}".format(e+1, epochs))
            print("Loss: ", loss.item()) # avg batch loss at this point in training
            valid_examples, valid_similarities = cosine_similarity(model.in_embed, device=device)
            _, closest_idxs = valid_similarities.topk(6)

            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
            print("...\n")

embeddings = model.in_embed.weight.to('cpu').data.numpy() # get embeddings[194261,300]
np.save('embeddings.npy', embeddings)
embeddings = np.load('embeddings.npy', allow_pickle=True)

#%% PLOT EMBEDDINGS
viz_words = 380
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

fig, ax = plt.subplots(figsize=(16, 16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
plt.show()

#%% SPLIT ORIGINAL DATASET (USED TO TRAIN EMBEDDINGS) -  FROM HERE BELOW, IS A DUPLICATE OF MLP_HISTOGRAM FILE.

"""
#Genres: Romance, Thriller, Action (,Drama, Comedy - added)

1.  I think I need to consolidate the subtitle files to represent the one subtitle file instead of individual lines at the momment.
3.  Map the words left in the data to the embeddings using the vocab_to_int dict and the embeddings vector
4.  Go from there & see where we're at.

Embeddings size of 300



"""

