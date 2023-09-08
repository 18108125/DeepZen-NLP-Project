# DeepZen

For our experiments we used a Windows PC with 64-bit Windows 10 installed.
PC specifications:
CPU: AMD Ryzen 5 3600XT 6-Core Processor               3.80 GHz
RAM: 32GB
GPU: NVIDIA MSI GeForce RTX 3060Ti Gaming X Trio 


1) Download both the movie_subtitles and movies_meta csv files from the website below.
https://www.kaggle.com/datasets/adiamaan/movie-subtitle-dataset?select=movies_subtitles.csv

2) Create the conda environment using the environment.yml file.

3) Download the files from this repository and be sure to run final_embeddings.py before final_mlp_histogram.py, as the mlp_histogram file uses the embeddings learned in the embeddings file.
