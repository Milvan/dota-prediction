# dota-prediciton
Predicting dota games with machine learning

Acamedic paper in the file evaluation-of-expert-knowledge.pdf

# How to run
Download data set from https://www.kaggle.com/devinanzelmo/dota-2-matches

Put the extracted folder into the root folder of this repo

Run extractor.ipynb or extractor_expert.ipynb to extract the data to the correct format

Run lstm.ipynb to train and test an LSTM network

Run full-connected.ipynb to train and test an MLP

# Perform random search of hyper parameters
Run the lstm.py or the full-connected.py to test combination of parameters

The scripts will produce a .csv file in /tmp with all values tested and their performance


