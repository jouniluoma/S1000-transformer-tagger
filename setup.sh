#!/bin/bash

# Download NER model trained with S1000 dataset
wget https://a3s.fi/s1000/S1000-ner-model.tar.gz
tar -xvzf S1000-ner-model.tar.gz
rm S1000-ner-model.tar.gz

# Download example data
wget https://a3s.fi/s1000/database_sample.tsv.gz
mkdir data
mv database_sample.tsv.gz data/
gunzip data/database_sample.tsv.gz




