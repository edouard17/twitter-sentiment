#!/usr/bin/env python

from __future__ import print_function
import zipfile
import requests
import io
import os


def unzip_and_extract(zip_url, directory_to_extract_to, object_string):
	"""
	Function to unzip and extract files
	"""
	print('Downloading {} ...'.format(object_string))
	print("This might take a few minutes.")
	r = requests.get(zip_url)
	z = zipfile.ZipFile(io.BytesIO(r.content))
	os.mkdir(directory_to_extract_to)
	print("Extracting {} ...".format(object_string))
	z.extractall(directory_to_extract_to)
	print("{} successfully dowloaded!".format(object_string))


# Download GloVe vectors
glove_name = 'GloVe Vectors'
glove_url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
directory_to_extract_glove = 'glove_vectors'
unzip_and_extract(glove_url, directory_to_extract_glove, glove_name)

# Download Sentiment140 dataset
sentiment_name = 'Sentiment140 dataset'
dataset_url = 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
directory_to_extract_dataset = 'data'
unzip_and_extract(dataset_url, directory_to_extract_dataset, sentiment_name)
