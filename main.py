#!/usr/bin/env python

from __future__ import print_function
from argparse import ArgumentParser
from preprocessing import get_tweets, normalize_tweets, get_max_len
from utils import text_to_indices, read_glove_vecs
from sentiment_model import sentiment_model


GLOVE_FILE = 'glove_vectors/glove.twitter.27B.50d.txt'
TRAIN_PATH = 'data/training.1600000.processed.noemoticon.csv'
TEST_PATH = 'data/testdata.manual.2009.06.14.csv'
NUM_EPOCHS = 10
BATCH_SIZE = 128
OPTIMIZER = 'adam'
LEARNING_RATE = 0.001


def build_parser():
	"""
	Building the parser
	"""
	parser = ArgumentParser()
	parser.add_argument('--train-path', type=str, 
		dest='train_path', help='path to training tweets folder',
		metavar='TRAIN_PATH', default=TRAIN_PATH)
	parser.add_argument('--test-path', type=str, 
		dest='test_path', help='test tweets folder',
		metavar='TEST_PATH', default=TEST_PATH)
	parser.add_argument('--optimizer', type=str, 
		dest='optimizer', help='choice of optimizer', 
		metavar='OPTIMIZER', default=OPTIMIZER)
	parser.add_argument('--epochs', type=int, 
		dest='epochs', help='num epochs',
		metavar='EPOCHS', default=NUM_EPOCHS)
	parser.add_argument('--batch-size', type=int, 
		dest='batch_size', help='batch size',
		metavar='BATCH_SIZE', default=BATCH_SIZE)
	parser.add_argument('--learning-rate', type=int, 
		dest='learning_rate', help='learning rate',
		metavar='LEARNING_RATE', default=LEARNING_RATE)
	return parser


def check_opts(opts):
	"""
	Checks if values entered are OK
	"""
    exists(opts.train_path, "train path not found!")
    exists(opts.test, "test path not found!")
    exists(opts.optimizer, "optimizer not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.learning_rate > 0


def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    train_path = options.train_path
    test_path = options.test_path
    optimizer = options.optimizer
    epochs = options.epochs
    batch_size = options.batch_size
    learning_rate = options.learning_rate

    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(GLOVE_FILE)
    X_train, y_train = get_tweets(train_path)
    X_train_norm = normalize_tweets(X_train)
    max_len = get_max_len(X_train_norm)
    input_shape = (max_len,)
    X_train_indices = text_to_indices(X_train_norm)

    model = sentiment_model(input_shape, word_to_vec_map, word_to_index)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print('Starting the training...')
	model.fit(X_train_indices, y_train, epochs=epochs, batch_size=batch_size, 
		validation_split=0.1, shuffle=True)
	print('Training finished! Now saving model in model directory...')
	if not os.path.exists('model'):
    	os.makedirs('model')
	model.save('model/sentiment_model.h5')
	print('Model saved!')


if __name__ == '__main__':
    main()





