
# generates the sequence of tokens and chars
# for input to the Model
import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
from preprocessing import *

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Input, Model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Concatenate
# import keras as k
import pickle
import operator
from keras_contrib.utils import save_load_utils
from MyModel import create_model
from MyModel import save_model
from keras_contrib.layers import CRF

from keras_contrib.utils import save_load_utils
from Utils import *

class Sequence(object):

	def __init__(self, words, sentences, tags = None, max_seq_len = 75, trained = False ):

		self.words = words
		self.tags = tags
		self.sentences = sentences
		self.max_seq_len = max_seq_len
		
		if words and tags:
			self.n_words = len(words)
			self.n_tags = len(tags)
		else:
			# verify Later
			self.n_words = 0
			self.n_tags = 0


 
	def load_dicts_from_disk(self):
		"""model is already trained """
		self.word2idx = save_load_word_idx("word2idx.pkl", load = True)
		self.idx2word = save_load_word_idx("idx2word.pkl", load = True)
		self.tag2idx = save_load_word_idx("tag2idx.pkl", load = True)
		self.idx2tag = save_load_word_idx("idx2tag.pkl" , load = True)

	def _generate_words_sequence(self):

		#//  TODO : check if the word is not found in sequence, append "UNK"(index=1) token instead

 		pass	

	def split_test_train(self, _test_size = 0.1, _shuffle = True):


		from sklearn.model_selection import train_test_split
		# split data into (test=90%,train=10%) percentage
		self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(self.X, self.y, test_size=_test_size, shuffle = _shuffle, random_state = 2018)
		self.X_char_tr, self.X_char_te, _, _ = train_test_split(self.X_char, self.y, test_size=_test_size, shuffle = _shuffle, random_state = 2018)

	def predictions_shuffle(self):

		self.X_te = self.X
		self.y_te = None
		self.X_char_te = self.X_char		


	def _generate_chars_sequence(self, _sentences = None, _trained = False):
		
		# create a vocabulary with all possible/unique chars
		self.chars = set([chars for word in self.words for chars in word])
		self.n_chars = len(self.chars)
		# print("n_chars::",self.n_chars) # 98
		self.max_len_chars = 10

		if not _trained:
			# create char2idx for converting chars as vector of integers to feed to LSTM
			self.char2idx = {char:i + 2 for i,char in enumerate(self.chars)}
			self.char2idx["ENDPAD"] = 0  # to ignore this by mask_zero = True
			self.char2idx["UNK"] = 1
			save_load_word_idx("char2idx.pkl", self.char2idx, save = True)
		else:
			self.char2idx = save_load_word_idx("char2idx.pkl", load = True)	
	
		# vice versa
		self.idx2char = {i:char for char,i in self.char2idx.items()}

		if not _sentences:
			sentences = self.sentences
		else:
			sentences = _sentences
		# generate char_sequence for input to model 
		self.X_char = []
		for sentence in sentences:
			sent_seq = []
			# max_len = 75
			for i in range(self.max_seq_len):
				word_seq = []
				# char sequence for words
				for j in range(self.max_len_chars):
					try:
						# chars of specific sentence of i
						word_seq.append(self.char2idx.get(sentence[i][j], 1)) 
					except:  # if char-sequence is out of range , pad it with "PAD" tag
						word_seq.append(self.char2idx.get("ENDPAD"))

				sent_seq.append(word_seq)
			# append sentence sequences as character-by-character to X_char for Model input
			self.X_char.append(np.array(sent_seq))
		# print(self.X_char[:2])


	def generate_input_sequence(self, _sentences, trained = False):
		# convert sequence of sentences into corresponding int vectors
		self.X = [[self.word2idx.get(w,1) for w in s] for s in _sentences]

		# max length of sequence/sentence	
		# print("Max length of sequence(len(sentence)):", max([len(x) for x in self.X]) )

		# add padding for same length i.e, max_len= 75 with "0" value
		self.X = pad_sequences(maxlen = self.max_seq_len, sequences = self.X,\
								truncating= 'post', padding ='post', value=0 )

		self._generate_chars_sequence(_sentences = _sentences, _trained= trained)


	def generate_output_sequence(self):

		self.y = [[self.tag2idx[w[2]] for w in s] for s in self.sentences]
		self.y = pad_sequences(maxlen=self.max_seq_len, sequences=self.y, padding="post",
								truncating="post", value=self.tag2idx["ENDPAD"])

		# y = [to_categorical(i,num_classes = n_tags + 1) for i in y] 
		# y: class vector to be converted into a matrix (integers from 0 to num_classes).
		# num_classes: total number of classes.
		self.y = [to_categorical(i, num_classes = self.n_tags + 1) for i in self.y]



	# def generate_sequence(self):
 # 		self._generate_words_sequence()
 # 		self._generate_chars_sequence()

	def generate_dicts_from_vocab(self):
 		""" model is not trained """
 		self.word2idx = {w: i + 2 for i, w in enumerate(self.words)}
 		self.word2idx["ENDPAD"] = 0
 		self.word2idx["UNK"] = 1
 		self.idx2word = {i : w for w, i in self.word2idx.items()}
 		self.tag2idx = {t: i + 1 for i, t in enumerate(self.tags)}
 		self.tag2idx["ENDPAD"] = 0
 		self.idx2tag = {i: t for t, i in self.tag2idx.items()}

 		save_load_word_idx("word2idx.pkl", word2idx = self.word2idx , save = True)
 		save_load_word_idx("idx2word.pkl", word2idx = self.idx2word , save = True)
 		save_load_word_idx("tag2idx.pkl", word2idx = self.tag2idx , save = True)
 		save_load_word_idx("idx2tag.pkl", word2idx = self.idx2tag , save = True)