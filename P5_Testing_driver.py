
# https://www.depends-on-the-definition.com/guide-sequence-tagging-neural-networks-python/

# setting git for proxy 
#git config --global http.proxy http://proxyuser:proxypwd@proxy.server.com:8080
#git config --global --unset http.proxy
#git config --global --get http.proxy

# import tensorflow as tf

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
import sys, os
from keras_contrib.utils import save_load_utils
from MyModel import create_model
from MyModel import plot_history
from MyModel import draw_histogram
# from MyModel import load_model
from MyModel import save_model
from keras_contrib.layers import CRF

from keras_contrib.utils import save_load_utils
from Utils import *

# sys.path.append("C:\\Users\\Usman Ahmad\\Desktop\\NER Implementation")
# sys.path.append("../")

# parentPath = os.path.abspath("..")
# if parentPath not in sys.path:
#     sys.path.insert(0, parentPath)


trained = False





def main():

	data, words, tags, pos = load_dataset() # unique
	print("Data Loaded successfully..")


	n_words = len(words)   # total words in vocab
	n_tags = len(tags)	   # total tags in vocab
	n_pos = len(pos)
	max_len = 75 # length of each sequence/sentence

	getter = SentenceGetter(data)  # preprocessing.py
	# list of (word,POS,Tag)
	sentences = getter.sentences

	# sequence_obj = Sequence(words, tags, sentences)
	# Sequence(n_words, n_ta)

	# print("First sentence")
	# print(sentences[0])

	
	# if model is trained, load previous results
	if trained:
		assert trained == True, "Trained must be True"

		# load trained indices 
		word2idx = save_load_word_idx("word2idx.pkl", load = True)
		idx2word = save_load_word_idx("idx2word.pkl", load = True)
		tag2idx = save_load_word_idx("tag2idx.pkl", load = True)
		idx2tag = save_load_word_idx("idx2tag.pkl" , load = True)

	else:
		assert trained == False

		# save trained indices
		word2idx = {w: i + 2 for i, w in enumerate(words)}
		word2idx["ENDPAD"] = 0
		word2idx["UNK"] = 1

		idx2word = {i : w for w, i in word2idx.items()}

		tag2idx = {t: i + 1 for i, t in enumerate(tags)}
		tag2idx["ENDPAD"] = 0

		idx2tag = {i: t for t, i in tag2idx.items()}
		
		save_load_word_idx("word2idx.pkl", word2idx = word2idx , save = True)
		save_load_word_idx("idx2word.pkl", word2idx = idx2word , save = True)
		save_load_word_idx("tag2idx.pkl", word2idx = tag2idx , save = True)
		save_load_word_idx("idx2tag.pkl", word2idx = idx2tag , save = True)

	
	print("word2idx[\"demonstrators\"].................", word2idx["demonstrators"])


	# convert sequence of sentences into corresponding int vectors
	X = [[word2idx[w[0]] for w in s] for s in sentences]

	# max length of sequence/sentence	
	print("Max length of sequence(len(sentence)):", max([len(x) for x in X]) )

	# add padding for same length i.e, max_len= 75 with "0" value
	X = pad_sequences(maxlen = max_len, sequences = X,truncating= 'post', padding ='post', value=0 )


	y = [[tag2idx[w[2]] for w in s] for s in sentences]
	y = pad_sequences(maxlen=max_len, sequences=y, padding="post",truncating="post", value=tag2idx["ENDPAD"])


	# y = [to_categorical(i,num_classes = n_tags + 1) for i in y] 
	# y: class vector to be converted into a matrix (integers from 0 to num_classes).
	# num_classes: total number of classes.
	y = [to_categorical(i, num_classes = n_tags + 1) for i in y]


	# create a list with all possible chars
	chars = set([chars for word in words for chars in word])
	n_chars = len(chars)
	print("n_chars::",n_chars) # 98
	max_len_chars = 10

	# i, len_word = max([(i,len(word)) for i,word in enumerate(words)])

	_max, imax = -1, 0
	for i, w in enumerate(words):
		if len(w) > _max:
			_max, imax = len(w), i

	print("Actual max len chars(n_chars){} and word is {}:".format(_max,words[imax]))

	# create char2idx for converting chars as vector of integers to feed to LSTM
	char2idx = {char:i + 2 for i,char in enumerate(chars)}
	char2idx["ENDPAD"] = 0  # to ignore this by mask_zero = True
	char2idx["UNK"] = 1

	# vice versa
	idx2char = {i:char for char,i in char2idx.items()}

	# generate char_sequence for input to model 
	X_char = []
	for sentence in sentences:
		sent_seq = []
		# max_len = 75
		for i in range(max_len):
			word_seq = []
			# char sequence for words
			for j in range(max_len_chars):
				try:
					# chars of specific sentence of i
					word_seq.append(char2idx.get(sentence[i][0][j])) 
				except:  # if char-sequence is out of range , pad it with "PAD" tag
					word_seq.append(char2idx.get("ENDPAD"))

			sent_seq.append(word_seq)
		# append sentence sequences as character-by-character to X_char for Model input
		X_char.append(np.array(sent_seq))


	print('*'*50)
	print(X_char[:1])
	print("*"*50)
	print(X[:1])
	print('*'*50)
	print("shape of one X_char[0]: ", X_char[0].shape)
	print("shape of  X_char:{} ".format(np.array(X_char).shape))
	print("shape of  X:{} ".format(X.shape))


	from sklearn.model_selection import train_test_split
	# split data into (test=90%,train=10%) percentage
	X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, shuffle = True, random_state = 2018)
	X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, test_size=0.1, shuffle = True, random_state = 2018)

	print("shape of  X_char_tr:{} ".format(np.array(X_char_tr).shape))
	print("shape of  X_char_te:{} ".format(np.array(X_char_te).shape))
	print("shape of  y_tr:{} ".format(np.array(y_tr).shape))


	print("Reshaped X_char_tr:", np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_chars)).shape)
	
	# print("Reshaped y_tr:",np.array(y_tr).reshape(len(y_tr), max_len, 1).shape)		
	print("X_tr : ",X_tr.shape)

	# import sys
	# sys.exit(0)

	if trained:
		# model.evaluate(X_te, np.array(y_te), verbose=1)
		main2(X,X_te,X_char_te, y_te ,words = words, tags= tags, idx2word = idx2word, idx2tag= idx2tag, word_idx = word2idx)
		return

	model = create_model(max_len, n_words, n_tags, n_pos, max_len_chars, n_chars, _word2idx=word2idx)

	# second input to be fed like : model.fit([X_tr, second_input])
	# second_input_emb = np.array(X_pos[:len(X_tr)])
	# second_input_hot = np.array(X_pos[:len(X_tr)])

	# print(np.array(X_char_tr)[12])
	print("X_char_te")
	# print(np.array(X_char_te)[12])

	# score = model.evaluate([X_te, np.array(X_pos[len(X_tr):])], np.array(y_te), verbose=1)
	# #print accuracy
	# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

	# score = model.evaluate([X_te, np.array(X_pos[len(X_tr):])], np.array(y_te), verbose=1)
	# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

	# history = model.fit([X_tr, second_input_hot], np.array(y_tr),\
	#  batch_size=32, epochs=2, validation_split=0.1, verbose=1)
																#.reshape(len(y_tr),max_len,1)		
	# history = model.fit([X_tr, np.array(X_char_tr)], np.array(y_tr),\
	# batch_size=32, epochs=15, validation_split=0.1, verbose=1)

	# history = model.fit([X_word_tr,
 #                     np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
 #                    np.array(y_tr).reshape(len(y_tr), max_len, 1),
 #                    batch_size=32, epochs=10, validation_split=0.1, verbose=1)

	# history = model.fit([X_tr,\
                    # np.array(X_char_tr)],np.array(y_tr).reshape(len(y_tr),max_len, 1),\
                    # batch_size=32, epochs=10, validation_split=0.1, verbose=1)

	history = model.fit([X_tr,\
                    np.array(X_char_tr)],np.array(y_tr),\
                    batch_size=32, epochs=8, validation_split=0.1, verbose=1)

    # TODO: pass second arg to model.evaluate()

	# score = model.evaluate(X_te, y_te, batch_size=16)
	# evaluate the model for training examples and print accuracy =>98.63% 

	# score = model.evaluate([X_te, np.array(X_pos[len(X_tr):])], np.array(y_te), verbose=1)
	# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
																	#.reshape(len(y_tr),max_len,1)
	# score = model.evaluate([X_tr, np.array(X_char_tr)],\
	# 	 np.array(y_tr).reshape(len(y_tr),max_len, 1), verbose=1)
	# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

	# save model on disk
	# save_path = "C:\\Users\\Usman Ahmad\\Desktop\\P3_LSTM_saved.pkl"
	
	# save_model(model, weights_file='weights.h5', params_file='params.json')
	save_model(model)
	model.save("My_Custom_Model3.h5")

	from keras_contrib.utils import save_load_utils 
	# save using keras_contrib.utils.save_load_utils

	save_load_utils.save_all_weights(model, "Model_saved_using_contrib.h5")

	model.save_weights("model_weights.h5")
	with open("model_architecture.json", "w") as json_file:
	    json_file.write(model.to_json())

	# print("Saved model to disk"
	# serialize weights to HDF5
	print("Saved model to disk")

	score = model.evaluate([X_te, np.array(X_char_te)],\
		 np.array(y_te), verbose=1)
	print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

	plot_history(history)

	print(model.summary())


	predict(model, X_te, y_te, X_char_te, idx2word, idx2tag)

	"""

	i = 2318
	y_pred = model.predict([X_te, np.array(X_char_te)])

	print("np.array(y_pred).shape::",(np.array(y_pred).shape))

	p = np.argmax(y_pred[i], axis=-1)

	print(p)

	print('*'*50, 'p[0]','*'*50 )
	print(p[0])
	print('*'*50)
	if words is not None and tags is not None:
		i = 2318
		# p = model.predict(np.array([X_te[i],np.array(X_char_te)[i]]))
		# p = np.argmax(p, axis=-1)
		print("{:15} ({:5}): {}".format("Word", "True", "Pred"))

		pp = p

		# if all(p[0]): # if p[0] is iterable
			# pp = p[0]

		for w, pred in zip(X_te[i], pp):
			if w != 0:
				print("{:15}: {}".format(words[w], tags[pred]))

		print('*'*50)
		print(p)
		print("len(p) = ",len(p))
####################################TRUE###################################
	i = 2318
	y_pred = model.predict([X_te, np.array(X_char_te)])

	p = np.argmax(y_pred[i], axis=-1)

	print('*'*50)
	if words is not None and tags is not None:
		i = 2318
		# p = model.predict(np.array([X_te[i],np.array(X_char_te)[i]]))
		# p = np.argmax(p, axis=-1)
		print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
		for w,t, pred in zip(X_te[i],np.array(y_te)[i], p):
			if w != 0:
				print("{:15}: {:15} {}".format(words[w-1],tags[t], tags[pred]))

		print('*'*50)
		print(p)
		print("len(p) = ",len(p))

	# for x in X_te[i]:
	# 	print(words[x], end = " ")
	# print(" ")


	del model

	from keras.models import load_model
	# loaded_model = load_model("My_Custom_Model3.h5")
	# loaded_model = ""
	load_all_weights(loaded_model, "Model_saved_using_contrib.h5")

	print("Model Loaded.. Evaluating again")


	score = loaded_model.evaluate(X_te, np.array(y_te), verbose=1)
	print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

	score = loaded_model.evaluate(X_te, np.array(y_te), verbose=1)
	print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
	
	# score = loaded_model.evaluate(X,Y, verbose=1)
	#print accuracy
	# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
	if words is not None and tags is not None:
		i = 2319
		p = loaded_model.predict(np.array([X[i]]))
		# p = loaded_model.predict([np.array(X[i]),second_input[i]])

		p = np.argmax(p, axis=-1)
		print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
		for w, pred in zip(X[i], p[0]):  # p[0] = p[[1,3,4,5]]
			print("{:15}: {}".format(words[w], tags[pred]))

	# for x in X[i]:
	# 	print(words[x], end = " ")
	# print(" ")


	"""

from keras_contrib.layers import CRF
from keras.models import load_model

def create_custom_objects():
    instanceHolder = {"instance": None}
    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)
    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)
    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)
    return {"ClassWrapper": ClassWrapper ,"CRF": ClassWrapper, "loss": loss, "accuracy":accuracy}

def load_keras_model(path):
    model = load_model(path, custom_objects=create_custom_objects())
    return model

def fake_loss(y_true, y_pred):
 		return 0


def main2(X,X_te,X_char_te,y_te,words = None, tags = None, idx2word = None, idx2tag = None, word_idx =None):

	i = 2318
	true = np.argmax(y_te[i], axis = -1)
	# print(np.array(y_te)[2318])
	# print("*"*50,"Y_te[i]","*"*50)
	# print(y_te[2318])
	print(true)
	print("*"*50)

	from keras.models import model_from_json
	# from keras.models import load_model

	# load json and create model
	json_file = open('model_architecture.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	# loaded_model = model_from_json(loaded_model_json)

	# # load weights into new model
	# loaded_model.load_weights("model_weights.h5")
	# print("Loaded model from disk")

	# loaded_model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
	# # load json and create model
	# with open('model.json', 'r') as json_file:
	# 	loaded_model_json = json_file.read()

	# loaded_model = model_from_json(loaded_model_json)
	# # load weights into new model
	# loaded_model.load_weights("model.h5")
	# # print("Loaded model from disk")

	# # Model reconstruction from JSON file
	# with open('model_architecture.json', 'r') as f:
	#     loaded_model = model_from_json(f.read())

	# Load weights into the new model
	# loaded_model.load_weights('model_weights.h5')
		 
	# evaluate loaded model on test data
	# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	# loaded_model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

	# loaded_model = load_model("My_Custom_Model3.h5")
	from keras_contrib.layers import CRF

	max_len = 75
	n_tags = len(tags)
	n_words = len(words)
	max_len_chars = 10
	n_chars = 98
	# word_input = Input(shape=(max_len,))
	# word_emb = Embedding(input_dim=n_words + 1, output_dim=20,input_length=max_len, mask_zero = True)(word_input)


	# model = load_model("My_Custom_Model3.h5", custom_objects={'CRF': CRF, 'loss': fake_loss})
 	# model = Dropout(0.1)(word_emb)
	# model = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout= 0.1))(model)
	# model = TimeDistributed(Dense(50, activation = "relu"))(model)
	# crf = CRF(n_tags)
	# out = crf(model)

	# model = Model(inputs = word_input , outputs =out)
	# model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
	# _ word2idx=None
	# word_idx = word2idx


	#*************************Working Lines**************************#
	# model = load_my_model()
	model = create_model(max_len, n_words, n_tags, tags , max_len_chars, n_chars)
	model.load_weights("model_weights.h5")

	#*******************Testing to Work********************#

	# from keras.models import load_model
	# from keras_contrib.layers.advanced_activations import PELU
	# from keras_contrib.layers import CRF
	# model = load_model("Model_saved_using_contrib.h5")

	# save_load_utils.load_all_weights(model, "Model_saved_using_contrib.h5")
	# model = load_keras_model("My_Custom_Model3.h5")
	# model = load_model(filename, custom_objects={'CRF':CRF})
	# model = load_model(weights_file ='model_weights.h5', params_file='model_architecture.json')
	# model = load_model()

	# from keras_contrib.layers import CRF

	# print(model)
	# print(model.CRF)
	# model.compile(optimizer = "adam" , loss = loss)

	score = model.evaluate([X_te, np.array(X_char_te)], np.array(y_te), verbose=1)
	print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

	predict(model, X_te, y_te, X_char_te, idx2word, idx2tag)

def convert_to_vocab(arr1, idx2tag):

	arr1_vocab, arr2_vocab = [], []
	for lst in arr1:
		dummy = []
		for i in lst:
			if idx2tag[i] != "ENDPAD":
				dummy.append(idx2tag[i])

		arr1_vocab.append(dummy)
	return arr1_vocab


from seqeval.metrics import f1_score, classification_report

def calculate_precision_recall(my_pred, my_true, idx2tag):
	
	pred_vocab = convert_to_vocab(my_pred, idx2tag)
	true_vocab = convert_to_vocab(my_true, idx2tag)
	
	score = f1_score(true_vocab, pred_vocab)
	print(' - f1: {:04.2f}'.format(score * 100))
	print(classification_report(true_vocab, pred_vocab))


def predict(model, X_te,y_te, X_char_te, idx2word, idx2tag ):

	i = 19#2318
	print("Computing precision_recall_fscore ...")
	y_pred = model.predict([X_te, np.array(X_char_te)])

	my_pred = np.argmax(y_pred, axis = -1)
	complete_pred = np.hstack(my_pred)

	my_true = np.argmax(y_te, axis = -1)
	complete_true = np.hstack(my_true)
	
	calculate_precision_recall(my_pred, my_true, idx2tag)

	from sklearn.metrics import precision_recall_fscore_support

	print('*'*20,"my_pred for i",'*'*20)
	# print(my_pred[i])   # same, np.hstack(my_pred)
	print(complete_pred[:20])
	print(complete_true[:20])


	print('*'*20,"precision_recall_fscore_support",'*'*20)
	print(precision_recall_fscore_support(complete_true, complete_pred, average='macro'))

	p = np.argmax(y_pred[i], axis=-1)

	if idx2word is not None and idx2tag is not None:
		# i = 19#2318
		print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
		true = np.argmax(y_te[i], axis = -1)
		print("First true, second Predicted...")
		print(true)
		print(p)
		for w,t, pred in zip(X_te[i],true, p):
			if w != 0:
				print("{:15}: {:10} {}".format(idx2word[w],idx2tag[t] ,idx2tag[pred]))

		print('*'*50)
		print(p)
		print("len(p) = ",len(p))



	print(precision_recall_fscore_support(true, p, average='macro'))

def load_my_model():

	# model.save_weights("model_weights.h5")
	model = create_model(max_len, n_words, n_tags, tags , max_len_chars, n_chars)
	model.load_weights("model_weights.h5")

	return model

if __name__=='__main__':
	global trained
	trained = True
	main()




# in case of multiple embeddings features we concatenate both [pos_emb + word_input]
# model.shape  = [?, 50, 30]

# word_input = Input(shape=(max_len,))
# word_emb = Embedding(input_dim=n_words + 1, output_dim=20,
#                      input_length=max_len, mask_zero=True)(word_input)

# pos_input = Input(shape=(max_len,))
# pos_emb = Embedding(input_dim=len(pos_tags), output_dim=10,
#                     input_length=max_len, mask_zero=True)(pos_input)

# model = keras.layers.concatenate([word_emb, pos_emb])
# model = Bidirectional(LSTM(units=50, return_sequences=True,
#                            recurrent_dropout=0.1))(model)


# in case of one-hot encoded vectors 

# pos_input = Input(shape=(max_len, len(pos_tags)))
# Note that the input has to be a matrix of the length of the sequence 
# of the one-hot vectors.

# model.fit([X_words, X_pos, X_other_features], y_train)