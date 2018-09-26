
# setting python working environment ...

# for variational RNN, we use SAME dropout mask(0.5) at each timestep including recurrent layers.
# applying DIFFERENT mask on input & output layers deteriorates it's performance(Naive dropout).
# embedding dropout MUST be applied AFTER conversion of words in embeddings
# embedding dropout drops word types from word-sequences/word-embeddings
# with NO embedding dropout, model overfits and needs early stopping
# learning rate DECAY also performs best
 
import numpy as np
import json
from matplotlib import pyplot as plt

from keras.models import Input, Model ,model_from_json
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D,Flatten,GlobalMaxPooling1D
from keras_contrib.layers import CRF

# cls.weights_file = os.path.join(SAVE_ROOT, 'weights.h5')
# cls.params_file = os.path.join(SAVE_ROOT, 'params.json')

def save_model(model, weights_file='weights.h5', params_file='params.json'):
    with open(params_file, 'w') as f:
        params = model.to_json()
        json.dump(json.loads(params), f, sort_keys=True, indent=4)
        model.save_weights(weights_file)


def load_model(weights_file ='weights.h5', params_file='params.json'):
    with open(params_file) as f:
        model = model_from_json(f.read(), custom_objects={'CRF': CRF})
        model.load_weights(weights_file)

    return model



def create_model(max_len, n_words, n_tags, pos_tags ,\
					max_len_chars, n_chars, _word2idx=None):

	# input and embedding for words
	word_in = Input(shape=(max_len,))

	if _word2idx != None:
		embeddings_index = prepare_glove_embeddings()
		embedding_matrix = leverage_embeddings(_word2idx, embeddings_index)
		emb_word = Embedding(input_dim=n_words + 2, output_dim=100,weights=[embedding_matrix],
	                     input_length=max_len, mask_zero=True)(word_in)
	else:
		emb_word = Embedding(input_dim=n_words + 2, output_dim=100,
	                     input_length=max_len, mask_zero=True, name="emb_word")(word_in)

	# input and character-embeddings
	char_in = Input(shape=(max_len, max_len_chars))
	emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=30,
	                           input_length=max_len_chars, name="emb_char"))(char_in)

	# character LSTM to get word encodings by characters
	# char_enc = TimeDistributed(Bidirectional(LSTM(units=30, return_sequences=False,
	                                # recurrent_dropout=0.5)))(emb_char)
	# print(emb_char)
	conv_1d = TimeDistributed(Conv1D(filters=40, kernel_size=3, padding="valid", activation="relu", name="Conv1D"))(emb_char)
	conv_1d = TimeDistributed(Dropout(0.4))(conv_1d)
	# print(conv_1d)
	maxpool1d = TimeDistributed(GlobalMaxPooling1D())(conv_1d)
	# print("GlobalMaxPooling1D: ",maxpool1d)
	char_enc = TimeDistributed(Flatten())(maxpool1d)
	# print("Flatten :", char_enc)
	# main LSTM
	word_embeddings = concatenate([emb_word, char_enc])
	# word_embeddings = Dropout(0.5)(word_embeddings)
	# word_embeddings = SpatialDropout1D(0.4)(word_embeddings)
	main_lstm = Bidirectional(LSTM(units=100, return_sequences=True,
	                               recurrent_dropout=0.6))(word_embeddings)
	# out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)
	# added from below
	# z = Bidirectional(LSTM(units=100, return_sequences=True))(word_embeddings)
	
	main_lstm = Dropout(0.5)(main_lstm)
	z = Dense(100, activation='tanh')(main_lstm)
	# from keras import optimizers

	crf = CRF(n_tags + 1, sparse_target=False)
	loss = crf.loss_function
	pred = crf(z)

	model = Model(inputs=[word_in, char_in], outputs=pred)
	model.compile(optimizer = "adam", loss = loss, metrics = [crf.accuracy])


	# model = Model(inputs = [word_input, char_input] , outputs=out)
	# SGD = optimizers.SGD(lr=0.015, decay=0.05, momentum=0.9)
	# model.compile(optimizer=SGD, loss=crf.loss_function, metrics=[crf.accuracy] )
	return model




def prepare_glove_embeddings(path = "./glove.6B.100d.txt"):

	print("started reading glove ...")
	embeddings_index = {}
	with open('glove.6B.100d.txt') as f:
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
			# f.close()
	print('Found %s word vectors.' % len(embeddings_index))
	return embeddings_index


def leverage_embeddings(_word2idx, embeddings_index):

	embedding_matrix = np.zeros((len(_word2idx), 100))
	for word, i in _word2idx.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector

	return embedding_matrix

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
# model = Model(inputs=[word_input, char_input], outputs=pred)
# model.compile(loss=loss, optimizer="adam")

# word_input = Input(shape=(max_len,))
# # print("word Input::", word_input)
# word_emb = Embedding(input_dim=n_words + 2,name="word_emb",\
# 			mask_zero = True, output_dim=100,input_length=max_len)(word_input)

# print(word_emb)
# char_input = Input(shape =(max_len, max_len_chars,))
# # print("char input::", char_input)

# char_emb = Embedding(input_dim=n_chars + 2, name="char_emb", mask_zero=True,\
# 						 output_dim = 25,input_length= max_len_chars)(char_input)

# char_emb = TimeDistributed(LSTM(25))(char_emb)
# print(char_emb)

"""word_embeddings = Dropout(0.5)(word_embeddings)
z = Bidirectional(LSTM(units=100, return_sequences=True))(word_embeddings)
z = Dense(100, activation='tanh')(z)
from keras import optimizers
crf = CRF(n_tags + 1, sparse_target=False)
loss = crf.loss_function
pred = crf(z)
model = Model(inputs=[word_input, char_input], outputs=pred)
model.compile(loss=loss, optimizer="adam")"""

# char_enc = TimeDistributed(LSTM(units = 30, return_sequences=False,recurrent_dropout = 0.5))(char_emb)
# print(char_enc)

# TODO: Add a dropout layer as specified in the NER Paper
# before entring into convolution phase
# dropped_out_char_emb = TimeDistributed(SpatialDropout1D(0.4))(char_emb)
# print("SpatialDropout1D", dropped_out_char_emb)

# conv1d_0  = TimeDistributed(Conv1D(filters = 30, kernel_size =3,\
					 # name = "conv1d_0", padding = 'valid', activation="relu"))(char_emb)
# print(conv1d_0)
# max pooling dimensionally - (max-0ver-time pooling)
# maxpool1d_0 = TimeDistributed(GlobalMaxPooling1D())(conv1d_0)
# print("GlobalMaxPooling1D::",maxpool1d_0)	
# flattens the last two dimensions of output so to make it a feature vector 
# for each of the input character
# char_enc_0 = TimeDistributed(Flatten())(maxpool1d_0)
# print("Flatten Layer::",char_enc_0)

# conv1d_1  = TimeDistributed(Conv1D(filters =14, kernel_size =2, name = "conv1d_1", padding = 'valid',activation="relu"))(char_emb)
# print(conv1d_1)
# maxpool1d_1 = TimeDistributed(GlobalMaxPooling1D())(conv1d_1)
# print(maxpool1d_1)
# flattens the last two dimensions of output so to make it a feature vector 
# for each of the input character
# char_enc_1 = TimeDistributed(Flatten())(maxpool1d_1)
# char_enc_1 = TimeDistributed(Flatten())(char_emb)
# print(char_enc_1)

# x = keras.layers.concatenate([word_emb, char_enc])
# print("x-concatenated::",x)
# x = keras.layers.concatenate([word_emb, char_enc_1])

# print("Concatenated[word_emb + char_enc_0 + char_enc_1]::", x)
# drops some of dimensions of input data: in our case, words
# x = SpatialDropout1D(0.4)(x)  # embedding dropout applied after converting word embeddings
# print("SpatialDropout1D::",x)
# PE = recurrent dropout-->0.5   (variational dropout)
# main_lstm = Bidirectional(LSTM(units = 50, return_sequences = True, 
						 # recurrent_dropout = 0.5))(x)
# print(main_lstm)		     # n_tags
# dense_layer = TimeDistributed(Dense(49 + 1, activation="relu"))(main_lstm)

# crf = CRF(n_tags + 1)
# out = crf(main_lstm)




# print(out)
# print(concat_in)
# print("\nword_input:",word_emb)
# # print("\npos_input:",pos_input)

# model = Dropout(0.1)(word_emb)
# model = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout= 0.1))(model)
# # Dense Layer with 50 neurons
# model = TimeDistributed(Dense(50, activation = "relu"))(model)
# crf = CRF(n_tags)
# # output layer is crf
# out = crf(model)

# model creation for compilation
# model = Model(inputs = [word_input,pos_input] , outputs =out)
# model = Model([word_in, char_in], out)
# from keras import optimizers

# model = Model(inputs = [word_input, char_input] , outputs=out)
# SGD = optimizers.SGD(lr=0.015, decay=0.05, momentum=0.9)
# model.compile(optimizer=SGD, loss=crf.loss_function, metrics=[crf.accuracy] )
# # model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
# print(model)
# model.compile(optimizer="adam" , loss="sparse_categorical_crossentropy", metrics = ["acc"])
