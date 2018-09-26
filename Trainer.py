

from Sequence import *
from preprocessing import *
from Utils import *
from Tagger import predict

class Trainer(object):

	def __init__(self, sequence, preprocessor):

		# self.model = model
		self.sObj = sequence
		self.preprocessor = preprocessor


	def build(self):

		# self.sObj = Sequence(self.preprocessor.words, self.preprocessor.sentences, tags=preprocessor.tags)
		refined_sentences = self.preprocessor.get_only_sentences_without_pos_nerTags()
		self.sObj.generate_dicts_from_vocab()
		self.sObj.generate_input_sequence(refined_sentences)
		print("printing refined_sentences now..")
		print(refined_sentences[:10])
		self.sObj.generate_output_sequence()

		self.model = create_model(self.sObj.max_seq_len, self.sObj.n_words, self.sObj.n_tags,\
									None, self.sObj.max_len_chars, self.sObj.n_chars,\
													 _word2idx=self.sObj.word2idx)


	def fit(self, _batch_size = 32, _epochs = 5):

		# self.sObj.split_test_train()
		# generate sequence of preprocessed words
		self.history = self.model.fit([self.sObj.X_tr, np.array(self.sObj.X_char_tr)],\
								np.array(self.sObj.y_tr),batch_size=_batch_size,\
								 epochs= _epochs, validation_split=0.1, verbose=1)

		self._save_model()
		# print("loading data to train..")
		# data, words, tags, pos = load_dataset() # unique
		

	def _save_model(self):

		self.model.save("MyTechniqueSaved.h5")
		model_params = {
			"max_len_chars": self.sObj.max_len_chars,
			"n_words": self.sObj.n_words,
			"max_seq_len": self.sObj.max_seq_len,
			"n_tags": self.sObj.n_tags,
			"n_chars": self.sObj.n_chars
		} 
		save_model_params("model_params.pkl", model_params)

	@staticmethod
	def load_model_parameters():

		model_params = load_model_params("model_params.pkl")
		return model_params

	def evaluate(self):

		score = self.model.evaluate([self.sObj.X_te, np.array(self.sObj.X_char_te)],\
									np.array(self.sObj.y_te), verbose=1)
		print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))

		plot_history(self.history)
		predict(self.model, self.sObj.X_te, self.sObj.y_te,\
				self.sObj.X_char_te, self.sObj.idx2word, self.sObj.idx2tag )
