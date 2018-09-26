
import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

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


def predict(model, X_te, y_te, X_char_te, idx2word, idx2tag, sentences ):

	# i =15#2318
	# print("Computing precision_recall_fscore ...")
	y_pred = model.predict([X_te, np.array(X_char_te)])

	my_pred = np.argmax(y_pred, axis = -1)
	complete_pred = np.hstack(my_pred)

	# my_true = np.argmax(y_te, axis = -1)
	# complete_true = np.hstack(my_true)
	
	# calculate_precision_recall(my_pred, my_true, idx2tag)

	from sklearn.metrics import precision_recall_fscore_support

	# print('*'*20,"my_pred for i",'*'*20)
	# print(my_pred[i])   # same, np.hstack(my_pred)
	# print(complete_pred[:20])
	# print(complete_true[:20])

	# print('*'*20,"precision_recall_fscore_support",'*'*20)
	# print(precision_recall_fscore_support(complete_true, complete_pred, average='macro'))

	# p = np.argmax(y_pred[i], axis=-1)

	p_arr = np.argmax(y_pred, axis = -1) 
	# print(idx2word)
	# print(p)


	# print(sentences[i])

	tags_arr = []

	if idx2word is not None and idx2tag is not None:
		# i = 19#2318
		# print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
		# true = np.argmax(y_te[i], axis = -1)
		# print("First true, second Predicted...")
		# print(true)
		# print(p)
		for sent, pred_arr in zip(sentences, p_arr):# second_arg:true
			tag_list = []
			for w , pred in zip(sent, pred_arr):
				# if w != 0:
				# ww = idx2word[w]
				tt = idx2tag[pred]
				tag_list.append(tt)
				# print("{:15}: {:10} ".format(w ,tt))

			tags_arr.append(tag_list)


		# print('*'*50)
		# print(p)
		# print("len(p) = ",len(p))
	return p_arr, tags_arr


def _build_response(self, words, tags, prob = None):
    # words = self.tokenizer(sent)
    res = {
        'words': words,
        'entities': [

        ]
    }
    chunks = get_entities(tags)

    for chunk_type, chunk_start, chunk_end in chunks:
        chunk_end += 1
        entity = {
            'text': ' '.join(words[chunk_start: chunk_end]),
            'type': chunk_type,
            'score': float(np.average(prob[chunk_start: chunk_end])),
            'beginOffset': chunk_start,
            'endOffset': chunk_end
        }
        res['entities'].append(entity)

    return res



class Tagger(object):


	def __init__(self, sequence, model):
		pass


	def build(self):
		pass

	def _build_response(self):
		pass


	def predict(self):
		pass

	# def analyze(self, text):

	# 	pass