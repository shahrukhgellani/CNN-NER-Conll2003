
# Orchestrates the whole activity of NER-Tagging
# main entry point of Programme

texty = "In the rugged Colorado Desert of California, there lies buried a treasure ship sailed there hundreds of years ago by either Viking or Spanish explorers. Some say this is legend; others insist it is fact. A few have even claimed to have seen the ship, its wooden remains poking through the sand like the skeleton of a prehistoric beast. Among those who say they've come close to the ship is small-town librarian Myrtle Botts. In 1933, she was hiking with her husband in the Anza-Borrego Desert, not far from the border with Mexico. It was early March, so the desert would have been in bloom, its washed-out yellows and grays beaten back by the riotous invasion of wildflowers. Those wildflowers were what brought the Bottses to the desert, and they ended up near a tiny settlement called Agua Caliente. Surrounding place names reflected the strangeness and severity of the land: Moonlight Canyon, Hellhole Canyon, Indian Gorge. Try Newsweek for only $1.25 per week To enter the desert is to succumb to the unknowable. One morning, a prospector appeared in the couple's camp with news far more astonishing than a new species of desert flora: He'd found a ship lodged in the rocky face of Canebrake Canyon. The vessel was made of wood, and there was a serpentine figure carved into its prow. There were also impressions on its flanks where shields had been attached—all the hallmarks of a Viking craft. Recounting the episode later, Botts said she and her husband saw the ship but couldn't reach it, so they vowed to return the following day, better prepared for a rugged hike. That wasn't to be, because, several hours later, there was a 6.4 magnitude earthquake in the waters off Huntington Beach, in Southern California. Botts claimed it dislodged rocks that buried her Viking ship, which she never saw again.There are reasons to doubt her story, yet it is only one of many about sightings of the desert ship. By the time Myrtle and her husband had set out to explore, amid the blooming poppies and evening primrose, the story of the lost desert ship was already about 60 years old. By the time I heard it, while working on a story about desert conservation, it had been nearly a century and a half since explorer Albert S. Evans had published the first account. Traveling to San Bernardino, Evans came into a valley that was the grim and silent ghost of a dead sea, presumably Lake Cahuilla. The moon threw a track of shimmering light, he wrote, directly upon the wreck of a gallant ship, which may have gone down there centuries ago. The route Evans took came nowhere near Canebrake Canyon, and the ship Evans claimed to see was Spanish, not Norse. Others have also seen this vessel, but much farther south, in Baja California, Mexico. Like all great legends, the desert ship is immune to its contradictions: It is fake news for the romantic soul, offering passage into some ancient American dreamtime when blood and gold were the main currencies of civic life.The legend does seem, prima facie, bonkers: a craft loaded with untold riches, sailed by early-European explorers into a vast lake that once stretched over much of inland Southern California, then run aground, abandoned by its crew and covered over by centuries of sand and rock and creosote bush as that lake dried out…and now it lies a few feet below the surface, in sight of the chicken-wire fence at the back of the Desert Dunes motel, $58 a night and HBO in most rooms.Totally insane, right? Let us slink back to our cubicles and never speak of the desert ship again. Let us only believe that which is shared with us on Facebook. Let us banish forever all traces of wonder from our lives. Yet there are believers who insist that, using recent advances in archaeology, the ship can be found. They point, for example, to a wooden sloop from the 1770s unearthed during excavations at the World Trade Center site in lower Manhattan, or the more than 40 ships, dating back perhaps 800 years, discovered in the Black Sea earlier this year."

# sentences : [ ["this", "is", "a", "cat"], 
#				["It", "is", "a", "dog"] ]
from Utils import *
from preprocessing import *
from Sequence import *
from Trainer import *
from MyModel import *
from Tagger import predict
from seqeval.metrics.sequence_labeling import get_entities
import os
# ignore the "tensorflow binary not compiled" warning for keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NERTagger(object):

	# def __init__(self):
		# self.text = text
		# self.trained = False

	# Utils.py
	@staticmethod
	def is_already_trained():
		"""checks if the model is already trained by 
		   checking directories that contain saved models """
		return True

	def evaluate_tagger(self):
		""" checks if already trained and then load data and evaluate tagger """
		pass

	def train(self, path_to_dataset , _epochs = 4, _batch_size = 32):
		
		_data , _words, _tags, _pos = load_dataset(path_to_dataset)
		preprocessor = Preprocessor(_data)
		
		sObj = Sequence(_words, preprocessor.sentences, tags=_tags)
		
		trainer = Trainer(sObj, preprocessor)
		trainer.build()
		sObj.split_test_train()
		trainer.fit(_epochs = _epochs)
		trainer.evaluate()



	def analyze(self, text, trained = False):
		""" returns an analyzed list of results for NER """
		from Tagger import _build_response
		model_params = Trainer.load_model_parameters()
		max_len, n_words, n_tags = 75, 23623, 9
		n_chars, max_len_chars = 84, 10


		sentences = normalize_to_sentences(text)
		words = get_tokens_from_sentences(sentences)

		if NERTagger.is_already_trained():

			sequenceObj = Sequence(words, sentences)
			sequenceObj.load_dicts_from_disk()
			sequenceObj.generate_input_sequence(sentences, trained = True)
			# sequenceObj.split_test_train()
			sequenceObj.predictions_shuffle()
			model = create_model(max_len, n_words, n_tags, None ,\
					max_len_chars, n_chars, _word2idx= None)

			model.load_weights("My_Custom_Model3.h5")
			pred_tags_arr, tags_text_arr = predict(model, sequenceObj.X_te, sequenceObj.y_te,\
				sequenceObj.X_char_te, sequenceObj.idx2word, sequenceObj.idx2tag, sentences)

			response_json_array = []
			for words, tags, prob in zip(sentences, tags_text_arr, pred_tags_arr):
				res = _build_response(self, words, tags, prob = prob)
				response_json_array.append(res)

			print(response_json_array[4])
			print(response_json_array[6])
			print(response_json_array[7])
			print(response_json_array[8])
			print(response_json_array[9])
			print(response_json_array[5])
			print(response_json_array[3])
			print(response_json_array[1])
			print(response_json_array[2])

			# sequenceObj.generate_sequence()

			# no need to generate output sequence

		else:
			sequenceObj.generate_dicts_from_vocab()



if __name__ ==  "__main__":

	# try:
	# NERTagger().train("./ner_dataset.csv")
	NERTagger().analyze(texty)
	# except:
		# pass