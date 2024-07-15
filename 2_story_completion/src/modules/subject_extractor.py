from typing import List, Tuple
import spacy

from spacy.tokens.token import Token
import neuralcoref

class SubjectExtractor(object):
	def __init__(self, model: str = "en_core_web_sm"):
		self.pipeline = self._init_coref_pipeline(model)

	def _init_coref_pipeline(self, spacy_model: str):
		pipeline = spacy.load(spacy_model)
		coref = neuralcoref.NeuralCoref(pipeline.vocab, max_dist=128, blacklist=False)
		pipeline.add_pipe(coref, name='neuralcoref')
		print('coref model loaded.')
		return pipeline
	
	def get_subjects(self, sentences: List[str]) -> List[dict]:
		# pass thorough spacy pipeline
		doc = self.pipeline(" ".join(sentences))

		# [Lana, she, song, she, She] <- identified subjects
		subject_tokens: List[Token] = self._get_main_subjects(doc)

		# [0, 0, 1, 0, 0] <- cluster ids of subjects
		subject_cluster_ids = self._get_cluster_of_subjects(doc, subject_tokens)

		# Map sentence & subject
		subjects = []
		for subject_token, cluster_id in zip(subject_tokens, subject_cluster_ids):
			subject_sentence_idx = -1
			for s_idx, sentence in enumerate(sentences):
				if str(subject_token.sent) in sentence:
					subject_sentence_idx = s_idx
					break
			subjects.append(
				{
					"sentence_idx": subject_sentence_idx,
					"value": str(subject_token),
					"cluster_id": cluster_id
				}
			)
		return subjects

	def _get_main_subjects(self, doc):
		roots = [t for t in doc if t.dep_ == 'ROOT']
		subjects = [[ch for ch in root.children if 'nsubj' in ch.dep_] for root in roots]
		subjects = [s[0] for s in subjects if len(s)]
		return subjects

	def _get_cluster_of_subjects(self, doc, subjects):
		clusters = list(doc._.coref_clusters)
		result = []
		for subject in subjects:
			cluster = self._get_cluster_of_subject(clusters, subject)
			result.append(cluster)
		return result
		
	def _get_cluster_of_subject(self, clusters, subject):
		for cidx, cluster in enumerate(clusters):
			for span in cluster.mentions:
				span_idxs = {x.idx for x in span}
				if subject.idx in span_idxs:
					return cidx
		return None

## Determine Subject Change
def determine_subject_change(subjects: List[dict], idx1: int, idx2: int):
	subj1_cluster_id = -1
	subj2_cluster_id = -1
	for subject in subjects:
		if idx1==subject["sentence_idx"]:
			subj1_cluster_id = subject["cluster_id"]
		elif idx2==subject["sentence_idx"]:
			subj2_cluster_id = subject["cluster_id"]
	
	if subj1_cluster_id==-1 or subj2_cluster_id==-1:
		return False ## assume no subject change
	elif subj1_cluster_id!=subj2_cluster_id:
		return True
	else:
		return False

if __name__=="__main__":
	extractor = SubjectExtractor()
	sentences = sentences = [
		"Lana was trying to figure out how to play a song.",
		"For some reason, she couldn't figure out how to play the song.",
		"The song is very difficult.",
		"Finally she decided to ask her friend for help.",
		"She ended up learning how to play the song."
	]

	'''
	[{'sentence_idx': 0, 'value': 'Lana', 'cluster_id': 0},
	{'sentence_idx': 1, 'value': 'she', 'cluster_id': 0},
	{'sentence_idx': 2, 'value': 'song', 'cluster_id': 1},
	{'sentence_idx': 3, 'value': 'she', 'cluster_id': 0},
	{'sentence_idx': 4, 'value': 'She', 'cluster_id': 0}]
	'''
	subjects = extractor.get_subjects(
		sentences = sentences
	)

	## determine subject change
	is_subject_changed = determine_subject_change(subjects, 0, 1)
	print(is_subject_changed) ## False

	is_subject_changed = determine_subject_change(subjects, 1, 2)
	print(is_subject_changed) ## True