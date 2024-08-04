import os
CNGCI_DEBUG = True if os.getenv("CNGCI_DEBUG", False) else False

from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
from src.story_dataclasses import CommonsenseRelation, StorySentence, ConflictStory

from src.modules.nli_predictor import NLIPredictor
from src.modules.subject_extractor import SubjectExtractor, determine_subject_change
#################### Implication-based rules  ####################
class ImplicationRuleScorer(object):
	def __init__(
		self,
		nli_rules: dict,
		weight_rules: dict,
		subject_extractor: SubjectExtractor,
		nli_predictor: NLIPredictor,
		nli_predictor_batch_size: int
	):
		'''
		Rule format:
		{
			"context": {
				"same_subject": [
					(target_relation, candidate_relation, threshold),
					...
				],
				"changed_subject": [
					(target_relation, candidate_relation, threshold),
					...
				]
			},
			"obstacle_later": ..., ## obstacle that happens after target (S2)
			"obstacle_earlier": ..., ## obstacle that happens before target (S4, S5)
			"preceding": ...
		}
		'''
		## NLI Rules
		self.context_nli_rules = nli_rules["context"]
		self.obstacle_later_nli_rules = nli_rules["obstacle_later"]
		self.obstacle_earlier_nli_rules = nli_rules["obstacle_earlier"] ## s4, s5
		self.preceding_nli_rules = nli_rules["preceding"]

		## Weight Rules
		# significance
		self.significance_weight_rules = weight_rules["significance"]
		# position
		self.position_weight_rules = weight_rules["position"]

		self.subject_extractor = subject_extractor
		self.nli_predictor = nli_predictor
		self.nli_predictor_batch_size = nli_predictor_batch_size

	def calculate_score(
		self, 
		story: ConflictStory,
		candidate_sentence_idx: int,
		comparing_sentence_type: str # context, obstacle, preceding
	) -> float:
		## Determine NLI Rule with information
		subjects = self._get_subjects(story)
		comparing_sentence_idx = self._determine_comparing_sentence_idx(
			story = story,
			comparing_sentence_type=comparing_sentence_type,
			candidate_sentence_idx = candidate_sentence_idx
		)
		is_obstacle_earlier = self._determine_is_obstacle_earlier(story=story, candidate_sentence_idx=candidate_sentence_idx)
		is_subject_changed = self._determine_is_subject_changed(
			subjects = subjects,
			candidate_sentence_idx = candidate_sentence_idx,
			comparing_sentence_idx = comparing_sentence_idx
		)
		nli_rules = self._determine_nli_rules(
			comparing_sentence_type=comparing_sentence_type,
			is_obstacle_earlier=is_obstacle_earlier,
			is_subject_changed=is_subject_changed
		)
		## Apply nli rule to calculate scores - NLI prediction
		score = self._score_with_nli_rules(
			comparing_sentence = story.sentences[comparing_sentence_idx],
			candidate_sentence = story.sentences[candidate_sentence_idx],
			rules = nli_rules
		)
		if CNGCI_DEBUG:
			print("NLI Score: {:.4f}".format(score))
		## get weights
		weight = self.get_weight(
			comparing_sentence_type = comparing_sentence_type,
			candidate_sentence_idx = candidate_sentence_idx,
			comparing_sentence_idx = comparing_sentence_idx
		)
		if CNGCI_DEBUG:
			print("Weight: {:.4f}".format(weight))
		return score * weight
		
	### Subject Condition
	def _get_subjects(self, story: ConflictStory) -> List[dict]:
		story_sentences = [story.sentences[i].value for i in range(story.num_sentences)]
		subjects = self.subject_extractor.get_subjects(story_sentences)
		return subjects

	def _determine_comparing_sentence_idx(
		self, story: ConflictStory, comparing_sentence_type: str, candidate_sentence_idx: int,
	) -> int:
		if comparing_sentence_type=="preceding":
			comparing_sentence_idx = candidate_sentence_idx-1
		elif comparing_sentence_type=="obstacle":
			comparing_sentence_idx = story.obstacle_idx
		elif comparing_sentence_type=="context":
			comparing_sentence_idx = story.context_idx
		else:
			raise Exception("Comparing sentence type {} not defined".format(comparing_sentence_type))
		return comparing_sentence_idx

	def _determine_is_subject_changed(
		self, subjects: List[dict], candidate_sentence_idx: int, comparing_sentence_idx: int
	) -> bool:
		## determine subject change
		is_subject_changed = determine_subject_change(
			subjects = subjects,
			idx1 = candidate_sentence_idx,
			idx2 = comparing_sentence_idx
		)
		return is_subject_changed

	def _determine_is_obstacle_earlier(
		self,
		story: ConflictStory,
		candidate_sentence_idx: int,
	) -> bool:
		return story.obstacle_idx < candidate_sentence_idx

	def _determine_nli_rules(
		self,
		comparing_sentence_type: str,
		is_obstacle_earlier: bool,
		is_subject_changed: bool,
	) -> List[List[str]]:
		if comparing_sentence_type=="context":
			comparing_type_rule = self.context_nli_rules
		elif comparing_sentence_type=="obstacle":
			if not is_obstacle_earlier: ## s2
				comparing_type_rule = self.obstacle_later_nli_rules
			else: ## s4, s5s
				comparing_type_rule = self.obstacle_earlier_nli_rules
		elif comparing_sentence_type=="preceding":
			comparing_type_rule = self.preceding_nli_rules
		else:
			raise Exception("comparing sentence type {} not defined".format(comparing_sentence_type))

		## by subject change
		if is_subject_changed:
			rules = comparing_type_rule["changed_subject"]
		else:
			rules = comparing_type_rule["same_subject"]
		return rules

	## Applying Rules
	def _score_with_nli_rules(
		self,
		comparing_sentence: StorySentence,
		candidate_sentence: StorySentence,
		rules: List[List[str]]
	) -> float:
		'''
		Rules format:
		[
			["oReact", "xReact", "entailment"],
			["xReact", "oReact", "entailment"]
		]
		'''
		rule_scores = []
		for cmp_relation_type, cand_relation_type, nli_cond in tqdm(rules):
			##
			cmp_relation = comparing_sentence.commonsense_relations[cmp_relation_type]
			cand_relation = candidate_sentence.commonsense_relations[cand_relation_type]
			relation_pair_score = self._calculate_relation_pair_score(
				relation1 = cmp_relation,
				relation2 = cand_relation,
				condition = nli_cond
			)
			rule_scores.append(relation_pair_score)
		
		## Average??? - TODO: inquire hyunju
		score = np.mean(rule_scores)
		return score

	def _calculate_relation_pair_score(
		self,
		relation1: CommonsenseRelation,
		relation2: CommonsenseRelation,
		condition: str
	) -> float:
		'''
		Use NLI Prediction result to determine if relation pair satisfies condition
		condition: "strong entailment", "entailment", "contradiction"
		given r1, r2 number of relation sentences
		-> r1*r2 number of input texts (pairwise)
		-> calculate ratio of "entailment"/"contradicted" predicted input to check satisfactions
		Final score is 1.0 if satisfied, 0.0 if not
		'''
		relation1_values = relation1.values
		relation2_values = relation2.values
		
		## Make Pairwise NLI Predictor Input
		check_is_not_none = lambda x: x!="none"
		paired_values = [[r1, r2] for r2 in filter(check_is_not_none, relation2_values) for r1 in filter(check_is_not_none, relation1_values)]
		if len(paired_values)==0:
			return 0.0

		# ['entailment', ...]
		if CNGCI_DEBUG:
			print("NLI Predictor input", len(paired_values))
		predicted: List[str] = self.nli_predictor.predict(texts = paired_values, batch_size = self.nli_predictor_batch_size)

		## ratio of entailment, contradiction, neutral
		entailment_ratio = predicted.count('entailment') / len(predicted)
		contradiction_ratio = predicted.count('contradiction') / len(predicted)
		satisfied = False
		if condition =="strong entailment" and entailment_ratio>0.5:
			satisfied = True
		elif condition == "entailment" and entailment_ratio>0.3:
			satisfied = True
		elif condition == "contradiction" and contradiction_ratio>0.3:
			satisfied = True

		## calculate pairwise score
		score = 1.0 if satisfied else 0.0
		return score

	## Calculate weight
	def get_weight(
		self,
		comparing_sentence_type: str,
		candidate_sentence_idx: int,
		comparing_sentence_idx: int
	) -> float:
		## get position & significance weight
		abs_dist = abs(candidate_sentence_idx - comparing_sentence_idx)
		if comparing_sentence_type=="context":
			position_weight = self.position_weight_rules["context"][abs_dist]
			significance_weight = self.significance_weight_rules["context"]
		elif comparing_sentence_type=="obstacle":
			position_weight = self.position_weight_rules["obstacle"][abs_dist]
			significance_weight = self.significance_weight_rules["obstacle"]
		else:
			position_weight = self.position_weight_rules["preceding"]
			significance_weight = self.significance_weight_rules["preceding"]
		return position_weight*significance_weight

#################### Similarity-based rules  ####################
def calculate_pairwise_cosine_similarity(x,y):
	'''
	shape:
	x: (n_samples_x, n_dim)
	y: (n_samples_y, n_dim)
	->
	(n_samples_x, n_samples_y)
	'''
	return cosine_similarity(x, y)

class SimilarityRuleScorer(object):
	def __init__(self, rules: dict, subject_extractor: SubjectExtractor,):
		'''
		Rule Format
		{
			"same_subject": [
				(target_relation, candidate_relation, threshold),
				...
			],
			"changed_subject": [
				(target_relation, candidate_relation, threshold),
				...
			]
		}
		'''
		self.same_subject_rules = rules["same_subject"]
		self.changed_subject_rules = rules["changed_subject"]

		self.subject_extractor = subject_extractor

	def calculate_score(
		self, 
		story: ConflictStory,
		candidate_sentence_idx: int
	) -> float:
		## Determine NLI Rule with information
		subjects = self._get_subjects(story)
		comparing_sentence_idx = self._determine_comparing_sentence_idx(
			story = story,
			candidate_sentence_idx = candidate_sentence_idx
		)
		is_subject_changed = self._determine_is_subject_changed(
			subjects = subjects,
			candidate_sentence_idx = candidate_sentence_idx,
			comparing_sentence_idx = comparing_sentence_idx
		)

		## check subject character change
		if is_subject_changed:
			rules = self.changed_subject_rules
		else:
			rules = self.same_subject_rules

		##
		comparing_sentence = story.sentences[comparing_sentence_idx]
		candidate_sentence = story.sentences[candidate_sentence_idx]

		## Calculate Score
		pair_scores = []
		for comparing_relation_type, candidate_relation_type, threshold in rules:
			comparing_relation: CommonsenseRelation = comparing_sentence.commonsense_relations[comparing_relation_type]
			candidate_relation: CommonsenseRelation = candidate_sentence.commonsense_relations[candidate_relation_type]
			pair_similarity_score = self._calculate_relation_pair_score(
				relation1 = comparing_relation,
				relation2 = candidate_relation, 
				threshold = threshold
			)
			if CNGCI_DEBUG:
				print("PAIR {} - {}: {:.4f}".format(comparing_relation_type, candidate_relation_type, pair_similarity_score))
			pair_scores.append(pair_similarity_score)
		score = np.mean(pair_scores) ## average across relation types
		return score
	
	def _get_subjects(self, story: ConflictStory) -> List[dict]:
		story_sentences = [story.sentences[i].value for i in range(story.num_sentences)]
		subjects = self.subject_extractor.get_subjects(story_sentences)
		return subjects

	def _determine_comparing_sentence_idx(
		self, story: ConflictStory, candidate_sentence_idx: int,
	) -> int:
		comparing_sentence_idx = candidate_sentence_idx-1
		return comparing_sentence_idx

	def _determine_is_subject_changed(
		self, subjects: List[dict], candidate_sentence_idx: int, comparing_sentence_idx: int
	) -> bool:
		## determine subject change
		is_subject_changed = determine_subject_change(
			subjects = subjects,
			idx1 = candidate_sentence_idx,
			idx2 = comparing_sentence_idx
		)
		return is_subject_changed

	def _calculate_relation_pair_score(
		self,
		relation1: CommonsenseRelation,
		relation2: CommonsenseRelation,
		threshold: float
	) -> float:
		'''
		r1*r2 number of input embeddings (pairwise)
		* (r1, dim) vs (r2, dim)
		-> calculate ratio of "entailment"/"contradicted" predicted input to check satisfactions
		Final score is 1.0 if satisfied, 0.0 if not
		'''
		relation1_values = relation1.values
		relation2_values = relation2.values
		
		## Make Pairwise NLI Predictor Input
		check_is_not_none = lambda x: x!="none"
		# print(relation1_values, relation2_values)
		relation1_nonempty_idxs = [i for i, val in enumerate(relation1_values) if check_is_not_none(val)]
		relation2_nonempty_idxs = [i for i, val in enumerate(relation2_values) if check_is_not_none(val)]
		# print(len(relation1_nonempty_idxs), len(relation2_nonempty_idxs))
		if len(relation1_nonempty_idxs)==0 or len(relation2_nonempty_idxs)==0:
			return 0.0

		pairwise_similarities = calculate_pairwise_cosine_similarity(
			relation1.embeddings[relation1_nonempty_idxs, :],
			relation2.embeddings[relation2_nonempty_idxs, :]
		)
		if CNGCI_DEBUG:
			print("PAIRWISE", pairwise_similarities.shape)
		pairwise_similarity_scores = np.ones_like(pairwise_similarities)
		pairwise_similarity_scores[pairwise_similarities<threshold] = 0.0
		pair_similarity_score = np.mean(pairwise_similarity_scores)
		return pair_similarity_score