import argparse
import os
import copy
import json
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

import torch
import pandas as pd
import numpy as np
from transformers import (
	AutoTokenizer, AutoModelForCausalLM
)

from src.modules.loader import (
	load_subject_extractor,
	load_commonsense_generator,
	load_nli_predictor
)
from src.modules.commonsense_relation_generator import CATEGORIES
from src.candidate_generator import ObsLM245NextSentenceCandidateGenerator
from src.scorer import ImplicationRuleScorer, SimilarityRuleScorer

from src.story_dataclasses import CommonsenseRelation, StorySentence, ConflictStory

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help = "tsv file with input data")
parser.add_argument('--config_dir', help = "json file with config contents")
parser.add_argument('--result_dir', help = "directory to save the results")
parser.add_argument('--seed', type = int, help = "random seed")
args = parser.parse_args()

## SET RANDOM SEED
import random
SEED = args.seed
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

## Load Config
with open(args.config_dir, "r") as f:
	config = json.load(f)

## Load Modules
candidate_gen_tokenizer = AutoTokenizer.from_pretrained("gpt2")
candidate_gen_model = AutoModelForCausalLM.from_pretrained(
	config['candidate_generator']['model_dir'],
	torch_dtype=torch.bfloat16
)
candidate_gen_model.resize_token_embeddings(len(candidate_gen_tokenizer))

generator = ObsLM245NextSentenceCandidateGenerator(
	model = candidate_gen_model,
	tokenizer = candidate_gen_tokenizer,
	device = torch.device(config['candidate_generator']['device'])
)
print("Loaded candidate generator")

## Comet Model
commonsense_generator = load_commonsense_generator(
	comet_model_dir = config['commonsense_generator']['model_dir'],
	embedding_model_dir = config['embedder']['model_dir'],
	device = torch.device(config['commonsense_generator']['device'])
)
print("Loaded comet")

subject_extractor = load_subject_extractor(model = config['subject_extractor']['spacy_model'])
print("Loaded subject extractor")

nli_predictor = load_nli_predictor(
	model_dir = config['nli']['model_dir'],
	device = torch.device( config['nli']['device'])
)
print("Loaded nli predictor")

## RULE SCORERS
rule_dir = config['rule_dir']
with open(rule_dir, "r") as f:
	RULES = json.load(f)

implication_scorer = ImplicationRuleScorer(
	nli_rules = RULES["implication"],
	weight_rules = RULES["weights"],
	subject_extractor = subject_extractor,
	nli_predictor = nli_predictor,
	nli_predictor_batch_size = config['nli']['batch_size']
)

similarity_scorer = SimilarityRuleScorer(
	rules = RULES["similarity"],
	subject_extractor = subject_extractor
)

########## Functions
def make_story_sentence(
	idx: int,
	value: str,
	sentence_type: str, # context/obstacle/other
) -> StorySentence:
	sentence = StorySentence(
		idx = idx,
		value = value,
		character = "",
		sentence_type = sentence_type,
		commonsense_relations = []
	)

	## Generate Commonsense
	decode_params = config['commonsense_generator']['decode_params']
	text_generator_batch_size = config['commonsense_generator']['batch_size']
	text_embedder_batch_size = config['embedder']['batch_size']
	sentence.commonsense_relations = commonsense_generator.generate(
		sentence.value,
		relation_types = CATEGORIES,
		decode_params = decode_params,
		text_generator_batch_size = text_generator_batch_size,
		text_embedder_batch_size = text_embedder_batch_size
	)
	return sentence

def evaluate_candidate(
	story: ConflictStory,
	candidate_sentence: StorySentence,
	candidate_idx: int
) -> float:
	candidate_story = copy.deepcopy(story)
	candidate_story.num_sentences = story.num_sentences+1
	candidate_story.sentences[candidate_idx] = candidate_sentence
	## Calculate Rule
	if RULES['implication']['enable']:
		implication_context_score = implication_scorer.calculate_score(
			story = candidate_story,
			candidate_sentence_idx=candidate_idx,
			comparing_sentence_type="context"
		)
		implication_obstacle_score = implication_scorer.calculate_score(
			story = candidate_story,
			candidate_sentence_idx=candidate_idx,
			comparing_sentence_type="obstacle"
		)
		implication_preceding_score = implication_scorer.calculate_score(
			story = candidate_story,
			candidate_sentence_idx=candidate_idx,
			comparing_sentence_type="preceding"
		)
	else:
		implication_context_score = 0
		implication_obstacle_score = 0
		implication_preceding_score = 0
	if RULES['similarity']['enable']:
		similarity_score = similarity_scorer.calculate_score(
			story = candidate_story,
			candidate_sentence_idx=candidate_idx
		)
	else:
		similarity_score = 0
	score = implication_context_score + implication_obstacle_score + implication_preceding_score + similarity_score
	del candidate_story
	return score

def generate_next_sentence(story: ConflictStory) -> StorySentence:
	## Determine candidate_idx
	if story.num_sentences==2:
		candidate_idx = 1
	else:
		candidate_idx = story.num_sentences

	decode_params = config['candidate_generator']['decode_params']

	candidates = generator.generate(story = story, decode_params = decode_params)
	scores = []
	for candidate in candidates:
		candidate_sentence = make_story_sentence(idx = candidate_idx, value = candidate, sentence_type="other")
		## Evaluate
		score = evaluate_candidate(story=story, candidate_sentence=candidate_sentence, candidate_idx=candidate_idx)
		scores.append(score)
		del candidate_sentence
	
	best_candidate_idx = np.argmax(scores)
	## Make Final Story Sentence
	selected_sentence = make_story_sentence(idx = candidate_idx, value = candidates[best_candidate_idx], sentence_type="other")
	return selected_sentence

def generate_story(
	context: str,
	obstacle: str
) -> ConflictStory:
	## Initialize Context, Obstacle Sentences
	if context[-1]!=".":
		context = context+"."
	if obstacle[-1]!=".":
		obstacle = obstacle+"."
	context_sentence = make_story_sentence(idx = 0, value = context, sentence_type = "context")
	obstacle_sentence = make_story_sentence(idx = 2, value = obstacle, sentence_type = "obstacle")
	## Initialize Story
	story = ConflictStory(
		num_sentences = 2,
		context_idx = 0,
		obstacle_idx = 2,
		sentences = {0: context_sentence, 2: obstacle_sentence}
	)

	## Generate Loop
	for i in range(3):
		next_sentence = generate_next_sentence(story)
		story.num_sentences+=1
		story.sentences[next_sentence.idx] = next_sentence
	return story

########## Main Loop
## Load Data
df = pd.read_csv(args.data_dir, sep = '\t')
print(df.shape, df.columns)

## Generate Story
columns = ['id', 'context', 'goal', 'obstacle', 's2', 's4', 's5']
generated_dict = {k: list() for k in columns}
for i in tqdm(range(df.shape[0])):
	row = df.iloc[i]

	story = generate_story(
		context = row['context'],
		obstacle = row['obstacle']
	)

	## Make Processed Dict
	generated_dict['id'].append(row['id'])
	generated_dict['context'].append(row['context'])
	generated_dict['goal'].append(row['goal'])
	generated_dict['obstacle'].append(row['obstacle'])
	generated_dict['s2'].append(story.sentences[1].value)
	generated_dict['s4'].append(story.sentences[3].value)
	generated_dict['s5'].append(story.sentences[4].value)
generated_df = pd.DataFrame.from_dict(generated_dict)

save_dir = args.result_dir
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
generated_df.to_csv(os.path.join(save_dir, f'generated_stories-seed{SEED}.tsv'), sep = "\t", index = None)
with open(os.path.join(save_dir, 'generate_config.json'), 'w') as f:
	f.write(json.dumps(config, indent = '\t'))
