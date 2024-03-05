import torch
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def label_aligning(sentence, label, tokenizer):
	tokens = tokenizer.tokenize(sentence)
	aligned_label = ['O']
	label_type = ''
	for token in tokens:
		if token.startswith("Ġ"):
			aligned_label.append(label)
			if label.__len__() != 1:
				label_type = label[2:]
		else:
			aligned_label.append('I-' + label_type)
	aligned_label.append('O')

	return aligned_label


def label_mapping(label):
	map_dict = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3,
	            "I-LOC": 4, "B-ORG": 5, "I-ORG": 6, "B-MISC": 7, "I-MISC": 8}
	new_label = [map_dict[key] for key in label]
	return new_label


class CoNLL2003(Dataset):

	def __init__(self, dir_path, phase, max_length):
		self.path = dir_path + phase
		self.sentences = list()
		self.sentences_label = list()
		self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
		self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
		self.max_length = max_length
		try:
			with open(self.path, 'r') as datafile:
				all_contents = datafile.readlines()
				sentence = list()
				sentence_label = list()
				for line in all_contents:
					if not line.strip():
						self.sentences.append(' '.join(sentence))
						self.sentences_label.append(sentence_label)
						sentence = []
						sentence_label = []
					else:
						content = line.strip().split()
						sentence.append(content[0])
						sentence_label.append(content[-1])
		except FileNotFoundError:
			print('Not found target file')

	def __getitem__(self, item):
		tokens = self.tokenizer(self.sentences[item], add_special_tokens=True,
		                        truncation=True, padding='max_length',
		                        max_len=self.max_length, return_tensors='pt')
		input_ids = tokens['input_ids']
		label = map(label_mapping, label_aligning(self.sentences[item],
		                                          self.sentences_label[item],
		                                          tokenizer=self.tokenizer))
		label = torch.as_tensor(label)

		return input_ids, label

	def __len__(self):
		return self.sentences.__len__()