import torch
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def label_aligning(sentence, label, tokenizer):
    sentence = " " +sentence
    tokens = tokenizer.tokenize(sentence)
    aligned_label = ['O']
    label_type = " "
    index = 0
    for token in tokens:
        if token.startswith("Ġ"):
            aligned_label.append(label[index])

            if len(label[index]) != 1:
                label_type = label[index][1:]
            else:
                label_type = label[index]

            index += 1

        else:
            if label_type != "O":
                aligned_label.append('I' + label_type)
            else:
                aligned_label.append(label_type)
    aligned_label.append('O')

    return aligned_label


def label_mapping(label):
    map_dict = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3,
                "I-LOC": 4, "B-ORG": 5, "I-ORG": 6, "B-MISC": 7, "I-MISC": 8}
    new_label = []
  
    for key in label:
        new_label.append(map_dict[key])
    return new_label


class CoNLL2003(Dataset):

    def __init__(self, dir_path, phase, max_length):
        self.path = dir_path + '/' + phase + '.txt'
        self.sentences = list()
        self.sentences_label = list()
        self.tokenizer_path = "EleutherAI/gpt-neox-20b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
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
            print('Not found target file, please check your code')

    def __getitem__(self, index):
        tokens = self.tokenizer(self.sentences[index], add_special_tokens=True,
                                truncation=True, padding='max_length',
                                max_length=self.max_length, return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze(0)
        label = label_mapping(label_aligning(self.sentences[index],
                                                  self.sentences_label[index],
                                                  tokenizer=self.tokenizer))
        if len(label) < self.max_length:
            num_padding = self.max_length - len(label)
            label.extend([-1] * num_padding)
            # label += [0] * num_padding 

        elif len(label) > self.max_length:
            num_truncation = len(label) - self.max_length
            label = label[:-num_truncation]
            # del label[-num_truncation:]

        label = torch.as_tensor(label)

        return input_ids, label

    def __len__(self):
        return self.sentences.__len__()
