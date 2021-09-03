
import numpy as np
import torch
from utils import *

class NamedEntityRecognitionDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        df, 
        tag_names, 
        tokenizer, 
    ):
        self.df = df
        self.tag_names = tag_names
        self.sentences, self.annotations = df.groupby("Sentence")["Word"].apply(list).values, df.groupby("Sentence")["Tag"].apply(list).values

        self.tokenizer = tokenizer
        self.max_length = 64
        self.criterion_ignored_class = -100

    def show_sentence(self, idx):
        sentence, annotation = self.sentences[idx], self.annotations[idx]
        for word, tag in zip(sentence, annotation):
            print("{:>40}: {}".format(word, tag))

    def __len__(self):
        dataset_len = len(self.sentences)

        return dataset_len

    def __getitem__(self, idx):
        sentence, annotation = self.sentences[idx], self.annotations[idx]
        sentence, annotation = tokenize(
            self, 
            sentence, annotation
        ).values()
        sentence, annotation = pad_and_add_special_tokens(
            self, 
            sentence, annotation
        ).values()

        return np.array(sentence), np.array(annotation)