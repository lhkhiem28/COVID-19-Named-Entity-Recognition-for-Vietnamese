
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
        self.sents, self.annos = df.groupby("Sentence")["Word"].apply(list).values, df.groupby("Sentence")["Tag"].apply(list).values

        self.tokenizer = tokenizer
        self.max_length = 64
        self.criterion_ignored_class = -100

    def show_sent(self, idx):
        sent, anno = self.sents[idx], self.annos[idx]
        for word, tag in zip(sent, anno):
            print("{:>40}: {}".format(word, tag))

    def __len__(self):
        dataset_len = len(self.sents)

        return dataset_len

    def __getitem__(self, idx):
        sent, anno = self.sents[idx], self.annos[idx]
        sent, anno = tokenize(
            self, 
            sent, anno
        ).values()
        sent, anno = pad_and_add_special_tokens(
            self, 
            sent, anno
        ).values()

        return np.array(sent), np.array(anno)