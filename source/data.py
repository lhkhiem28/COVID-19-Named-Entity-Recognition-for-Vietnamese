
import numpy as np
import torch
import viet_text_tools as vtts
from utils.preprocessing import *

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        df, 
        tag_names, 
        tokenizer, 
    ):
        self.df = df
        self.df["sent"] = self.df["sent"].apply(vtts.normalize_diacritics)
        self.tag_names = tag_names
        self.sents, self.annos = [words.split() for words in list(df["sent"].values)], [tags.split() for tags in list(df["anno"].values)]

        self.tokenizer = tokenizer
        self.max_length = 64
        self.criterion_ignored_la = -100
        self.specials = {
            "pad_token_id": self.tokenizer.pad_token_id, 
            "cls_token_id": self.tokenizer.cls_token_id, 
            "sep_token_id": self.tokenizer.sep_token_id, 
            "pad_token_la": self.criterion_ignored_la, 
            "cls_token_la": self.criterion_ignored_la, 
            "sep_token_la": self.criterion_ignored_la, 
        }

    def show_sent(self, idx):
        sent, anno = self.sents[idx], self.annos[idx]
        print("{:>40}: {}".format("word", "tag"))
        for i in range(len(sent)):
            print("{:>40}: {}".format(sent[i], anno[i]))

    def __len__(self):
        dataset_len = len(self.df)

        return dataset_len

    def __getitem__(self, idx):
        sent, anno = self.sents[idx], self.annos[idx]
        sent, anno = pad_and_add_special_tokens(
            self, 
            *tokenize(
                self, 
                sent, anno
            ).values()
        ).values()
        sent, anno = np.array(sent), np.array(anno)

        return sent, anno