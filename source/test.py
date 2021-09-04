
import argparse
import yaml
import pandas as pd
import torch
import transformers
from data import NamedEntityRecognitionDataset
from engines import test_fn

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str)
parser.add_argument("--hyps_file", type=str)
args = parser.parse_args()

data_file = yaml.load(open(args.data_file), Loader=yaml.FullLoader)
hyps_file = yaml.load(open(args.hyps_file), Loader=yaml.FullLoader)

test_loader = torch.utils.data.DataLoader(
    NamedEntityRecognitionDataset(
        df=pd.read_csv(data_file["test_df_path"]), 
        tag_names=data_file["tag_names"], 
        tokenizer=transformers.AutoTokenizer.from_pretrained(hyps_file["model"], use_fast=False), 
    ), 
    num_workers=hyps_file["num_workers"], 
    batch_size=hyps_file["batch_size"]*2, 
)

model = transformers.RobertaForTokenClassification.from_pretrained(hyps_file["model"], num_labels=data_file["num_tags"])
model.load_state_dict(torch.load("../ckps/{}.pt".format(hyps_file["model"].split("/")[-1]), map_location=torch.device(hyps_file["device"])))

test_fn(
    test_loader, model, torch.device(hyps_file["device"]), 
)