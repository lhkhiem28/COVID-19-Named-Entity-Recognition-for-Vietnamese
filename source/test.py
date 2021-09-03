
import argparse
import yaml
import pandas as pd
import torch
import transformers
from data import NamedEntityRecognitionDataset
from engines import val_fn

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--batch_size", type=int)
args = parser.parse_args()

data_file = yaml.load(open(args.data_file), Loader=yaml.FullLoader)

device = torch.device("cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
model = transformers.RobertaForTokenClassification.from_pretrained(args.model_name, num_labels=data_file["num_tags"])
model.load_state_dict(torch.load("../ckps/{}.pt".format(args.model_name.split("/")[-1]), map_location=device))

test_df = pd.read_csv(data_file["test_df_path"])

test_loader = torch.utils.data.DataLoader(
    NamedEntityRecognitionDataset(
        df=test_df, 
        tag_names=data_file["tag_names"], 
        tokenizer=tokenizer, 
    ), 
    num_workers=4, 
    batch_size=args.batch_size, 
)

val_fn(
    test_loader, 
    model, 
    device=device, 
)