
import argparse
import yaml
import pandas as pd
import torch, torchcrf
import transformers
from data import Dataset
from engines import train_fn

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str)
parser.add_argument("--hyps_file", type=str)
args = parser.parse_args()

data_file = yaml.load(open(args.data_file), Loader=yaml.FullLoader)
hyps_file = yaml.load(open(args.hyps_file), Loader=yaml.FullLoader)

train_loader = torch.utils.data.DataLoader(
    Dataset(
        df=pd.read_csv(data_file["train_df_path"]), 
        tag_names=data_file["tag_names"], 
        tokenizer=transformers.AutoTokenizer.from_pretrained(hyps_file["encoder"], use_fast=False), 
    ), 
    num_workers=hyps_file["num_workers"], 
    batch_size=hyps_file["batch_size"], 
    shuffle=True, 
)
val_loader = torch.utils.data.DataLoader(
    Dataset(
        df=pd.read_csv(data_file["val_df_path"]), 
        tag_names=data_file["tag_names"], 
        tokenizer=transformers.AutoTokenizer.from_pretrained(hyps_file["encoder"], use_fast=False), 
    ), 
    num_workers=hyps_file["num_workers"], 
    batch_size=hyps_file["batch_size"]*2, 
)

loaders = {
    "train": train_loader, 
    "val": val_loader, 
}

model = transformers.RobertaForTokenClassification.from_pretrained(hyps_file["encoder"], num_labels=data_file["num_tags"])
if hyps_file["use_crf"]:
    criterion = torchcrf.CRF(num_tags=data_file["num_tags"], batch_first=True)
else:
    criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(hyps_file["lr"]))

train_fn(
    loaders, model, torch.device(hyps_file["device"]), hyps_file["device_ids"], 
    criterion, 
    optimizer, 
    epochs=hyps_file["epochs"], 
    ckp_path="../ckps/{}.pt".format(hyps_file["encoder"].split("/")[-1]), 
)