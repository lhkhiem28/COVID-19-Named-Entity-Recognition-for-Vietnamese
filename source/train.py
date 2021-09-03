
import argparse
import yaml
import pandas as pd
import torch
import transformers
from data import NamedEntityRecognitionDataset
from engines import train_fn

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--batch_size", type=int), parser.add_argument("--epochs", type=int)
args = parser.parse_args()

data_file = yaml.load(open(args.data_file), Loader=yaml.FullLoader)

device = torch.device("cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
model = transformers.RobertaForTokenClassification.from_pretrained(args.model_name, num_labels=data_file["num_tags"])
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

train_df = pd.read_csv(data_file["train_df_path"])
val_df = pd.read_csv(data_file["val_df_path"])

train_loader = torch.utils.data.DataLoader(
    NamedEntityRecognitionDataset(
        df=train_df, 
        tag_names=data_file["tag_names"], 
        tokenizer=tokenizer, 
    ), 
    num_workers=4, 
    batch_size=args.batch_size, 
    shuffle=True, 
)
val_loader = torch.utils.data.DataLoader(
    NamedEntityRecognitionDataset(
        df=val_df, 
        tag_names=data_file["tag_names"], 
        tokenizer=tokenizer, 
    ), 
    num_workers=4, 
    batch_size=args.batch_size*2, 
)

loaders = {
    "train": train_loader, 
    "val": val_loader, 
}

train_fn(
    loaders, 
    model, 
    criterion, 
    optimizer, 
    device=device, 
    epochs=args.epochs, 
    ckp_path="../ckps/{}.pt".format(args.model_name.split("/")[-1])
)