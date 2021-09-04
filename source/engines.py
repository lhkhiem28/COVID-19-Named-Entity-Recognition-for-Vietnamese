
import tqdm
import numpy as np
import torch
from metrics import *

def train_fn(
    loaders, model, device, 
    criterion, optimizer, 
    epochs, 
    ckp_path, 
):
    print("Number of Epochs: {}\n".format(epochs))
    best_micro_f1 = 0.0

    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, epochs + 1):
        print("epoch {:2}/{:2}".format(epoch, epochs) + "\n" + "-"*16)

        model.train()
        running_loss = 0.0
        running_annos, running_preds = [], []
        for sents, annos in tqdm.tqdm(loaders["train"]):
            masks = (sents != loaders["train"].dataset.tokenizer.pad_token_id).type(sents.type())
            sents, masks = sents.to(device), masks.to(device)
            annos = annos.view(-1).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(sents, masks)
                logits = outputs.logits.view(-1, outputs.logits.shape[-1])
                loss = criterion(logits, annos.long())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss = running_loss + loss.item()*sents.size(0)
            annos, preds = list(annos.detach().cpu().numpy()), list(np.argmax(logits.detach().cpu().numpy(), axis=1))
            running_annos.extend(annos), running_preds.extend(preds)

        epoch_loss = running_loss/len(loaders["train"].dataset)
        epoch_micro_f1 = entity_f1_score(
            np.array(running_annos), np.array(running_preds)
            , loaders["train"].dataset.criterion_ignored_la, loaders["train"].dataset.tag_names
            , average="micro"
        )
        print("{}-loss: {:.4f}".format("train", epoch_loss))
        print("{}-entity-micro-f1: {:.4f}".format("train", epoch_micro_f1))

        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            running_annos, running_preds = [], []
            for sents, annos in tqdm.tqdm(loaders["val"]):
                masks = (sents != loaders["val"].dataset.tokenizer.pad_token_id).type(sents.type())
                sents, masks = sents.to(device), masks.to(device)
                annos = annos.view(-1).to(device)

                outputs = model(sents, masks)
                logits = outputs.logits.view(-1, outputs.logits.shape[-1])
                loss = criterion(logits, annos.long())

                running_loss = running_loss + loss.item()*sents.size(0)
                annos, preds = list(annos.detach().cpu().numpy()), list(np.argmax(logits.detach().cpu().numpy(), axis=1))
                running_annos.extend(annos), running_preds.extend(preds)

        epoch_loss = running_loss/len(loaders["val"].dataset)
        epoch_micro_f1 = entity_f1_score(
            np.array(running_annos), np.array(running_preds)
            , loaders["val"].dataset.criterion_ignored_la, loaders["val"].dataset.tag_names
            , average="micro"
        )
        print("{}-loss: {:.4f}".format("val", epoch_loss))
        print("{}-entity-micro-f1: {:.4f}".format("val", epoch_micro_f1))

        if epoch_micro_f1 > best_micro_f1:
            best_micro_f1 = epoch_micro_f1
            torch.save(model.state_dict(), ckp_path)

    print("Finish-Best entity-micro-f1: {:.4f}".format(best_micro_f1))

def test_fn(
    test_loader, model, device, 
):
    model.to(device)

    with torch.no_grad():
        model.eval()
        running_annos, running_preds = [], []
        for sents, annos in tqdm.tqdm(test_loader):
            masks = (sents != test_loader.dataset.tokenizer.pad_token_id).type(sents.type())
            sents, masks = sents.to(device), masks.to(device)
            annos = annos.view(-1).to(device)

            outputs = model(sents, masks)
            logits = outputs.logits.view(-1, outputs.logits.shape[-1])

            annos, preds = list(annos.detach().cpu().numpy()), list(np.argmax(logits.detach().cpu().numpy(), axis=1))
            running_annos.extend(annos), running_preds.extend(preds)

    test_micro_f1 = entity_f1_score(
        np.array(running_annos), np.array(running_preds)
        , test_loader.dataset.criterion_ignored_la, test_loader.dataset.tag_names
        , average="micro"
    )
    test_macro_f1 = entity_f1_score(
        np.array(running_annos), np.array(running_preds)
        , test_loader.dataset.criterion_ignored_la, test_loader.dataset.tag_names
        , average="macro"
    )
    test_classification_report = entity_classification_report(
        np.array(running_annos), np.array(running_preds)
        , test_loader.dataset.criterion_ignored_la, test_loader.dataset.tag_names
    )
    print("{}-entity-micro-f1: {:.4f}".format("test", test_micro_f1))
    print("{}-entity-macro-f1: {:.4f}".format("test", test_macro_f1))
    print("test-classification-report:\n", test_classification_report)