
import tqdm
import numpy as np
import torch
from metrics import entity_f1_score, classification_report

def train_fn(
    loaders, 
    model, 
    criterion, 
    optimizer, 
    device, 
    epochs, 
    ckp_path
):
    print("Number of Epochs: {}\n".format(epochs))
    best_f1_micro = 0.0
    best_f1_macro = 0.0

    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, epochs + 1):
        print("epoch {:2}/{:2}".format(epoch, epochs) + "\n" + "-"*16)

        model.train()
        running_annotations, running_predictions = [], []
        for sentences, annotations in tqdm.tqdm(loaders["train"]):
            attention_masks = (sentences != loaders["train"].dataset.tokenizer.pad_token_id).type(sentences.type())
            sentences, attention_masks = sentences.to(device), attention_masks.to(device)
            annotations = annotations.view(-1).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(sentences, attention_masks)
                logits = outputs.logits.view(-1, outputs.logits.shape[-1])
                loss = criterion(logits, annotations.long())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            annotations, predictions = list(annotations.detach().cpu().numpy()), list(np.argmax(logits.detach().cpu().numpy(), axis=1))
            running_annotations.extend(annotations), running_predictions.extend(predictions)

        epoch_f1_micro = entity_f1_score(
            np.array(running_annotations), np.array(running_predictions)
            , loaders["train"].dataset.criterion_ignored_class, loaders["train"].dataset.tag_names
            , average="micro"
        )
        epoch_f1_macro = entity_f1_score(
            np.array(running_annotations), np.array(running_predictions)
            , loaders["train"].dataset.criterion_ignored_class, loaders["train"].dataset.tag_names
            , average="macro"
        )
        print("{}-entity-f1-micro: {:.4f}".format("train", epoch_f1_micro))
        print("{}-entity-f1-macro: {:.4f}".format("train", epoch_f1_macro))

        with torch.no_grad():
            model.eval()
            running_annotations, running_predictions = [], []
            for sentences, annotations in tqdm.tqdm(loader):
                attention_masks = (sentences != loader.dataset.tokenizer.pad_token_id).type(sentences.type())
                sentences, attention_masks = sentences.to(device), attention_masks.to(device)
                annotations = annotations.view(-1).to(device)

                outputs = model(sentences, attention_masks)
                logits = outputs.logits.view(-1, outputs.logits.shape[-1])
                loss = criterion(logits, annotations.long())

                annotations, predictions = list(annotations.detach().cpu().numpy()), list(np.argmax(logits.detach().cpu().numpy(), axis=1))
                running_annotations.extend(annotations), running_predictions.extend(predictions)

        epoch_f1_micro = entity_f1_score(
            np.array(running_annotations), np.array(running_predictions)
            , loader.dataset.criterion_ignored_class, loader.dataset.tag_names
            , average="micro"
        )
        epoch_f1_macro = entity_f1_score(
            np.array(running_annotations), np.array(running_predictions)
            , loader.dataset.criterion_ignored_class, loader.dataset.tag_names
            , average="macro"
        )
        print("{}-entity-f1-micro: {:.4f}".format("val", epoch_f1_micro))
        print("{}-entity-f1-macro: {:.4f}".format("val", epoch_f1_macro))

        if epoch_f1_micro > best_f1_micro:
            best_f1_micro = epoch_f1_micro
            best_f1_macro = epoch_f1_macro
            torch.save(model.state_dict(), ckp_path)

    print("Finish-Best val-f1-micro: {:.4f}".format(best_f1_micro))
    print("Finish-Best val-f1-macro: {:.4f}".format(best_f1_macro))

def val_fn(
    loader, 
    model, 
    device, 
):
    model.to(device)

    with torch.no_grad():
        model.eval()
        running_annotations, running_predictions = [], []
        for sentences, annotations in tqdm.tqdm(loader):
            attention_masks = (sentences != loader.dataset.tokenizer.pad_token_id).type(sentences.type())
            sentences, attention_masks = sentences.to(device), attention_masks.to(device)
            annotations = annotations.view(-1).to(device)

            outputs = model(sentences, attention_masks)
            logits = outputs.logits.view(-1, outputs.logits.shape[-1])

            annotations, predictions = list(annotations.detach().cpu().numpy()), list(np.argmax(logits.detach().cpu().numpy(), axis=1))
            running_annotations.extend(annotations), running_predictions.extend(predictions)

    epoch_f1_micro = entity_f1_score(
        np.array(running_annotations), np.array(running_predictions)
        , loader.dataset.criterion_ignored_class, loader.dataset.tag_names
        , average="micro"
    )
    epoch_f1_macro = entity_f1_score(
        np.array(running_annotations), np.array(running_predictions)
        , loader.dataset.criterion_ignored_class, loader.dataset.tag_names
        , average="macro"
    )
    print("{}-entity-f1-micro: {:.4f}".format("test", epoch_f1_micro))
    print("{}-entity-f1-macro: {:.4f}".format("test", epoch_f1_macro))
    report = classification_report(
        np.array(running_annotations), np.array(running_predictions)
        , loader.dataset.criterion_ignored_class, loader.dataset.tag_names
    )
    print("classification-report:")
    print(report)