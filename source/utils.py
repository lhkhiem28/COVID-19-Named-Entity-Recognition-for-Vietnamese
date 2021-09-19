
from keras.preprocessing.sequence import pad_sequences as pad

def tokenize(
    dataset, 
    sent, 
    anno, 
):
    tokenized_sent = []
    tokenized_anno = []

    for i, word in enumerate(sent):
        tokenized_word = dataset.tokenizer.convert_tokens_to_ids(dataset.tokenizer.tokenize(word))
        tokenized_sent.extend(tokenized_word)
        tokenized_anno.extend([dataset.tag_names.index(anno[i])] + [dataset.criterion_ignored_la]*(len(tokenized_word) - 1))

    return {
        "tokenized_sent": tokenized_sent, 
        "tokenized_anno": tokenized_anno, 
    }

def pad_and_add_special_tokens(
    dataset, 
    sent, 
    anno, 
):
    sent = pad([sent], dataset.max_length - 1, truncating="post", padding="post", value=dataset.specials["pad_token_id"])[0].tolist()
    anno = pad([anno], dataset.max_length - 1, truncating="post", padding="post", value=dataset.specials["pad_token_la"])[0].tolist()
    if dataset.specials["pad_token_id"] in sent:
        eos_index = sent.index(dataset.specials["pad_token_id"]) + 1
    else:
        eos_index = -1

    sent = [dataset.specials["cls_token_id"]] + sent
    anno = [dataset.specials["cls_token_la"]] + anno
    sent[eos_index] = dataset.specials["sep_token_id"]
    anno[eos_index] = dataset.specials["sep_token_la"]

    return {
        "sent": sent, 
        "anno": anno, 
    }