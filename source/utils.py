
from keras.preprocessing.sequence import pad_sequences as pad

def tokenize(
    dataset, 
    sentence, 
    annotation, 
):
    tokenized_sentence = []
    tokenized_annotation = []

    for i, word in enumerate(sentence):
        tokenized_word = dataset.tokenizer.convert_tokens_to_ids(dataset.tokenizer.tokenize(word))
        tokenized_sentence.extend(tokenized_word)
        tokenized_tag = [dataset.tag_names.index(annotation[i])] + [dataset.criterion_ignored_class]*(len(tokenized_word) - 1)
        tokenized_annotation.extend(tokenized_tag)

    return {
        "tokenized_sentence": tokenized_sentence, 
        "tokenized_annotation": tokenized_annotation, 
    }

def pad_and_add_special_tokens(
    dataset, 
    sentence, 
    annotation, 
):
    sentence = pad([sentence], dataset.max_length - 1, truncating="post", padding="post", value=dataset.tokenizer.pad_token_id)[0].tolist()
    annotation = pad([annotation], dataset.max_length - 1, truncating="post", padding="post", value=dataset.criterion_ignored_class)[0].tolist()
    if dataset.tokenizer.pad_token_id in sentence:
        eos_index = sentence.index(dataset.tokenizer.pad_token_id) + 1
    else:
        eos_index = -1

    sentence = [dataset.tokenizer.cls_token_id] + sentence
    annotation = [dataset.criterion_ignored_class] + annotation
    sentence[eos_index] = dataset.tokenizer.sep_token_id
    annotation[eos_index] = dataset.criterion_ignored_class

    return {
        "sentence": sentence, 
        "annotation": annotation, 
    }