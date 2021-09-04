
import yaml
import numpy as np
import torch
import transformers

class NamedEntityRecognizer():
    def __init__(
        self, 
        data_file, 
        hyps_file, 
    ):
        self.data_file = yaml.load(open(data_file), Loader=yaml.FullLoader)
        self.hyps_file = yaml.load(open(hyps_file), Loader=yaml.FullLoader)

        self.tag_names = self.data_file["tag_names"]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.hyps_file["model"], use_fast=False)

        self.device = torch.device(self.hyps_file["device"])
        self.model = transformers.RobertaForTokenClassification.from_pretrained(self.hyps_file["model"], num_labels=self.data_file["num_tags"])
        self.model.load_state_dict(torch.load("../ckps/{}.pt".format(self.hyps_file["model"].split("/")[-1]), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sent, show_sent=False):
        original_sent = sent

        sent = self.tokenizer.encode(sent)
        with torch.no_grad():
            output = self.model(torch.tensor([sent]).to(self.device))
            pred = list(np.argmax(output[0].squeeze(0).detach().cpu().numpy(), axis=1))[1:-1]

        accumulated_subwords = 0
        for i, word in enumerate(original_sent.split()):
            tokenized_word = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
            if len(tokenized_word) > 1:
                del pred[i + accumulated_subwords + 1:i + accumulated_subwords + len(tokenized_word)]

        if show_sent:
            for word, tag in zip(original_sent.split(), pred):
                print("{:>40}: {}".format(word, self.tag_names[tag]))

        return [(word, tag) for word, tag in zip(original_sent.split(), pred)]