import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaPreTrainedModel
import torch.nn.functional as F
from model import AttenLSTM


class CustomRobertaForPipeline(RobertaPreTrainedModel):
    def __init__(self, config, device="gpu"):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.classifier = AttenLSTM(input_size=1024, hidden_size=256, batch_first=True, dropout=0.5, bidirectional=True,
                              num_layers=2, device=device)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                   position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        features = F.pad(outputs.last_hidden_state, (0, 0, 0, 512 - outputs.last_hidden_state.size(1)))
        logits = self.classifier(features)
        return logits


device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
config = RobertaConfig.from_pretrained("roberta-large", num_labels=2)
model = CustomRobertaForPipeline.from_pretrained("roberta-large", config=config, device=device_name).to(device)
model.eval()


def eval_one(model, input):
    item = input.replace("\n", " ").replace("  ", " ").strip()
    tokens = tokenizer.encode(item)
    if len(tokens) > 512:
        tokens = tokens[:512]
        print("!!!Input too long. Truncated to first 512 tokens.")
    outputs = model(torch.tensor(tokens).unsqueeze(0).to(device))
    pred = torch.max(outputs.data, 1)[1]
    (gpt_prob, hum_prob) = F.softmax(outputs.data, dim=1)[0]
    return pred[0].data, 100 * gpt_prob, 100 * hum_prob

try:
    while True:
        a = input("Please input the text to be evaluated: (0 for gpt, 1 for human)\n")
        if a == "exit":
            raise KeyboardInterrupt
        model.classifier.load_state_dict(torch.load("../Pretrained/Unified_Task1.pth"))
        print("- Decision of GPT-Written: {}, Probability: GPT: {:.4f}%, Human: {:.4f}%.".format(*eval_one(model, a)))
        model.classifier.load_state_dict(torch.load("../Pretrained/Unified_Task2.pth"))
        print("- Decision of GPT-Completed: {}, Probability: GPT: {:.4f}%, Human: {:.4f}%.".format(*eval_one(model, a)))
        model.classifier.load_state_dict(torch.load("../Pretrained/Unified_Task3.pth"))
        print("- Decision of GPT-Polished: {}, Probability: GPT: {:.4f}%, Human: {:.4f}%.".format(*eval_one(model, a)))
        model.classifier.load_state_dict(torch.load("../Pretrained/Unified_Task123.pth"))
        print("- Decision of GPT-Generated (any kind): {}, Probability: GPT: {:.4f}%, Human: {:.4f}%.\n".format(*eval_one(model, a)))
except KeyboardInterrupt:
    exit()
