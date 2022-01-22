import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class DistilBERTModel(nn.Module):

    def __init__(self, pretrained_model_path, num_labels, output_hidden_states=False):

        super(DistilBERTModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model_path, output_hidden_states=output_hidden_states)
        self.config = DistilBertConfig.from_pretrained(pretrained_model_path)
        self.config.update({'num_labels': num_labels})

        self.head = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.head)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):

        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        output = hidden_state[:, 0]
        output = self.head(output)
        output = output.view(-1)

        return output
