import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class DistilBERT(nn.Module):

    def __init__(self, pretrained_model_path, num_labels, output_hidden_states=False):

        super(DistilBERT, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model_path, output_hidden_states=output_hidden_states)
        self.config = DistilBertConfig.from_pretrained(pretrained_model_path)
        self.config.update({'num_labels': num_labels})

        self.linear = nn.Linear(self.config.hidden_size, num_labels)
        self.softmax = nn.Softmax()
        self._init_weights(self.linear)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):

        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        output = hidden_state[:, 0]
        output = self.softmax(self.linear(output))

        return output
