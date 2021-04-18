"""
@Time : 2021/4/159:31
@Auth : 周俊贤
@File ：model.py
@DESCRIPTION:

"""
from torch import nn
from transformers import BertModel


class DuEEEvent_model(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(DuEEEvent_model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None):
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        sequence_output, pooled_output = output[0], output[1]
        logits = self.classifier(sequence_output)
        return logits

class DuEECls_model(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(DuEECls_model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None):
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        sequence_output, pooled_output = output[0], output[1]
        logits = self.classifier(pooled_output)
        return logits

if __name__ == '__main__':
    model = DuEEEvent_model("/data/zhoujx/prev_trained_model/rbt3", num_classes=60)
    a = 1