import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel


class ClassificationModel(nn.Module):
    def __init__(self, model_path, hidden_dropout_prob, hidden_size, classifier_ids, freeze_layers=None):
        super(ClassificationModel, self).__init__()
        # Transformers architecture  not configurable
        self.transformer = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.classifier_ids = classifier_ids
        for classifier_id in classifier_ids:
            self.__setattr__(f"classifier_{classifier_id}", nn.Linear(hidden_size, 2))

        if freeze_layers:
            if freeze_layers == "TRAIN_BIAS_ONLY":
                for name, param in self.named_parameters():
                    if "transformer" in name and "weight" in name:
                        param.requires_grad = False
            else:
                for name, param in self.named_parameters():
                    for layer_type in freeze_layers:
                        if layer_type in name:  # classifier layer
                            param.requires_grad = False
                            break

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                cls_head_id=None,
                output_attn_and_hidden=False,
                inference=False):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attn_and_hidden,
            output_hidden_states=output_attn_and_hidden
        )
        # Complains if input_embeds is kept
        pooled_output = outputs[1]
        hidden_cls = self.dropout(pooled_output)

        if inference:
            logits = {}
            for cls_head in self.classifier_ids:
                classifier = self.__getattr__(f"classifier_{cls_head}")
                logits[cls_head] = classifier(hidden_cls)
            return logits
        else:
            if isinstance(cls_head_id, (int, str)):
                classifier = self.__getattr__(f"classifier_{cls_head_id}")
                logits = classifier(hidden_cls)
            else:
                logits = []
                for output, cls_head in zip(hidden_cls, cls_head_id):
                    classifier = self.__getattr__(f"classifier_{cls_head}")
                    logits.append(classifier(output))
                logits = torch.stack(logits)

            if output_attn_and_hidden:
                return logits, outputs[2], outputs[3]
            return logits, pooled_output

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'
