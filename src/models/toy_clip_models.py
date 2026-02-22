# This is a modified CLIP model, which is designed for toy experiments.
# The image encoder is a ResNet18, and the text encoder is a mini BERT model.
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class ImageEncoder(nn.Module):
    def __init__(self, feat_dim:int=512, output_dim=64,  img_model='resnet18', unfreeze_n_blocks=4):
        super().__init__()
        self.encoder = torch.hub.load('pytorch/vision', img_model, pretrained=True)
        # NOTE: remove parts of the backbone. 
        del self.encoder.fc
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        # unfreeze the norm layer
        #for param in self.encoder.norm.parameters():
        #    param.requires_grad = True

        self.fc = nn.Linear(feat_dim, output_dim)
        
    def forward(self, x):
        for name, module in self.encoder.named_children():
            #if name == 'fc':  # unfreeze the last few layers
            #    break
            x = module(x)

        feat = x.squeeze()
        #feat = self.encoder.forward_features(x)
        x = self.fc(feat)
        return x

class TextEncoder(nn.Module):
    def __init__(self, output_dim=64, lang_model="distilbert-base-uncased-finetuned-sst-2-english", unfreeze_n_blocks=4):
        super().__init__()
        self.lang_model = lang_model

        self.tokenizer = AutoTokenizer.from_pretrained(lang_model)
        self.encoder = AutoModel.from_pretrained(lang_model)
        #print (self.encoder)       
        for param in self.encoder.parameters():
            param.requires_grad = True
        """
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # unfreeze the last few encoder layers
        for layer in self.encoder.encoder.layer[ - unfreeze_n_blocks :]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # unfreeze the pooler layer
        for param in self.encoder.pooler.parameters():
            param.requires_grad = True
        """
        
        self.fc = nn.Linear(self.encoder.config.hidden_size, output_dim)
    
    def forward(self, input_ids, attention_mask=None):
        # the last hidden state of the [CLS] token is used as the sentence representation
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = self.fc(x)
        return x