from transformers import AutoModel, AutoTokenizer, AutoConfig, LlamaTokenizer, GPTNeoModel, AutoModelForSequenceClassification
from torch import nn
import torch
import math
from torch.nn.init import xavier_uniform_

from copy import deepcopy as cp
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

import bitsandbytes as bnb
from peft import LoraConfig, PeftConfig, get_peft_model
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)

class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, pooled_output):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
        
class FineTunedModel(torch.nn.Module):
    def __init__(self, label_num, config, d_model, latent_size=1024, pretrained_model_name='bert-base-uncased', \
                 model_type='non-llm', dropout=0.1, ortho_student=False):
        super(FineTunedModel, self).__init__()

        self.config = config
        self.latent_size = latent_size
        self.pretrained_model_name = pretrained_model_name
        self.model_type = model_type

        if model_type == 'non-llm':
            if 'deberta' in pretrained_model_name.lower():
                self.encoder = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, config=config)
            else:
                self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=config)
        else:
            if 'opt' in pretrained_model_name:
                self.encoder = AutoModelForCausalLM.from_pretrained(pretrained_model_name, config=config)
            else:
                compute_dtype = getattr(torch, "float16")

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=False,
                )

                self.encoder = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name,
                    device_map="auto"
                )

                self.encoder.config.use_cache = False
                self.encoder.config.pretraining_tp = 1

                peft_config = LoraConfig(
                    lora_alpha=16,
                    lora_dropout=0.1,
                    r=64,
                    bias="none",
                    task_type="TOKEN_CLS",
                    )

                self.encoder = get_peft_model(self.encoder, peft_config)

        self.drop = nn.Dropout(dropout)
        self.ortho_student = ortho_student
        if ortho_student:
            self.projection_head  = torch.nn.utils.parametrizations.orthogonal(nn.Linear(self.config.hidden_size, self.latent_size, bias=False))
        else:
            self.projection_head = nn.Linear(self.config.hidden_size, self.latent_size, bias=False)
        self.output_head1 = SequenceClassificationHead(self.config.hidden_size, label_num)
        self.output_head2 = SequenceClassificationHead(self.latent_size, label_num)

        self._init_weights()

    def _init_weights(self):
        self.projection_head.weight.data.normal_(mean=0.0, std=0.02)
        if self.projection_head.bias is not None:
            self.projection_head.bias.data.zero_()

    def forward(self, task_name, src=None, mask=None, token_type_ids=None, image=None):
        
        if self.model_type == 'non-llm':
            if 'deberta' in self.pretrained_model_name.lower():
                outputs = self.encoder.deberta(
                    src,
                    attention_mask=mask,
                    token_type_ids=token_type_ids)

                pooled_output = self.encoder.pooler(outputs[0])

            else:
                outputs = self.encoder(
                src,
                attention_mask=mask,
                token_type_ids=token_type_ids)
            
                encoder_output = outputs[0]
                pooled_output = outputs[1]
        else:
            if 'opt' in self.pretrained_model_name:
                outputs = self.encoder(
                    src,
                    output_hidden_states=True)
                
                pooled_output = outputs[2][-1][:,-1,:]
            else:
                outputs = self.encoder(
                    src,
                    attention_mask=mask, output_hidden_states=True)
            
                pooled_output = outputs[1][-1][:,0,:]

        out1 = self.output_head1(pooled_output)
        proj = self.projection_head(pooled_output)
        out2 = self.output_head2(proj)

        if task_name == 'sts-b':
            out1 = nn.ReLU()(out1)
            out2 = nn.ReLU()(out2)
        
        return (out1, out2, proj) 
        
class TeacherImageModel(torch.nn.Module):
    def __init__(self, label_num, device, latent_size=1024, d_model=512, model_id="CompVis/stable-diffusion-v1-4", dtype=torch.float32, dropout=0.1):
        super(TeacherImageModel, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = d_model
        self.device = device

        self.h1 = nn.Linear(self.latent_size,self.hidden_size)
        self.h2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

        self.projection_head = torch.nn.utils.parametrizations.orthogonal(nn.Linear(self.latent_size, self.latent_size, bias=False))
        self.output_head1 = SequenceClassificationHead(self.hidden_size, label_num)
        self.output_head2 = SequenceClassificationHead(self.latent_size, label_num)

        self._init_weights()
        
    def _init_weights(self):
        self.h1.weight.data.normal_(mean=0.0, std=0.02)
        if self.h1.bias is not None:
            self.h1.bias.data.zero_()

        self.h2.weight.data.normal_(mean=0.0, std=0.02)
        if self.h2.bias is not None:
            self.h2.bias.data.zero_()

    def forward(self, text, image, task_name):
        h = self.drop2(nn.ReLU()(self.h2(self.drop1(nn.ReLU()(self.h1(image[:,0,:]))))))
        out1 = self.output_head1(h)
        proj = self.projection_head(image[:,0,:])

        out2 = self.output_head2(self.drop3(proj))
        
        if task_name == 'sts-b':
            out1 = nn.ReLU()(out1)
            out2 = nn.ReLU()(out2)

        return (out1, out2, proj)
    
class FineTunedModel2(torch.nn.Module):
    def __init__(self, label_num, config, d_model, latent_size=1024, pretrained_model_name='bert-base-uncased', \
                 model_type='non-llm', dropout=0.1):
        super(FineTunedModel2, self).__init__()

        self.config = config
        self.latent_size = latent_size
        self.pretrained_model_name = pretrained_model_name
        self.model_type = model_type

        if model_type == 'non-llm':
            if 'deberta' in pretrained_model_name.lower():
                self.encoder = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, config=config)
            else:
                self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=config)
        else:
            compute_dtype = getattr(torch, "float16")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
            )

            self.encoder = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                device_map="auto"
            )

            self.encoder.config.use_cache = False
            self.encoder.config.pretraining_tp = 1

            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type="TOKEN_CLS",
                )

            self.encoder = get_peft_model(self.encoder, peft_config)

        self.drop = nn.Dropout(dropout)
        self.projection_head = torch.nn.utils.parametrizations.orthogonal(nn.Linear(self.config.hidden_size, self.latent_size, bias=True))
        self.output_head1 = SequenceClassificationHead(self.config.hidden_size, label_num)
        self.output_head2 = SequenceClassificationHead(self.latent_size, label_num)


    def _init_weights(self):
        self.projection_head.weight.data.normal_(mean=0.0, std=0.02)
        if self.projection_head.bias is not None:
            self.projection_head.bias.data.zero_()

    def forward(self, task_name, src=None, mask=None, token_type_ids=None, image=None):
        
        if self.model_type == 'non-llm':
            if 'deberta' in self.pretrained_model_name.lower():
                outputs = self.encoder.deberta(
                    src,
                    attention_mask=mask,
                    token_type_ids=token_type_ids)

                pooled_output = self.encoder.pooler(outputs[0])

            else:
                outputs = self.encoder(
                src,
                attention_mask=mask,
                token_type_ids=token_type_ids)
            
                encoder_output = outputs[0]
                pooled_output = outputs[1]
        else:
            outputs = self.encoder(
                src,
                attention_mask=mask, output_hidden_states=True)
            
            pooled_output = outputs[1][-1][:,0,:]

        out1 = self.output_head1(pooled_output)
        proj = self.projection_head(pooled_output)
        out2 = self.output_head2(proj)

        if task_name == 'sts-b':
            out1 = nn.ReLU()(out1)
            out2 = nn.ReLU()(out2)
        
        return (out1, out2, proj) 