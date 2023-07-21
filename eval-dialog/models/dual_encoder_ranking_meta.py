# Downstream task script for domain adaptation.
# The script is modified from TOD-BERT models.
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os.path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss
from sklearn.metrics import f1_score #, average_precision_score
import numpy as np

from transformers import *
import logging

from .meta_model import BertEmbed, BertMetaEmbed

class dual_encoder_ranking(nn.Module):
    def __init__(self, args): #, num_labels, device):
        super(dual_encoder_ranking, self).__init__()
        self.args = args
        self.xeloss = nn.CrossEntropyLoss()
        self.n_gpu = args["n_gpu"]

        ### Utterance Encoder
        self.utterance_encoder = args["model_class"].from_pretrained(self.args["model_name_or_path"], cache_dir = args["cache_dir"])
        embedding_1 = BertEmbed.from_pretrained(self.args["model_name_or_path"], cache_dir = args["cache_dir"]) #TOD-BERT
        embedding_2 = BertEmbed.from_pretrained(self.args["model_name_or_path_2"], cache_dir = args["cache_dir"]) #TOD-BERT-MLM-SAMEDOMAIN
        embedding_3 = None if self.args["model_name_or_path_3"]==None else BertEmbed.from_pretrained(self.args["model_name_or_path_3"], cache_dir = args["cache_dir"]) #TOD-BERT-MLM-OTHERDOMAIN
        embedding_4 = None if self.args["model_name_or_path_4"]==None else BertEmbed.from_pretrained(self.args["model_name_or_path_4"], cache_dir = args["cache_dir"]) #TOD-BERT-MLM-OTHERDOMAIN
        embedding_5 = None if self.args["model_name_or_path_5"]==None else BertEmbed.from_pretrained(self.args["model_name_or_path_5"], cache_dir = args["cache_dir"]) #TOD-BERT-MLM-OTHERDOMAIN
        embedding_6 = None if self.args["model_name_or_path_6"]==None else BertEmbed.from_pretrained(self.args["model_name_or_path_6"], cache_dir = args["cache_dir"]) #TOD-BERT-MLM-OTHERDOMAIN
        logging.info("embedding 1: {}".format(self.args["model_name_or_path"]))
        logging.info("embedding 2: {}".format(self.args["model_name_or_path_2"]))
        logging.info("embedding 3: {}".format(self.args["model_name_or_path_3"]))
        logging.info("embedding 4: {}".format(self.args["model_name_or_path_4"]))
        logging.info("embedding 5: {}".format(self.args["model_name_or_path_5"]))
        logging.info("embedding 6: {}".format(self.args["model_name_or_path_6"]))
        self.utterance_encoder.embeddings = BertMetaEmbed(config = self.utterance_encoder.config, 
                                                          embedding_1 = embedding_1, 
                                                          embedding_2 = embedding_2,
                                                          embedding_3 = embedding_3,
                                                          embedding_4 = embedding_4,
                                                          embedding_5 = embedding_5,
                                                          embedding_6 = embedding_6,
                                                          use_average = True if self.args["use_average"] else False,
                                                          use_attention = True if self.args["use_attention"] else False,
                                                          ignore_tod = True if self.args["ignore_tod"] else False
                                                          )
        logging.info("Use Attention: {}".format(self.utterance_encoder.embeddings.use_attention))
        logging.info("Use Average: {}".format(self.utterance_encoder.embeddings.use_average))
        logging.info("Ignore TOD: {}".format(self.utterance_encoder.embeddings.ignore_tod))
        
        for param in self.utterance_encoder.named_parameters():
            if "embedding_" in param[0]:
                param[1].requires_grad=False
#             if "embedding_2" in param[0]:
#                 param[1].requires_grad=False
                
        logging.info(self.utterance_encoder)
        for param in self.utterance_encoder.named_parameters():
            logging.info("Param: {} Requires_grad: {}".format(param[0], param[1].requires_grad))

        if self.args["fix_encoder"]:
            for p in self.utterance_encoder.parameters():
                p.requires_grad = False

        ## Prepare Optimizer
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args["learning_rate"]},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args["learning_rate"]},
            ]
            return optimizer_grouped_parameters

        if self.n_gpu == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.module)

        
        self.optimizer = AdamW(optimizer_grouped_parameters,
                                 lr=args["learning_rate"],)
                                 #warmup=args["warmup_proportion"],
                                 #t_total=t_total)

    def optimize(self):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args["grad_clip"])
        self.optimizer.step()
    
    def forward(self, data):
                #input_ids, input_len, labels=None, n_gpu=1, target_slot=None):
       
        self.optimizer.zero_grad()
        
        batch_size = data["context"].size(0)
        #max_seq_len = 256

        interval = 25
        start_list = list(np.arange(0, batch_size, interval))
        end_list = start_list[1:] + [None]
        context_outputs, response_outputs = [], []
        #logging.info("start: {}".format(start_list))
        #logging.info("end: {}".format(end_list))
        for start, end in zip(start_list, end_list):
            #logging.info("{}:{}".format(start, end))
            inputs_con = {"input_ids": data["context"][start:end],
                          "attention_mask": (data["context"][start:end] > 0).long()}
            inputs_res = {"input_ids": data["response"][start:end], 
                          "attention_mask": (data["response"][start:end] > 0).long()}
            #print(inputs_con, inputs_res)
            if "bert" in self.args["model_type"]:
                context_output = self.utterance_encoder(**inputs_con)[1] #hidden_state, pooler_output
                response_output = self.utterance_encoder(**inputs_res)[1]#hidden_state, pooler_output
            elif self.args["model_type"] == "gpt2":
                context_output = self.utterance_encoder(**inputs_con)[0].mean(1)
                response_output = self.utterance_encoder(**inputs_res)[0].mean(1)
            elif self.args["model_type"] == "dialogpt":
                transformer_outputs = self.utterance_encoder.transformer(**inputs_con)
                context_output = transformer_outputs[0].mean(1)
                transformer_outputs = self.utterance_encoder.transformer(**inputs_res)
                response_output = transformer_outputs[0].mean(1)
            
#             print(self.utterance_encoder(**inputs_con))
#             print(self.utterance_encoder(**inputs_res))
            context_outputs.append(context_output.cpu())
            response_outputs.append(response_output.cpu())
        
#         logging.info(self.utterance_encoder)
#         for param in self.utterance_encoder.named_parameters():
#             logging.info("Param: {} Requires_grad: {}".format(param[0], param[1].requires_grad))
            
        # evaluation for k-to-100
        if (not self.training) and (batch_size < self.args["eval_batch_size"]): 
            response_outputs.append(self.final_response_output[:self.args["eval_batch_size"]-batch_size, :])
        
        final_context_output = torch.cat(context_outputs, 0)
        final_response_output = torch.cat(response_outputs, 0)
        
        if torch.cuda.is_available(): 
            final_context_output = final_context_output.cuda()
            final_response_output = final_response_output.cuda()
        
        if (not self.training):
            self.final_response_output = final_response_output.cpu()

        # mat
        logits = torch.matmul(final_context_output, final_response_output.transpose(1, 0))

        # loss
        labels = torch.tensor(np.arange(batch_size))
        if torch.cuda.is_available(): labels = labels.cuda()
        loss = self.xeloss(logits, labels)

        if self.training: 
            self.loss_grad = loss
            self.optimize()
        
        predictions = np.argsort(logits.detach().cpu().numpy(), axis=1) #torch.argmax(logits, -1)
        
        outputs = {"loss":loss.item(), 
                   "pred":predictions, 
                   "label":np.arange(batch_size)} 
        
        return outputs
    
    def evaluation(self, preds, labels):
        assert len(preds) == len(labels)
        
        preds = np.array(preds)
        labels = np.array(labels)
        
        def _recall_topk(preds_top10, labels, k):
            preds = preds_top10[:, -k:]
            acc = 0
            for li, label in enumerate(labels):
                if label in preds[li]: acc += 1
            acc = acc / len(labels)       
            return acc
        
        results = {"top-1": _recall_topk(preds, labels, 1), 
                   "top-3": _recall_topk(preds, labels, 3), 
                   "top-5": _recall_topk(preds, labels, 5), 
                   "top-10": _recall_topk(preds, labels, 10)}
        
        print(results)
        
        return results
