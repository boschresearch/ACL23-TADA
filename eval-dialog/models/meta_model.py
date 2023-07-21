# TADA Model script for domain adaptation.
# The script is modified from 
# (a) adapter-transformers v2.2.0 https://github.com/adapter-hub/adapter-transformers/blob/adapters2.2.0/src/transformers/models/bert/modeling_bert.py
# and
# (b) FAME https://github.com/boschresearch/adversarial_meta_embeddings/blob/main/src/models.py
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

import math
import os
import warnings
from dataclasses import dataclass
from collections import Counter
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.adapters.model_mixin import ModelWithHeadsAdaptersMixin
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward
)
from transformers import AutoTokenizer
from transformers.adapters.models.bert import BertModelAdaptersMixin
from itertools import repeat

from transformers.utils import logging
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPreTrainedModel


logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertEmbed(BertModelAdaptersMixin, BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)

        self._init_adapter_modules()

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.pre_transformer_forward(**kwargs)

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        embedding_output = self.invertible_adapters_forward(embedding_output)

        return embedding_output
    

class BertMetaDomainEmbed(BertEmbed, AutoTokenizer):
    def __init__(self, 
                 config,
                 embedding_1: BertEmbed,
                 tokenizer_1: AutoTokenizer,
                 embedding_2: BertEmbed,
                 tokenizer_2: AutoTokenizer,
                 embedding_3: Optional[BertEmbed]=None,
                 tokenizer_3: Optional[AutoTokenizer]=None,
                 embedding_4: Optional[BertEmbed]=None,
                 tokenizer_4: Optional[AutoTokenizer]=None,
                 embedding_5: Optional[BertEmbed]=None,
                 tokenizer_5: Optional[AutoTokenizer]=None,
                 embedding_6: Optional[BertEmbed]=None,
                 tokenizer_6: Optional[AutoTokenizer]=None,
                 method: str="dynamic", #dynamic, whitespace, truncation
                 attn_size: int=20,
                 use_average: bool=True,
                 use_attention: bool=True,
                 use_feature: bool=False,
                 ignore_tod: bool=False,
                 ):
        super().__init__(config)
        self.config = config
        self.embedding_1 = embedding_1
        self.embedding_2 = embedding_2
        self.embedding_3 = embedding_3
        self.embedding_4 = embedding_4
        self.embedding_5 = embedding_5
        self.embedding_6 = embedding_6
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.tokenizer_4 = tokenizer_4
        self.tokenizer_5 = tokenizer_5
        self.tokenizer_6 = tokenizer_6
        self.attn_size = 20
        self.method = method
        self.use_average = use_average
        self.use_attention = use_attention
        self.use_feature = use_feature
        self.ignore_tod = ignore_tod
        self.num_embeddings = sum(x is not None for x in [self.embedding_1, self.embedding_2, self.embedding_3, self.embedding_4, self.embedding_5, self.embedding_6])
        #self.cpu()
        #self._init_adapter_modules()
        #self.init_weights()
        if torch.cuda.is_available():
            self.cuda()
    
    def convert_offsets_to_tuple(self, pt_batch, tokenizer):
        return_offsets = []
        return_toks = []
        for i, seq in enumerate(pt_batch.offset_mapping):
            offset_tuple = [tuple(l) for l in seq.cpu().detach().numpy()]
            toks = tokenizer.convert_ids_to_tokens(pt_batch['input_ids'][i])
            input_ids = pt_batch['input_ids'][i].cpu().detach().numpy()
            word_ids = pt_batch.word_ids(i)
            return_offsets.append(list(zip(toks, word_ids, input_ids, offset_tuple)))
        return return_offsets
    
    def combine_subword(self, embs):
        common_begin, common_end = Counter(), Counter()
        for emb in embs:
            for tok, wordid, input_id, (begin, end) in emb:
                if (begin, end)!=(0, 0):
                    common_begin[begin] += 1
                    common_end[end] += 1

        common_begin = [x for x, c in common_begin.most_common() if c == len(embs)]
        common_end = [x for x, c in common_end.most_common() if c == len(embs)]
        pairs = [(b, e) for b, e in zip(common_begin, common_end)]
        return pairs

    def combine_whitespace(self, embs):
        pairs = []
        cur_word_id = -1
        cur_word_begin = -1
        cur_word_end = -1
    
        for tok, wordid, input_id, (begin, end) in embs[0]:
            if wordid!=None:
                if wordid > cur_word_id:
                    if cur_word_id >= 0:
                        pairs.append((cur_word_begin, cur_word_end))
                    cur_word_id = wordid
                    cur_word_begin = begin
                cur_word_end = end        
        pairs.append((cur_word_begin, cur_word_end))
        return pairs

    def matched_pairs(self, pairs, emb, pad="x"):
        offset_map = [offset for tok, wordid, input_id, offset in emb]
        final = []

        pre = [i for i, (tok, wordid, input_id, (s, e)) in enumerate(emb) if (s, e)==(0, 0) and tok=="[CLS]"]
        after = [i for i, (tok, wordid, input_id, (s, e)) in enumerate(emb) if (s, e)==(0, 0) and tok!="[CLS]"]
        final.append(pre)
        for i, (begin, end) in enumerate(pairs):
            results = [i for i, (tok, wordid, input_id, (s, e)) in enumerate(emb) if begin<=s and e<=end and (s, e)!=(0, 0)]
            final.append(results)
        if pad == "duplicate":
            after = [[i] for i in after]
            final+=after
            final = [x for item in final for x in repeat(item, len(item))]
        else:
            final.append(after)
        return final
    
    def same_diff(self, a, b):
        final = []
        i = 0
        j = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                final.append(b[j])
                i += 1
                j += 1
            elif len(a[i]) >= len(b[j]):
                final+=[b[j]]*len(a[i])
                i += len(a[i])
                j += len(b[j])
            else:
                final.append(b[j])
                i += len(a[i])
                j += len(b[j])
        return final

    def realign_pairs(self, input_pairs, pad_length=128):
        new_pairs = []
        for i, k in enumerate(input_pairs):
            if i==0:
                new_pairs.append(k)
                if len(new_pairs[i])<pad_length:
                    new_pairs[i]+=[new_pairs[i][-1]]*(pad_length-len(new_pairs[i]))
            else:
                new_pairs.append(self.same_diff(input_pairs[0], input_pairs[i]))
                if len(new_pairs[i])<pad_length:
                    new_pairs[i]+=[new_pairs[i][-1]]*(pad_length-len(new_pairs[i]))
        return new_pairs
    
    def prepare_batch(self, batch_texts, tokenizer, device, **kwargs):
        return tokenizer(batch_texts, **kwargs).to(device)
    
    def prepare_batch_text(self, pt_batch, tokenizer, method="dynamic"): 
        all_inputs = []
        for k in pt_batch["input_ids"]:
            new_input = [tok for tok in k if tok not in [tokenizer.cls_token_id, tokenizer.pad_token_id]]
            all_inputs.append(new_input[:-1])

        if method in ["orig_subword", "orig_whitespace", "dynamic", "whitespace", "truncation", "trun_whitespace"]:
            batch_texts = [tokenizer.decode(k) for k in all_inputs]
            return batch_texts
        else:
            raise ValueError("Please pick the method among: dynamic, whitespace, truncation")
    
    def pad_tensor(self, selected_embedding, selected_tokenizer, device, max_length, start, end):
        seq = "a"
        trial_tensor = selected_tokenizer(seq, padding= "max_length", max_length = max_length, return_attention_mask=False, return_tensors="pt").to(device)
        return selected_embedding(**trial_tensor)[:, start:end][0]
    
    def get_count(self, input_list):
        return list(map(lambda x:len(x), input_list))

    def forward(
        self,
        **kwargs
    ):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        batch_texts = self.prepare_batch_text(kwargs, self.tokenizer_1, method=self.method) #Convert the tokenized input_ids into texts
        
        batch_tensor = BatchTensor(self.embedding_1, self.tokenizer_1)
        embedding_1_output, embedding_1_convert_offsets = batch_tensor.convert(batch_texts, max_length=kwargs["input_ids"].shape[1])
        batch_tensor = BatchTensor(self.embedding_2, self.tokenizer_2)
        embedding_2_output, embedding_2_convert_offsets = batch_tensor.convert(batch_texts, max_length=kwargs["input_ids"].shape[1])
        embs = list(zip(embedding_1_convert_offsets, embedding_2_convert_offsets))
        if self.embedding_3 is not None:
            batch_tensor = BatchTensor(self.embedding_3, self.tokenizer_3)
            embedding_3_output, embedding_3_convert_offsets = batch_tensor.convert(batch_texts, max_length=kwargs["input_ids"].shape[1])
            embs = list(zip(embedding_1_convert_offsets, embedding_2_convert_offsets, embedding_3_convert_offsets))
        if self.embedding_4 is not None:
            batch_tensor = BatchTensor(self.embedding_4, self.tokenizer_4)
            embedding_4_output, embedding_4_convert_offsets = batch_tensor.convert(batch_texts, max_length=kwargs["input_ids"].shape[1])
            embs = list(zip(embedding_1_convert_offsets, embedding_2_convert_offsets, embedding_3_convert_offsets, embedding_4_convert_offsets))
        if self.embedding_5 is not None:
            batch_tensor = BatchTensor(self.embedding_5, self.tokenizer_5)
            embedding_5_output, embedding_5_convert_offsets = batch_tensor.convert(batch_texts, max_length=kwargs["input_ids"].shape[1])
            embs = list(zip(embedding_1_convert_offsets, embedding_2_convert_offsets, embedding_3_convert_offsets, embedding_4_convert_offsets, embedding_5_convert_offsets))
        if self.embedding_6 is not None:
            batch_tensor = BatchTensor(self.embedding_6, self.tokenizer_6)
            embedding_6_output, embedding_6_convert_offsets = batch_tensor.convert(batch_texts, max_length=kwargs["input_ids"].shape[1])
            embs = list(zip(embedding_1_convert_offsets, embedding_2_convert_offsets, embedding_3_convert_offsets, embedding_4_convert_offsets, embedding_5_convert_offsets, embedding_6_convert_offsets))
        
        if self.method == "orig_subword":
            final_combine = [self.matched_pairs(self.combine_subword(emb), emb[0]) for emb in embs]
        elif self.method == "orig_whitespace":
            final_combine = [self.matched_pairs(self.combine_whitespace(emb), emb[0]) for emb in embs]
        elif self.method in ["dynamic", "truncation"]:
            final_final = []
            for i in range(len(embs[0])):
                final_x = [self.matched_pairs(self.combine_subword(emb), emb[i], pad="duplicate") for emb in embs]
                final_final.append(final_x)
            final_final = list(map(list, zip(*final_final)))
            final_combine = [self.realign_pairs(final, pad_length=kwargs["input_ids"].shape[1]) for final in final_final]
        elif self.method in ["whitespace", "trun_whitespace"]:
            final_final = []
            for i in range(len(embs[0])):
                final_x = [self.matched_pairs(self.combine_whitespace(emb), emb[i], pad="duplicate") for emb in embs]
                final_final.append(final_x)
            final_final = list(map(list, zip(*final_final)))
            final_combine = [self.realign_pairs(final, pad_length=kwargs["input_ids"].shape[1]) for final in final_final]
        else:
            raise ValueError("Please pick the method among: dynamic, whitespace, truncation")
        
        stack_embeddings = torch.stack([embedding_1_output, embedding_2_output, embedding_3_output, embedding_4_output, embedding_5_output, embedding_6_output], dim=1)
            
        emb_all = torch.zeros(stack_embeddings.shape)
        #print(emb_all.shape)
        if self.method in ["orig_subword", "orig_whitespace"]:
        # Stack six of them
            for k in range(len(final_combine)):
                stack_before_mean = torch.stack([stack_embeddings[k][:, i].mean(1) for i in final_combine[k]], dim=1) # should be able to do attention here, pad to the max value
                #print(stack_before_mean.shape, stack_embeddings.shape[2], stack_before_mean.shape[1], stack_embeddings.shape[2])
                padding_to_max = self.pad_tensor(self.embedding_1, self.tokenizer_1, device, stack_embeddings.shape[2], stack_before_mean.shape[1], stack_embeddings.shape[2])
                emb_all[k][0] = torch.cat((stack_before_mean[0], padding_to_max))
                padding_to_max = self.pad_tensor(self.embedding_2, self.tokenizer_2, device, stack_embeddings.shape[2], stack_before_mean.shape[1], stack_embeddings.shape[2])
                emb_all[k][1] = torch.cat((stack_before_mean[1], padding_to_max))
                padding_to_max = self.pad_tensor(self.embedding_3, self.tokenizer_3, device, stack_embeddings.shape[2], stack_before_mean.shape[1], stack_embeddings.shape[2])
                emb_all[k][2] = torch.cat((stack_before_mean[2], padding_to_max))
                padding_to_max = self.pad_tensor(self.embedding_4, self.tokenizer_4, device, stack_embeddings.shape[2], stack_before_mean.shape[1], stack_embeddings.shape[2])
                emb_all[k][3] = torch.cat((stack_before_mean[3], padding_to_max))
                padding_to_max = self.pad_tensor(self.embedding_5, self.tokenizer_5, device, stack_embeddings.shape[2], stack_before_mean.shape[1], stack_embeddings.shape[2])
                emb_all[k][4] = torch.cat((stack_before_mean[4], padding_to_max))
                padding_to_max = self.pad_tensor(self.embedding_6, self.tokenizer_6, device, stack_embeddings.shape[2], stack_before_mean.shape[1], stack_embeddings.shape[2])
                emb_all[k][5] = torch.cat((stack_before_mean[5], padding_to_max))
                
        elif self.method in ["dynamic", "whitespace"]:
            for k in range(stack_embeddings.shape[0]):
                for j in range(stack_embeddings.shape[1]):
                    emb_all[k][j] = torch.stack([stack_embeddings[k][j][i].mean(0) for i in final_combine[k][j]], dim=0)
                    
        elif self.method in ["truncation", "trun_whitespace"]:
            for k in range(stack_embeddings.shape[0]):
                min_length = [min(i) for i in list(zip(*[self.get_count(k) for k in final_combine[k][:stack_embeddings.shape[1]]]))]
                for j in range(stack_embeddings.shape[1]):
                    emb_all[k][j] = torch.stack([stack_embeddings[k][j][i][:min_length[d]].mean(0) for d, i in enumerate(final_combine[k][j])], dim=0)
            
        emb_all = emb_all.permute(1, 0, 2, 3)
        if self.ignore_tod:
            emb_all = emb_all[1:, :] #ignore the tod-bert one
            
        if torch.cuda.is_available():
            emb_all = emb_all.cuda()
        #return emb_all
        if self.use_attention:
            #print("Use Attention")
            self.attention = BertAttentionEmbed(emb_all.shape[-1], self.use_feature, self.attn_size, self.use_average)
            #self.attention.vlinear.weight.data.normal_(std=0.086)
            embedding_output, _ = self.attention(emb_all)
        elif self.use_average and not self.use_attention:
            #print("Use Average")
            embedding_output = torch.mean(emb_all, dim=0)
        else:
            embedding_output = self.embedding_1_output
        return embedding_output
    
class BertMetaEmbed(BertEmbed):
    def __init__(self, 
                 config,
                 embedding_1: BertEmbed,
                 embedding_2: BertEmbed,
                 embedding_3: Optional[BertEmbed]=None,
                 embedding_4: Optional[BertEmbed]=None,
                 embedding_5: Optional[BertEmbed]=None,
                 embedding_6: Optional[BertEmbed]=None,
                 attn_size: int=20,
                 use_average: bool=True,
                 use_attention: bool=True,
                 use_feature: bool=False,
                 ignore_tod: bool=False,
                 ):
        super().__init__(config)
        self.config = config
        self.embedding_1 = embedding_1
        self.embedding_2 = embedding_2
        self.embedding_3 = embedding_3
        self.embedding_4 = embedding_4
        self.embedding_5 = embedding_5
        self.embedding_6 = embedding_6
        self.attn_size = 20
        self.use_average = use_average
        self.use_attention = use_attention
        self.use_feature = use_feature
        self.ignore_tod = ignore_tod
        self.num_embeddings = sum(x is not None for x in [self.embedding_1, self.embedding_2, self.embedding_3, self.embedding_4, self.embedding_5, self.embedding_6])
        #self._init_adapter_modules()
        #self.init_weights()
        
    def forward(
        self,
        **kwargs
    ):
        self.embedding_1_output = self.embedding_1(**kwargs)
        self.embedding_2_output = self.embedding_2(**kwargs)
        emb_all = torch.stack((self.embedding_1_output, self.embedding_2_output))
        if self.embedding_3 is not None:
            self.embedding_3_output = self.embedding_3(**kwargs)
            emb_all = torch.stack((self.embedding_1_output, self.embedding_2_output, self.embedding_3_output))
        if self.embedding_4 is not None:
            self.embedding_4_output = self.embedding_4(**kwargs)
            emb_all = torch.stack((self.embedding_1_output, self.embedding_2_output, self.embedding_3_output, self.embedding_4_output))
        if self.embedding_5 is not None:
            self.embedding_5_output = self.embedding_5(**kwargs)
            emb_all = torch.stack((self.embedding_1_output, self.embedding_2_output, self.embedding_3_output, self.embedding_4_output, self.embedding_5_output))
        if self.embedding_6 is not None:
            self.embedding_6_output = self.embedding_6(**kwargs)
            emb_all = torch.stack((self.embedding_1_output, self.embedding_2_output, self.embedding_3_output, self.embedding_4_output, self.embedding_5_output, self.embedding_6_output))
        
        if self.ignore_tod:
            emb_all = torch.stack((self.embedding_2_output, self.embedding_3_output, self.embedding_4_output, self.embedding_5_output, self.embedding_6_output))
            
        # The input should be the same as the results of tokenizer(input_sequences)
        if self.use_attention:
            #print("Use Attention")
            self.attention = BertAttentionEmbed(emb_all.shape[-1], self.use_feature, self.attn_size, self.use_average)
            #self.attention.vlinear.weight.data.normal_(std=0.086)
            embedding_output, _ = self.attention(emb_all)
        elif self.use_average and not self.use_attention:
            #print("Use Average")
            embedding_output = torch.mean(emb_all, dim=0)
        else:
            embedding_output = self.embedding_1_output
        return embedding_output
    

class BertAttentionEmbed(nn.Module):
    """
    An embedding-augmented attention layer where the attention weight is
    a = V . tanh(Ux + Wf)
    where x is the input embedding, and f is a word feature vector.
    if feature_size=0, it's only attention weights, no features included.
    U: input_size * attn_size
    W: feature_size * attn_size
    V: attn_size * 1
    """
    def __init__(self, input_size, feature_size, attn_size, use_attn_sum=True, fixed_weights=None):
        super(BertAttentionEmbed, self).__init__()
        self.input_size = input_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.use_attn_sum = use_attn_sum
        self.fixed_weights = fixed_weights
        
        if self.fixed_weights is False:
            self.fixed_weights = None
            
        if self.fixed_weights is None:
            self.ulinear = nn.Linear(self.input_size, self.attn_size)
            if self.feature_size > 0:
                self.wlinear = nn.Linear(self.feature_size, self.attn_size)
            self.vlinear = nn.Linear(self.attn_size, 1)
            self.init_weights()
            #print(f'Initialized Attention with size ({self.input_size, self.feature_size, self.attn_size})')
        else:
            print(f'Use fixed weights for Attention: {self.fixed_weights}')
        if torch.cuda.is_available(): 
            self.cuda()
    
    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        if self.feature_size > 0:
            self.wlinear.weight.data.normal_(std=0.001)
        #self.vlinear.weight.data.normal_(std=0.086)
        self.vlinear.weight.data.zero_() # use zero to give uniform attention at the beginning
    
    def forward(self, x):
        """
        x : batch_size * seq_len * emb_num * input_size  -> the results of torch.stack((embed_output1, embed_output2, ...))
        f : batch_size * seq_len * emb_num * feature_size -> we don't need it at current moment
        """
        emb_num, batch_size, seq_len, input_size = x.size() # need to double check here, a single embedding output: batch_size * seq_len, input_size
        proj_x = self.ulinear(x.contiguous())
        scores = self.vlinear(torch.tanh(proj_x))
        weights = F.softmax(scores, dim=0)
        
        if self.use_attn_sum:
            outputs = torch.bmm(weights.view(emb_num, batch_size*seq_len, 1).permute(1, 2, 0), x.reshape(emb_num, batch_size*seq_len, input_size).permute(1, 0, 2))
            outputs = outputs.view(batch_size, seq_len, input_size)
        return outputs, weights

class BatchTensor:
    def __init__(self, embedding, tokenizer):
        self.embedding = embedding
        self.tokenizer = tokenizer
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    def prepare_batch(self, batch_texts, **kwargs):
        return self.tokenizer(batch_texts, **kwargs).to(self.device)
    
    def convert_offsets_to_tuple(self, pt_batch):
        return_offsets = []
        return_toks = []
        for i, seq in enumerate(pt_batch.offset_mapping):
            offset_tuple = [tuple(l) for l in seq.cpu().detach().numpy()]
            toks = self.tokenizer.convert_ids_to_tokens(pt_batch['input_ids'][i])
            input_ids = pt_batch['input_ids'][i].cpu().detach().numpy()
            word_ids = pt_batch.word_ids(i)
            return_offsets.append(list(zip(toks, word_ids, input_ids, offset_tuple)))
        return return_offsets
    
    def convert(self, batch_texts, max_length=128):
        batch_tensor = self.prepare_batch(batch_texts, padding="max_length", 
                        truncation=True, max_length=max_length, return_tensors="pt", is_split_into_words=False, 
                        return_attention_mask=False, return_offsets_mapping=True)
        embedding_output = self.embedding(**batch_tensor)
        convert_offsets = self.convert_offsets_to_tuple(batch_tensor)
        return embedding_output, convert_offsets
