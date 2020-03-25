import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
sys.path.append(".")

from PyTorch.modeling_utils import PreTrainedModel
from PyTorch.configuration_bert import BertConfig


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}

BertLayerNorm = nn.LayerNorm

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
    "bert-base-japanese": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-pytorch_model.bin",
    "bert-base-japanese-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking-pytorch_model.bin",
    "bert-base-japanese-char": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-pytorch_model.bin",
    "bert-base-japanese-char-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking-pytorch_model.bin",
    "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/pytorch_model.bin",
    "bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/pytorch_model.bin",
    "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/pytorch_model.bin",
}


class BertEmbeddings(nn.Module):
    '''embedding层，包含token、segment、position'''
    def __init__(self,config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,config.hidden_size,padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,input_ids=None,token_type_ids=None,position_ids=None):
        # batchSize, seqLength
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(seq_length,dtype=torch.long,device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,dtype=torch.long,device=device)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 输入为三者相加
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """"""
    def __init__(self,config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden size: {},不是多头head: {}的整数倍".format(config.hidden_size,config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size // config.num_attention_heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads

        self.output_attentions = config.output_attentions  # 是否输出注意力分布

        # q、k、v(将多头注意力一起处理)
        self.query = nn.Linear(config.hidden_size,self.all_head_size)
        self.key = nn.Linear(config.hidden_size,self.all_head_size)
        self.value = nn.Linear(config.hidden_size,self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_atten_scores(self,x):
        '''将输入x的hidden size分割成多头'''
        # x [batch, seq, hidden]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,2,1,3)

    def forward(self,hidden_states,attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # batch,num_heads,seq_len,one_head_size
        query_layer = self.transpose_atten_scores(mixed_query_layer)
        key_layer = self.transpose_atten_scores(mixed_key_layer)
        value_layer = self.transpose_atten_scores(mixed_value_layer)

        # 实现softmax，自注意分布
        # batch, num_heads, seq_len, seq_len
        attention_scores = torch.matmul(query_layer,key_layer.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # mask 这里是2-，包含了attention mask与padding mask
            attention_scores = attention_scores - (2-attention_mask) * 1e6
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        # batch,seq,n_heads,one_heads
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # batch，seq，all_heads
        context_layer = context_layer.view(*new_context_layer_shape)
        return (context_layer, attention_probs) if self.output_attentions else (context_layer,)


class BertSelfOutput(nn.Module):
    '''多头注意力后接LayerNormal + 残差连接'''
    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size,eps=config.layer_norm_eps)

    def forward(self,hidden_states,input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    '''组合多头注意力和selfoutput'''
    def __init__(self,config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self,hidden_states,attention_mask=None):
        # batch, seq, hidden
        self_outputs = self.self(hidden_states,attention_mask)
        attention_output = self.output(self_outputs[0],hidden_states)
        output = (attention_output,) + self_outputs[1:]

        return output


class BertIntermediate(nn.Module):
    """自注意力层后接FF层"""
    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.intermediate_size)
        if isinstance(config.hidden_act,str):
            # 选择设定的激活函数， gelu
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self,hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """FF层后接LaymerNorm + 残差"""
    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size,eps=config.layer_norm_eps)

    def forward(self,hidden_states,input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertLayer(nn.Module):
    """一个完整的block"""
    def __init__(self,config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self,hidden_states,attention_mask=None):
        self_attention_outputs = self.attention(hidden_states,attention_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output,attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class BertEncoder(nn.Module):
    """组合多个block"""
    def __init__(self,config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,hidden_states,attention_mask=None):
        all_hidden_states = ()  # 所有层的隐藏状态
        all_attentions = ()     #所有层的注意力分布

        for i, layer_module in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            lay_outputs = layer_module(hidden_states,attention_mask)
            hidden_states = lay_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (lay_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # 最后一层的hidden states，+ all_hidden_states + all_attention_probs


class BertPooler(nn.Module):
    """最后输出的CLS token用于分类类任务"""
    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self,hidden_states):
        first_token_tensor = hidden_states[:,0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    # pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'bert'

    def _init_weights(self,module):
        """参数初始化"""
        if isinstance(module,(nn.Linear,nn.Embedding)):
            module.weight.data.normal_(mean=0.0,std=self.config.initializer_range)
        elif isinstance(module,BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class UniLMModel(BertPreTrainedModel):
    """"""
    def __init__(self,config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classification = nn.Linear(self.config.hidden_size,len(self.config.keep_words))

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self,value):
        self.embeddings.word_embeddings = value

    def forward(self,input_ids,catcu_lss=True,attention_mask=None,token_type_ids=None,position_ids=None):
        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            padding_mask = self.compute_padding_mask(input_ids)
            seq2seq_atten_mask = self.compute_attention_mask(token_type_ids)

            attention_mask = padding_mask + seq2seq_atten_mask

        embedding_output = self.embeddings(input_ids=input_ids,token_type_ids=token_type_ids,position_ids=position_ids)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask = attention_mask
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classification(sequence_output)
        outputs = (logits,) + encoder_outputs[1:]

        if catcu_lss:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            preds = logits[:,:-1,:].reshape(-1,len(self.config.keep_words)).contiguous()
            labels = input_ids[:,1:].reshape(-1).contiguous()
            loss = loss_fct(preds,labels)

            outputs = (loss,) + outputs
        return outputs

    def compute_attention_mask(self,token_type_ids):
        """计算seq2seq的mask矩阵"""
        device = token_type_ids.device
        seq_len = token_type_ids.shape[1]
        # 1,num_heads,seq,seq
        ones = torch.ones(1,self.config.num_attention_heads,seq_len,seq_len,device=device)

        # 下三角矩阵
        a_mask = torch.tril(ones)
        s_ex12 = token_type_ids.unsqueeze(1).unsqueeze(1)
        s_ex13 = token_type_ids.unsqueeze(1).unsqueeze(3)

        # batch,num_heads,seq,seq
        a_mask = (1-s_ex13) * (1-s_ex12) + a_mask * s_ex13

        # a_mask = a_mask.view(-1,seq_len,seq_len).contiguous()
        return a_mask

    def compute_padding_mask(self, input_ids):
        """计算padding部分的mask"""
        seq_len = input_ids.shape[1]
        padding_mask = ((input_ids>0) * 1).to(input_ids.device)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).expand(-1,self.config.num_attention_heads,seq_len,-1)
        return padding_mask





