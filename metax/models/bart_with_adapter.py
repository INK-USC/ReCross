# from https://github.com/INK-USC/hypter
from transformers.models.bart.modeling_bart import EncoderLayer, DecoderLayer, BartEncoder, BartDecoder, BartModel, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.configuration_bart import BartConfig
from transformers.configuration_utils import PretrainedConfig

from .utils import label_smoothed_nll_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

# Typing
from typing import Any
from torch import Tensor

import copy


class BartWithAdapterConfig(BartConfig):
    def __init__(
        self,
        activation_dropout=0.0,
        activation_function="gelu",
        vocab_size=50265,
        d_model=1024,
        encoder_ffn_dim=4096,
        encoder_layers=12,
        encoder_attention_heads=16,
        decoder_ffn_dim=4096,
        decoder_layers=12,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        attention_dropout=0.0,
        dropout=0.1,
        max_position_embeddings=1024,
        init_std=0.02,
        classifier_dropout=0.0,
        num_labels=3,
        is_encoder_decoder=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        normalize_before=False,
        add_final_layer_norm=False,
        scale_embedding=False,
        normalize_embedding=True,
        static_position_embeddings=False,
        add_bias_logits=False,
        adapter_dim=64,
        adapt_layer_norm=False,
        unfreeze_hyper_encoder=False,
        **common_kwargs
    ):

        if "hidden_size" in common_kwargs:
            raise ValueError("hidden size is called d_model")

        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **common_kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model  # encoder_embed_dim and decoder_embed_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std  # Normal(0, this parameter)
        self.activation_function = activation_function

        # Params introduced for Mbart
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.normalize_embedding = normalize_embedding  # True for mbart, False otherwise
        self.normalize_before = normalize_before  # combo of fairseq's encoder_ and decoder_normalize_before
        self.add_final_layer_norm = add_final_layer_norm

        # Params introduced for Marian
        self.add_bias_logits = add_bias_logits
        self.static_position_embeddings = static_position_embeddings

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        # Classifier stuff
        self.classif_dropout = classifier_dropout

        # Adapter
        self.adapter_dim = adapter_dim
        self.generator_hdim = int(self.d_model * 0.25)  # TODO: make it a tunable hp.
        self.adapt_layer_norm = adapt_layer_norm
        self.unfreeze_hyper_encoder = unfreeze_hyper_encoder    # TODO: should be

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight, gain=0.0000001)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class EncoderLayerWithAdapter(EncoderLayer):

    def __init__(self, config: BartConfig):
        super(EncoderLayerWithAdapter, self).__init__(config)

        self.adapter_dim = config.adapter_dim
        # self.adapter_down_weight = torch.zeros(self.embed_dim, self.adapter_dim)
        # self.adapter_down_bias = torch.zeros(self.adapter_dim)

        # self.adapter_up_weight = torch.zeros(self.adapter_dim, self.embed_dim)
        # self.adapter_up_bias = torch.zeros(self.embed_dim)
        self.adapter_down_layer = Linear(self.embed_dim, self.adapter_dim)
        self.adapter_up_layer = Linear(self.adapter_dim, self.embed_dim)

    def adapter_down(self, x):
        # print(x.size())
        # print(self.adapter_down_weight.size())
        # z = x * self.adapter_down_weight
        # print(z.size())
        # return F.linear(x, self.adapter_down_weight.t(), self.adapter_down_bias)
        # return x * self.adapter_down_weight + self.adapter_down_bias
        return self.adapter_down_layer(x)

    def adapter_up(self, x):
        # return F.linear(x, self.adapter_up_weight.t(), self.adapter_up_bias)
        # return x * self.adapter_up_weight + self.adapter_up_bias
        return self.adapter_up_layer(x)

    def forward(self, x, encoder_padding_mask):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(
            query=x, key=x, key_padding_mask=encoder_padding_mask, need_weights=self.output_attentions
        )
        x = F.dropout(x, p=self.dropout, training=self.training)

        residual_adapter = x
        x = self.adapter_down(x)
        x = self.activation_fn(x)
        x = self.adapter_up(x)
        x = residual_adapter + x

        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn_weights

class DecoderLayerWithAdapter(DecoderLayer):
    def __init__(self, config: BartConfig):
        super(DecoderLayerWithAdapter, self).__init__(config)

        self.adapter_dim = config.adapter_dim

        # self.adapter_down_weight = torch.zeros(self.embed_dim, self.adapter_dim)
        # self.adapter_down_bias = torch.zeros(self.adapter_dim)

        # self.adapter_up_weight = torch.zeros(self.adapter_dim, self.embed_dim)
        # self.adapter_up_bias = torch.zeros(self.embed_dim)
        self.adapter_down_layer = Linear(self.embed_dim, self.adapter_dim)
        self.adapter_up_layer = Linear(self.adapter_dim, self.embed_dim)

    def adapter_down(self, x):
        # return F.linear(x, self.adapter_down_weight.t(), self.adapter_down_bias)
        return self.adapter_down_layer(x)

    def adapter_up(self, x):
        # return F.linear(x, self.adapter_up_weight.t(), self.adapter_up_bias)
        return self.adapter_up_layer(x)

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            need_weights=self.output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)

        residual_adapter = x
        x = self.adapter_down(x)
        x = self.activation_fn(x)
        x = self.adapter_up(x)
        x = residual_adapter + x

        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding

class BartEncodeWithAdapter(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens):
        super(BartEncodeWithAdapter, self).__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [EncoderLayerWithAdapter(config) for _ in range(config.encoder_layers)]
        )

class BartDecoderWithAdapter(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super(BartDecoderWithAdapter, self).__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [DecoderLayerWithAdapter(config) for _ in range(config.decoder_layers)]
        )

class BartModelWithAdapter(BartModel):
    def __init__(self, config: BartConfig):
        super(BartModelWithAdapter, self).__init__(config)
        self.encoder = BartEncodeWithAdapter(config, self.shared)
        self.decoder = BartDecoderWithAdapter(config, self.shared)

class BartForConditionalGenerationWithAdapter(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartModelWithAdapter(config)
        self.model = base_model
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

class MyBartWithAdapter(BartForConditionalGenerationWithAdapter):
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):

        if is_training:
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
        else:
            _decoder_input_ids = decoder_input_ids

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            # loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.config.pad_token_id)
            # loss = loss_fct(lm_logits.view(-1, self.config.vocab_size),
            #                   decoder_input_ids.view(-1))
            lprobs = F.log_softmax(lm_logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(lprobs, decoder_input_ids, epsilon=0.1, ignore_index=self.config.pad_token_id)
            return loss
        return (lm_logits, ) + outputs[1:]

    def encoders(self):
        return self.model.encoder.layers

    def decoders(self):
        return self.model.decoder.layers

    def backup_layer_norm_parameters(self):
        for encoder in self.encoders():
            encoder.self_attn_layer_norm_bc = copy.deepcopy(encoder.self_attn_layer_norm)
        for decoder in self.decoders():
            decoder.self_attn_layer_norm_bc = copy.deepcopy(decoder.self_attn_layer_norm)

    def restore_layer_norm_parameters(self):
        for encoder in self.encoders():
            encoder.self_attn_layer_norm = copy.deepcopy(encoder.self_attn_layer_norm_bc)
        for decoder in self.decoders():
            decoder.self_attn_layer_norm = copy.deepcopy(decoder.self_attn_layer_norm_bc)

    def get_task_params(self) -> Tensor:
        '''
        External use, gets task-specific parameters (in this case, adapter weights)
        '''
        d_model = self.config.d_model
        d_adapter = self.config.adapter_dim
        layer_params = []
        for i, encoder_layer in enumerate(self.encoders()):
            layer_params.append(self.get_params_from_layer(d_model, d_adapter, encoder_layer))

        for i, decoder_layer in enumerate(self.decoders()):
            layer_params.append(self.get_params_from_layer(d_model, d_adapter, decoder_layer))

        return torch.stack(layer_params)

    def get_params_from_layer(self, d_model: int, d_adapter: int, layer: Any):
        '''
        Internal use, gets adapter parameters for the given encoder or decoder layer
        '''
        dw = layer.adapter_down_layer.weight.view(d_model*d_adapter)
        uw = layer.adapter_up_layer.weight.view(d_model*d_adapter)
        db = layer.adapter_down_layer.bias
        ub = layer.adapter_up_layer.bias

        # Concat all of them on the source dimension
        return torch.cat((dw, uw, db, ub), 0)

    def apply_task_params(self, generated_params: Tensor):
        '''
        External use, takes a tensor of generated parameters and applies them to the model adapter layers
        '''
        encoder_params, decoder_params = generated_params[:self.config.encoder_layers], generated_params[self.config.encoder_layers:]

        d_model = self.config.d_model
        d_adapter = self.config.adapter_dim

        for p, encoder_layer in zip(encoder_params, self.encoders()):
            self.apply_params_to_layer(p, d_model, d_adapter, encoder_layer)

        for p, decoder_layer in zip(decoder_params, self.decoders()):
            self.apply_params_to_layer(p, d_model, d_adapter, decoder_layer)

    # TODO [chrismiller]: Validate typing
    # Layer has to be any (I think) because could be EncoderLayerWithAdapter or DecoderLayerWithAdapter
    def apply_params_to_layer(self, p: Tensor, d_model: int, d_adapter: int, layer: Any):
        '''
        Internal use, apply a tensor of parameters to a single encoder or decoder layer
        '''
        # dw, db: down weight, down bias
        # uw, ub: up weight, up bias
        dw, uw, db, ub = p[0:d_model*d_adapter], \
                        p[d_model*d_adapter:d_model*d_adapter*2], \
                        p[d_model*d_adapter*2:d_model*d_adapter*2+d_adapter], \
                        p[d_model*d_adapter*2+d_adapter:d_model*d_adapter*2+d_adapter+d_model]
        # Since these are the weights directly, the shape is reversed from what we would expect
        # e.g. down weight is d_adapter x d_model rather than d_model x d_adapter
        layer.adapter_down_layer.weight = torch.nn.Parameter(dw.view(d_adapter, d_model))
        layer.adapter_down_layer.bias = torch.nn.Parameter(db.view(d_adapter))
        layer.adapter_up_layer.weight = torch.nn.Parameter(uw.view(d_model, d_adapter))
        layer.adapter_up_layer.bias = torch.nn.Parameter(ub.view(d_model))

        if self.config.adapt_layer_norm:
            layer.self_attn_layer_norm.weight.data = layer.self_attn_layer_norm.weight.data + p[-2*d_model: -1*d_model]
            layer.self_attn_layer_norm.bias.data = layer.self_attn_layer_norm.bias.data + p[-1*d_model:]
