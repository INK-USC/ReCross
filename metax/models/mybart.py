import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right  # for transformers==2.9.0

from .utils import label_smoothed_nll_loss


class MyBart(BartForConditionalGeneration):
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_training=False,
        **unused,
    ):
        if is_training:
            _decoder_input_ids = shift_tokens_right(
                decoder_input_ids,
                self.config.pad_token_id,
                self.config.eos_token_id # only needed when transformers==4.7.0
            )
        else:
            _decoder_input_ids = decoder_input_ids

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=_decoder_input_ids,
            encoder_outputs=encoder_outputs, # need
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions, # need
            output_hidden_states=output_hidden_states, # need
            return_dict=return_dict,
        )

        lm_logits = F.linear(outputs[0],
                             self.model.shared.weight,
                             bias=self.final_logits_bias)
        if is_training:
            lprobs = F.log_softmax(lm_logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(
                lprobs,
                decoder_input_ids,
                epsilon=0.1,
                ignore_index=self.config.pad_token_id)
            if not return_dict:
                return loss
            outputs['loss'] = loss
            return outputs
        if not return_dict:
            return (lm_logits, ) + outputs[1:]

        outputs['logits'] = lm_logits
        return outputs
