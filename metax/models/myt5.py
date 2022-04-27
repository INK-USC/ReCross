import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right  # for transformers==2.9.0

from .utils import label_smoothed_nll_loss

# https://huggingface.co/transformers/v4.8.1/_modules/transformers/models/t5/modeling_t5.html#T5ForConditionalGeneration 

class MyT5(T5ForConditionalGeneration):
    def forward(
            self,
            input_ids=None,
            decoder_input_ids=None,
            is_training=False,    
            return_dict=None,
            **kwargs
            ):
        '''
        Migrating from transformers 3.x to 4.x, `decoder_cached_state` becomes `past_key_values`
        '''
        if is_training:
            _decoder_input_ids = shift_tokens_right(
                decoder_input_ids,
                self.config.pad_token_id,
                self.config.eos_token_id # only needed when transformers==4.7.0
            )
        else:
            _decoder_input_ids = decoder_input_ids
        
        outputs = super().forward(
                input_ids=input_ids,
                decoder_input_ids=_decoder_input_ids,
                **kwargs
            )
        
        if return_dict:
            lm_logits = outputs.logits
            # lm_logits = F.linear(outputs.logits,
            #                     self.shared.weight.T,
            #                     #  self.model.shared.weight,
            #                     #  bias=self.final_logits_bias
            #                     )  
        else:
            lm_logits = outputs[0]
            # lm_logits = F.linear(outputs[0],
            #                     self.shared.weight.T,
            #                     #  self.model.shared.weight,
            #                     #  bias=self.final_logits_bias
            #                     )  
        if is_training:
            lprobs = F.log_softmax(lm_logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(
                lprobs,
                decoder_input_ids,
                epsilon=0.1,
                ignore_index=self.config.pad_token_id)
            return loss

        if return_dict:
            outputs.logits = lm_logits
            return outputs
        else:
            return (lm_logits, ) + outputs[1:]
        