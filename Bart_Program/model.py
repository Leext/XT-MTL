import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (BartConfig, BartForConditionalGeneration,
                          BartTokenizer, BartModel)


class BartCBR(nn.Module):
    def __init__(self, pretrained_model_or_path: str):
        super().__init__()
        self.bart_gen = BartForConditionalGeneration.from_pretrained(
            pretrained_model_or_path)
        self.bart = self.bart_gen.model

    def get_sent_rep(
        self,
        input_ids,
        attention_mask,
    ):
        r"""
            empty decoder_input_ids -> repeat input seq
        """
        outputs = self.bart(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.bart_gen.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError(
                "All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

        return sentence_representation

    def similarity_loss(self, sent_rep, text_simi_matrix):
        r"""
            original similarity loss in CBR paper
            Args:
                sent_rep: [B, L]
                text_simi_matrix: [B, B]
        """
        sim_mat = torch.matmul(sent_rep, sent_rep.T)
        sim_mat = F.log_softmax(sim_mat, dim=0)  # [B, B]
        loss = -torch.sum(text_simi_matrix * sim_mat)
        return loss
