import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import (
    AutoModel,
    BertConfig,
    EncoderDecoderModel,
    FunnelConfig,
    RobertaConfig,
)

from .constants import TOKENIZER_MODEL


class LMHead(nn.Module):
    """(Roberta?) Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class Encoder(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        vocab_size: int = 32100,
        masked_lm: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if model_name_or_path.lower() == "roberta":
            self.config = RobertaConfig()
        elif model_name_or_path.lower() == "bert":
            self.config = BertConfig()
        else:
            self.config = FunnelConfig()  # vocab-size?

        self.config.vocab_size = vocab_size  # len(AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True))
        self.model = AutoModel.from_config(self.config)

        if self.hparams.masked_lm:
            self.lm_head = LMHead(self.config)

    def forward(self, input_ids, attention_mask) -> torch.TensorType:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    def training_step(self, batch, batch_idx):
        assert self.hparams.masked_lm
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        prediction_scores = self.lm_head(outputs["last_hidden_state"])
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(
            prediction_scores.view(-1, self.config.vocab_size), batch["labels"].view(-1)
        )

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        assert self.hparams.masked_lm
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        prediction_scores = self.lm_head(outputs["last_hidden_state"])
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(
            prediction_scores.view(-1, self.config.vocab_size), batch["labels"].view(-1)
        )

        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        assert self.hparams.masked_lm
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        prediction_scores = self.lm_head(outputs["last_hidden_state"])
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(
            prediction_scores.view(-1, self.config.vocab_size), batch["labels"].view(-1)
        )

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )

        return optimizer

    def save(self, path: str = "models/codecse-encoder"):
        self.model.save_pretrained(path)
        return path


class EncoderForWithDFG(Encoder):
    ...


class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        encoder_pretrained_model_name_or_path: str = "roberta-base",
        decoder_pretrained_model_name_or_path: str = "gpt2",
        tie_encoder_decoder: bool = True,  # TODO: Experiment
        decoder_start_token_id: int = 0,
        pad_token_id: int = 0,
        vocab_size: int = 32100,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=encoder_pretrained_model_name_or_path,
            decoder_pretrained_model_name_or_path=decoder_pretrained_model_name_or_path,
            tie_encoder_decoder=tie_encoder_decoder,
        )

        self.model.config.decoder_start_token_id = decoder_start_token_id
        self.model.config.pad_token_id = pad_token_id
        self.model.config.vocab_size = vocab_size

    def forward(self, **kwargs) -> torch.TensorType:
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["target_input_ids"],
        )

        self.log("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["target_input_ids"],
        )

        self.log("val_loss", outputs.loss)
        return outputs.loss

    def test_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["target_input_ids"],
        )

        self.log("test_loss", outputs.loss)
        return outputs.loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )

        return optimizer

    def generate(self, input_ids: torch.TensorType):
        ...

    def save(self, path: str = "models/codecse-seq2seq"):
        self.model.save_pretrained(path)
        return path