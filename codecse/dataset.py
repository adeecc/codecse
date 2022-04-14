from typing import Optional
from tokenizers import Tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import datasets as ds

from .constants import DATASET_NAME, NUM_WORKERS, TOKENIZER_MODEL


class PLToNLDM(pl.LightningDataModule):

    loader_cols = [
        "input_ids",
        "attention_mask",
        "target_input_ids",
        "target_attention_mask",
    ]

    def __init__(
        self,
        pl: str = "java",
        max_seq_length: int = 128,
        padding: str = "max_length",
        batch_size: int = 2,
    ):
        super().__init__()
        self.pl = pl
        self.max_seq_length = max_seq_length
        self.padding = padding
        self.batch_size = batch_size

        self.tokenizer: Tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_MODEL, use_fast=True
        )

    def prepare_data(self) -> None:
        ds.load_dataset(DATASET_NAME, self.pl)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ds.load_dataset(DATASET_NAME, self.pl)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._to_features,
                batched=True,
                # remove_columns=[]
            )

            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_cols
            ]

        self.dataset.set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )

    def _to_features(self, batch, indices=None):
        # TODO: Add DataFlow graph to the encoded features
        lang_name_code_pair = list(zip(batch["language"], batch["code"]))

        features = self.tokenizer(
            lang_name_code_pair,
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=True,
        )

        targets = self.tokenizer(
            batch["docstring"],
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=True,
        )

        features["target_input_ids"] = targets["input_ids"]
        features["target_attention_mask"] = targets["attention_mask"]

        # TODO: Check if works

        return features


class AugmentedPLToNLDM(PLToNLDM):
    loader_cols = [
        "input_ids",
        "attention_mask",
        "dfg_to_code",
        "dfg_to_dfg",
        "pos_idx",
        "target_input_ids",
        "target_attention_mask",
    ]

    def __init__(
        self,
        pl: str = "java",
        max_seq_length: int = 128,
        padding: str = "max_length",
        batch_size: int = 2,
    ):
        super().__init__(pl, max_seq_length, padding, batch_size)

    def _to_features(self, batch, indices=None):
        # TODO: Add DataFlow graph to the encoded features
        lang_name_code_pair = list(zip(batch["language"], batch["code"]))

        features = self.tokenizer(
            lang_name_code_pair,
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=True,
        )

        targets = self.tokenizer(
            batch["docstring"],
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=True,
        )

        features["target_input_ids"] = targets["input_ids"]
        features["target_attention_mask"] = targets["attention_mask"]

        # TODO: Check if works

        return features


class PLToNLDMForMaskedLM(PLToNLDM):

    loader_cols = ["input_ids", "attention_mask", "labels"]

    def __init__(
        self,
        pl: str = "java",
        max_seq_length: int = 128,
        padding: str = "max_length",
        batch_size: int = 2,
        mlm_probability: float = 0.15,
    ):
        super().__init__(
            pl=pl, max_seq_length=max_seq_length, padding=padding, batch_size=batch_size
        )
        self.data_collator = DataCollatorForLanguageModeling(
            mlm_probability=mlm_probability, tokenizer=self.tokenizer
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )

    def _to_features(self, batch, indices=None):
        # TODO: Add DataFlow graph to the encoded features
        lang_name_code_pair = list(zip(batch["language"], batch["code"]))

        features = self.tokenizer(
            lang_name_code_pair,
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=True,
        )

        return features


if __name__ == "__main__":
    dm = PLToNLDMForMaskedLM(mlm_probability=1.0)
    # dm = AugmentedPLToNLDM()
    dm.prepare_data()
    dm.setup("fit")

    print(next(iter(dm.train_dataloader())))

    # fmt: off
    import IPython; IPython.embed(); exit(0);
    # fmt: on
