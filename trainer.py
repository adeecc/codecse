import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2Model

from codecse.dataset import (
    AugmentedCodeToTextDataModule,
    CodeToTextDataModule,
    CodeToTextDataModuleForMaskedLM,
)

from codecse.model import Encoder, Seq2Seq

from codecse.constants import (
    AVAIL_GPUS,
)


def train_masked_lm():
    # dm = AugmentedCodeToTextDataModule()
    dm = CodeToTextDataModuleForMaskedLM(batch_size=8)

    dm.prepare_data()
    dm.setup("fit")

    encoder = Encoder("roberta", masked_lm=True, vocab_size=len(dm.tokenizer))
    trainer = pl.Trainer(
        max_epochs=10,
        gpus=AVAIL_GPUS,
        log_every_n_steps=2,
        precision=16,
        stochastic_weight_avg=True,
        logger=TensorBoardLogger("logs/main/masked_lm/"),
    )

    trainer.fit(encoder, dm)


def train_code_to_text():
    dm = CodeToTextDataModule(batch_size=1)

    dm.prepare_data()
    dm.setup("fit")

    tokenizer: Tokenizer = dm.tokenizer

    encoder = Encoder("roberta", masked_lm=True, vocab_size=len(tokenizer))
    encoder_path = "models/codecse-encoder"
    encoder.save(encoder_path)

    decoder_config = GPT2Config(
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoder = GPT2Model(decoder_config)
    decoder_path = "models/codecse-decoder"
    decoder.save_pretrained(decoder_path)


    seq2seq = Seq2Seq(
        encoder_pretrained_model_name_or_path=encoder_path,
        decoder_pretrained_model_name_or_path=decoder_path,
        decoder_start_token_id=tokenizer.cls_token_id,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
    )

    trainer = pl.Trainer(
        max_epochs=1,
        gpus=AVAIL_GPUS,
        log_every_n_steps=2,
        precision=16,
        stochastic_weight_avg=True,
        logger=TensorBoardLogger("logs/main/seq2seq/"),
    )

    trainer.fit(seq2seq, dm)
