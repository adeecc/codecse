import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2Model

from codecse.dataset import (
    PLToNLDM,
    PLToNLDMForMaskedLM,
)

from codecse.model import Encoder, Seq2Seq

from codecse.constants import (
    AVAIL_GPUS,
)

from config import Config


def train_masked_lm(config: Config = Config()):
    # Create and Prepare the DataModule
    dm = PLToNLDMForMaskedLM(
        pl=config.pl,
        max_seq_length=config.max_seq_length,
        padding=config.max_seq_length,
        batch_size=config.batch_size,
    )

    dm.prepare_data()
    dm.setup("fit")

    # Create the encoder
    encoder = Encoder(
        "roberta",
        learning_rate=config.learning_rate,
        adam_epsilon=config.adam_epsilon,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        vocab_size=len(dm.tokenizer),
        masked_lm=True,
    )

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        gpus=AVAIL_GPUS,
        log_every_n_steps=config.log_every_n_steps,
        precision=16,
        stochastic_weight_avg=True,
        logger=TensorBoardLogger("logs/main/masked_lm/"),
    )

    # Train the encoder
    trainer.fit(encoder, dm)

    encoder.save(config.encoder_path)


def train_code_to_text(
    config: Config = Config(), use_untrained_encoder_decoder: bool = False
):
    dm = PLToNLDM(
        pl=config.pl,
        max_seq_length=config.max_seq_length,
        padding=config.padding,
        batch_size=config.batch_size,
    )

    dm.prepare_data()
    dm.setup("fit")

    tokenizer: Tokenizer = dm.tokenizer

    if use_untrained_encoder_decoder:
        encoder = Encoder(
            "roberta",
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            vocab_size=len(dm.tokenizer),
        )

        encoder.save(config.encoder_path)

        # TODO: Set number of decoder layers to 1?
        decoder_config = GPT2Config(
            n_layer=config.n_decoder_layers,
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        decoder = GPT2Model(decoder_config)
        decoder.save_pretrained(config.decoder_path)

    seq2seq = Seq2Seq(
        encoder_pretrained_model_name_or_path=config.encoder_path,
        decoder_pretrained_model_name_or_path=config.decoder_path,
        decoder_start_token_id=tokenizer.cls_token_id,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
    )

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        gpus=AVAIL_GPUS,
        log_every_n_steps=config.log_every_n_steps,
        precision=16,
        stochastic_weight_avg=True,
        logger=TensorBoardLogger("logs/main/seq2seq/"),
    )

    trainer.fit(seq2seq, dm)
    seq2seq.save(config.seq2seq_path)
