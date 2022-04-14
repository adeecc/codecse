import pytorch_lightning as pl
from config import Config
from trainer import train_code_to_text, train_masked_lm


pl.seed_everything(42)


if __name__ == "__main__":
    config = Config(batch_size=2)

    train_masked_lm(config)
    train_code_to_text(config)
