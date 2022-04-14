import pytorch_lightning as pl

from trainer import train_code_to_text, train_masked_lm

pl.seed_everything(42)


if __name__ == "__main__":
    # train_masked_lm()
    train_code_to_text()
