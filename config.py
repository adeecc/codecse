from dataclasses import dataclass

@dataclass
class Config:
    encoder_mode: str = "roberta"

    encoder_path: str = "models/codecse-encoder"
    decoder_path: str = "models/codecse-decoder"
    seq2seq_path: str = "models/codecse-seq2seq"

    batch_size: int = 128
    pl: str = "Java"
    max_seq_length: int = 128
    padding: str = "max_length"

    n_decoder_layers: int = 2

    learning_rate: float = 2e-5
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    weight_decay: float = 0.0

    num_epochs: int = 5
    log_every_n_steps: int = 2
