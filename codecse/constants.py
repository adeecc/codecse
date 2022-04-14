import os
import torch

AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = max(int(os.cpu_count() / 2), os.cpu_count() - 2)
DATASET_NAME = "code_x_glue_ct_code_to_text"
TOKENIZER_MODEL = "Salesforce/codet5-base"