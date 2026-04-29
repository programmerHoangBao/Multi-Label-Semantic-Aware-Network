import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def build_label_mappings(parquet_path: str, label_name: str):
    df = pd.read_parquet(parquet_path)
    if label_name not in df.columns:
        raise ValueError(f"Column {label_name} not found in dataset")
    all_labels = set()
    for array in df[label_name]:
        if isinstance(array, (list, np.ndarray)):
            all_labels.update(array)
    TAGS = sorted(list(all_labels))
    TAG_TO_IDX = {tag: idx for idx, tag in enumerate(TAGS)}
    IDX_TO_TAG = {idx: tag for idx, tag in enumerate(TAGS)}

    logger.info(f"TAGS: {TAGS}")
    logger.info(f"TAG_TO_IDX: {TAG_TO_IDX}")
    logger.info(f"IDX_TO_TAG: {IDX_TO_TAG}")

    return TAGS, TAG_TO_IDX, IDX_TO_TAG

def save_label_mappings_txt(
    TAGS,
    TAG_TO_IDX,
    IDX_TO_TAG,
    model_save_path: str
):
    save_dir = os.path.dirname(model_save_path)
    os.makedirs(save_dir, exist_ok=True)
    tags_file = os.path.join(save_dir, "TAGS.txt")
    with open(tags_file, "w", encoding="utf-8") as f:
        for tag in TAGS:
            f.write(f"{tag}\n")
            
    tag_to_idx_file = os.path.join(save_dir, "TAG_TO_IDX.txt")
    with open(tag_to_idx_file, "w", encoding="utf-8") as f:
        for tag, idx in TAG_TO_IDX.items():
            f.write(f"{tag}\t{idx}\n")
            
    idx_to_tag_file = os.path.join(save_dir, "IDX_TO_TAG.txt")
    with open(idx_to_tag_file, "w", encoding="utf-8") as f:
        for idx, tag in IDX_TO_TAG.items():
            f.write(f"{idx}\t{tag}\n")

    logger.info("Saved label mappings:")
    logger.info(f"  TAGS -> {tags_file}")
    logger.info(f"  TAG_TO_IDX -> {tag_to_idx_file}")
    logger.info(f"  IDX_TO_TAG -> {idx_to_tag_file}")

class Config:
    """Configuration for Multi-Label Classification Model"""
    
    # Data
    TRAIN_PATH = "./data/train.parquet"
    VAL_PATH = "./data/val.parquet"
    TEST_PATH = "./data/test.parquet"
    
    TAGS, TAG_TO_IDX, IDX_TO_TAG = build_label_mappings(parquet_path=TRAIN_PATH, label_name='tags')
    NUM_TAGS = len(TAGS)
    
    # Model
    MODEL_PATH = "./codebert-base"
    CODEBERT_HIDDEN_SIZE = 768
    LSTM_HIDDEN_SIZE = 512
    NUM_ATTENTION_HEADS = 4
    DROPOUT = 0.2
    
    # Training
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Device
    DEVICE = "cuda"  # or "cpu"
    
    # Prediction threshold
    PREDICTION_THRESHOLD = 0.5
    
    # Others
    MAX_LENGTH = 512
    SEED = 42
    SAVE_PATH = "./models/bbla_model.pt"
    save_label_mappings_txt(TAGS, TAG_TO_IDX, IDX_TO_TAG, SAVE_PATH)