import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Tuple, List, Dict
import logging
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class MultiLabelDataset(Dataset):
    """Custom Dataset for Multi-Label Classification"""
    
    def __init__(self, 
                 file_path: str, 
                 tokenizer,
                 max_length: int = 512,
                 tags_list: List[str] = None):
        
        # Load data
        self.df = pd.read_parquet(file_path)
        self.df['text'] = self.df['title'] + ". " + self.df['question']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tags_list = tags_list
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.tags_list)}
        
        logger.info(f"Loaded {len(self.df)} samples from {file_path}")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single sample
        
        Returns:
            Dictionary with:
                - input_ids
                - attention_mask
                - labels (multi-hot encoded)
        """
        
        row = self.df.iloc[idx]
        question = row['text']
        tags = row['tags']
        
        # Convert tags to list if it's numpy array
        if isinstance(tags, np.ndarray):
            tags = tags.tolist()
        
        # Tokenize question
        encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create multi-hot encoded labels
        labels = np.zeros(len(self.tags_list), dtype=np.float32)
        for tag in tags:
            if tag in self.tag_to_idx:
                labels[self.tag_to_idx[tag]] = 1.0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }
        
def create_data_loaders(
    model_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    tags_list: List,
    max_length: int = 512,
    train_batch_size: int = 32,
    val_batch_size: int = 64,
    test_batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create datasets
    train_dataset = MultiLabelDataset(
        train_path,
        tokenizer,
        max_length=max_length,
        tags_list=tags_list
    )
    
    val_dataset = MultiLabelDataset(
        val_path,
        tokenizer,
        max_length=max_length,
        tags_list=tags_list
    )
    
    test_dataset = MultiLabelDataset(
        test_path,
        tokenizer,
        max_length=max_length,
        tags_list=tags_list
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader