import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class Trainer:
    """Trainer class for multi-label classification model"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda',
                 save_path: str = './models/best_model.pt'):
        self.model = model.to(device)
        self.device = device
        self.save_path = save_path
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Loss function for multi-label: BCELoss (Binary Cross Entropy)
        self.criterion = nn.BCELoss()
        
        self.best_f1 = 0
        self.patience_counter = 0
        
    def train_epoch(self,
                   train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                ) -> float:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            predictions = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self,
                val_loader: DataLoader,
                threshold: float = 0.5,
        ) -> Dict[str, float]:
        """
        Evaluate model on validation/test set
        
        Returns:
            Dictionary with metrics: accuracy, precision, recall, f1
        """
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Evaluating")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                predictions = self.model(input_ids, attention_mask)
                
                # Loss
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                # Convert to binary predictions
                binary_preds = (predictions > threshold).cpu().numpy()
                
                all_predictions.extend(binary_preds)
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_labels)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, 
                         labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics for multi-label classification
        
        Args:
            predictions: [num_samples, num_tags] binary predictions
            labels: [num_samples, num_tags] ground truth labels
        """
        
        # Exact Match Ratio (Hamming Loss)
        exact_match = np.mean(np.all(predictions == labels, axis=1))
        
        # Subset Accuracy
        subset_accuracy = np.mean(
            (predictions == labels).sum(axis=1) == predictions.shape[1]
        )
        
        # Per-label metrics
        tp = np.sum(predictions * labels, axis=0)
        fp = np.sum(predictions * (1 - labels), axis=0)
        fn = np.sum((1 - predictions) * labels, axis=0)
        
        # Precision, Recall, F1 per label
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Macro-averaged metrics
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Micro-averaged metrics
        micro_tp = np.sum(tp)
        micro_fp = np.sum(fp)
        micro_fn = np.sum(fn)
        
        micro_precision = micro_tp / (micro_tp + micro_fp + 1e-8)
        micro_recall = micro_tp / (micro_tp + micro_fn + 1e-8)
        micro_f1 = 2 * (micro_precision * micro_recall) / (
            micro_precision + micro_recall + 1e-8
        )
        
        return {
            'exact_match': exact_match,
            'subset_accuracy': subset_accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
        }
    
    def save_model(self):
        """Save model checkpoint"""
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")
    
    def load_model(self):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(self.save_path))
        print(f"Model loaded from {self.save_path}")
        
        
def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          num_epochs: int = 10,
          device: str = "cuda",
          save_path: str = "./models/best_model.pt",
          learning_rate: float = 1e-4,
          weight_decay: float = 1e-5,
          prediction_threshold: float=0.5
    ) -> Trainer:
    
    trainer = Trainer(model, device=device, save_path=save_path)
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    logger.info("\n" + "="*80)
    logger.info("Starting Training")
    logger.info("="*80)
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"{'='*80}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = trainer.evaluate(val_loader, 
                                       threshold=prediction_threshold,
                                       )
        
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
        logger.info(f"Val Micro F1: {val_metrics['micro_f1']:.4f}")
        logger.info(f"Val Exact Match: {val_metrics['exact_match']:.4f}")
        
        # Save best model
        if val_metrics['macro_f1'] > trainer.best_f1:
            trainer.best_f1 = val_metrics['macro_f1']
            trainer.save_model()
            trainer.patience_counter = 0
            logger.info("Best model saved!")
        else:
            trainer.patience_counter += 1
            logger.info(f"Patience: {trainer.patience_counter}")
    
    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info("="*80)
    
    return trainer