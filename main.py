import torch
import numpy as np
from models import BBLAMultiLabelModel
from data_loader import create_data_loaders
from trainer import Trainer, train
from config import Config
import pandas as pd
import os

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_test_metrics_to_csv(test_metrics: dict, file_path: str):
    df = pd.DataFrame([test_metrics])
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

    print(f"Saved test metrics to: {file_path}")

def main():
    # Set seed
    config_obj = Config()
    set_seed(config_obj.SEED)
    
    # Check device
    print(f"Using device: {config_obj.DEVICE}")
    if config_obj.DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "="*80)
    print("Multi-Label Classification with Attention-based Multi-Head")
    print("="*80)
    
    # Create data loaders
    print("\Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        model_path=config_obj.MODEL_PATH,
        train_path=config_obj.TRAIN_PATH,
        val_path=config_obj.VAL_PATH,
        test_path=config_obj.TEST_PATH,
        tags_list=config_obj.TAGS,
        max_length=config_obj.MAX_LENGTH,
        train_batch_size=config_obj.TRAIN_BATCH_SIZE,
        val_batch_size=config_obj.VAL_BATCH_SIZE,
        test_batch_size=config_obj.TEST_BATCH_SIZE
    )
    
    # Create model
    print("\nCreating model...")
    model = BBLAMultiLabelModel(
        model_path=config_obj.MODEL_PATH,
        lstm_hidden=config_obj.LSTM_HIDDEN_SIZE,
        num_tags=config_obj.NUM_TAGS,
        num_attention_heads=config_obj.NUM_ATTENTION_HEADS,
        dropout=config_obj.DROPOUT
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    print("\nTraining model...")
    trainer = train(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        num_epochs=config_obj.NUM_EPOCHS,
        device=config_obj.DEVICE,
        save_path=config_obj.SAVE_PATH,
        learning_rate=config_obj.LEARNING_RATE,
        weight_decay=config_obj.WEIGHT_DECAY,
        prediction_threshold=config_obj.PREDICTION_THRESHOLD
    )
    
    # Load best model
    print("\nLoading best model...")
    trainer.load_model()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader, 
                                    threshold=config_obj.PREDICTION_THRESHOLD,
                                )
    
    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)
    print(f"Test Loss:           {test_metrics['loss']:.4f}")
    print(f"Test Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"Test Macro Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"Test Macro F1:        {test_metrics['macro_f1']:.4f}")
    print(f"Test Micro Precision: {test_metrics['micro_precision']:.4f}")
    print(f"Test Micro Recall:    {test_metrics['micro_recall']:.4f}")
    print(f"Test Micro F1:        {test_metrics['micro_f1']:.4f}")
    print(f"Test Exact Match:     {test_metrics['exact_match']:.4f}")
    print("="*80)
    
    save_test_metrics_to_csv(
        test_metrics,
        file_path="test_results.csv"
    )
    
if __name__ == "__main__":
    main()