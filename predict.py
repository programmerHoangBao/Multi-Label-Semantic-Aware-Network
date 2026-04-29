import torch
from transformers import AutoTokenizer
from models import BBLAMultiLabelModel
from config import Config
import numpy as np

import os

def load_label_mappings_txt(model_save_path: str):
    save_dir = os.path.dirname(model_save_path)

    tags_file = os.path.join(save_dir, "TAGS.txt")
    tag_to_idx_file = os.path.join(save_dir, "TAG_TO_IDX.txt")
    idx_to_tag_file = os.path.join(save_dir, "IDX_TO_TAG.txt")
    
    TAGS = []

    with open(tags_file, "r", encoding="utf-8") as f:
        for line in f:
            tag = line.strip()
            if tag:
                TAGS.append(tag)

    TAG_TO_IDX = {}

    with open(tag_to_idx_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tag, idx = line.split("\t")
                TAG_TO_IDX[tag] = int(idx)
    IDX_TO_TAG = {}

    with open(idx_to_tag_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                idx, tag = line.split("\t")
                IDX_TO_TAG[int(idx)] = tag
                
    return TAGS, TAG_TO_IDX, IDX_TO_TAG

class Predictor:
    """Class for making predictions on new questions"""
    
    def __init__(
        self, 
        model_path: str,
        bert_model_path: str = "microsoft/codebert-base",
        device: str = "cuda",
        threshold: float = 0.5,
        lstm_hidden_size: int = 512,
        num_attention_heads: int = 4,
        dropout: float = 0.2
    ):
        self.device = device
        self.threshold = threshold
        
        # Load label mappings
        self.TAGS, self.TAG_TO_IDX, self.IDX_TO_TAG = load_label_mappings_txt(model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        
        # Load model
        self.model = BBLAMultiLabelModel(
            model_path=bert_model_path,
            lstm_hidden=lstm_hidden_size,
            num_tags=len(self.TAGS),
            num_attention_heads=num_attention_heads,
            dropout=dropout
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def predict(self, 
                question: str,
                return_probabilities: bool = False):
        """
        Predict tags for a question
        
        Args:
            question: Input question text
            return_probabilities: If True, return probabilities; else binary predictions
        
        Returns:
            Dictionary with:
                - tags: List of predicted tags
                - probabilities: Dict of tag -> probability
                - prediction_array: Binary array
        """
        
        # Tokenize
        encoding = self.tokenizer(
            question,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            probabilities = self.model(input_ids, attention_mask)
            probabilities = probabilities.cpu().numpy()[0]  # [num_tags]
        
        # Convert to binary predictions
        predictions = (probabilities > self.threshold).astype(int)
        
        # Get predicted tags
        predicted_tags = [
            self.TAGS[i] for i in range(len(self.TAGS))
            if predictions[i] == 1
        ]
        
        # Create result dictionary
        result = {
            'question': question,
            'predicted_tags': predicted_tags,
            'prediction_probabilities': {
                self.TAGS[i]: float(probabilities[i])
                for i in range(len(self.TAGS))
            },
            'prediction_array': predictions.tolist()
        }
        
        return result
    
    def predict_batch(self, questions: list):
        """Predict tags for multiple questions"""
        results = []
        for question in questions:
            result = self.predict(question)
            results.append(result)
        return results


def demo_predict():
    """Demo prediction function"""
    
    print("="*80)
    print("Multi-Label Classification - Prediction Demo")
    print("="*80)
    
    config_obj = Config()
    # Initialize predictor
    predictor = Predictor(
        model_path=config_obj.SAVE_PATH,
        bert_model_path=config_obj.MODEL_PATH,
        device=config_obj.DEVICE,
        threshold=config_obj.PREDICTION_THRESHOLD,
        lstm_hidden_size=config_obj.LSTM_HIDDEN_SIZE,
        num_attention_heads=config_obj.NUM_ATTENTION_HEADS,
        dropout=config_obj.DROPOUT
    )
    
    # Test questions
    test_questions = [
        "How to create an ArrayList in Java?",
        "What is CSS in HTML?",
        "How to use jQuery for DOM manipulation?",
        "How to build an iOS app with Swift?",
    ]
    
    print("\nTest Predictions:\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"{i}. Question: {question}")
        result = predictor.predict(question)
        
        print(f"   Predicted Tags: {', '.join(result['predicted_tags'])}")
        print(f"   Probabilities:")
        
        # Sort by probability
        sorted_probs = sorted(
            result['prediction_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for tag, prob in sorted_probs[:5]:  # Top 5
            print(f"     - {tag}: {prob:.4f}")
        
        print()

if __name__ == "__main__":
    demo_predict()