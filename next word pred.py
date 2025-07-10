# full_next_word_prediction.py

import os
import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader

# Load Dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Add pad_token to GPT-2

# Tokenize
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data Collator for padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Load Model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # Resize in case pad token was added

# Training Args
training_args = TrainingArguments(
    output_dir="./gpt2_nextword",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=2,
    logging_dir="./logs"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])
print(f"\nFinal Perplexity: {perplexity:.2f}")

# Top-k Accuracy Calculation
def calculate_top_k_accuracy(model, tokenizer, eval_dataset, k_values=[1, 5, 10], max_samples=1000):
    """Calculate top-k accuracy for next word prediction"""
    model.eval()
    device = next(model.parameters()).device
    
    accuracies = {k: 0 for k in k_values}
    total_predictions = 0
    
    # Create a subset of evaluation data
    eval_subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    eval_dataloader = DataLoader(eval_subset, batch_size=8, collate_fn=data_collator)
    
    print(f"Calculating top-k accuracy on {len(eval_subset)} samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(eval_dataloader)}")
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get model predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits
            
            # For each sequence in the batch
            for seq_idx in range(input_ids.size(0)):
                seq_input_ids = input_ids[seq_idx]
                seq_attention_mask = attention_mask[seq_idx]
                seq_predictions = predictions[seq_idx]
                
                # Find valid positions (not padding and not the last token)
                valid_positions = (seq_attention_mask == 1).nonzero(as_tuple=True)[0]
                
                # Skip the last position since we can't predict next word
                if len(valid_positions) > 1:
                    valid_positions = valid_positions[:-1]
                
                for pos in valid_positions:
                    if pos + 1 < len(seq_input_ids):
                        # Get top-k predictions for this position
                        next_token_logits = seq_predictions[pos]
                        top_k_indices = torch.topk(next_token_logits, max(k_values)).indices
                        
                        # True next token
                        true_next_token = seq_input_ids[pos + 1].item()
                        
                        # Skip if true token is padding token
                        if true_next_token == tokenizer.pad_token_id:
                            continue
                        
                        # Check accuracy for different k values
                        for k in k_values:
                            if true_next_token in top_k_indices[:k]:
                                accuracies[k] += 1
                        
                        total_predictions += 1
    
    # Calculate final accuracies
    final_accuracies = {}
    for k in k_values:
        if total_predictions > 0:
            final_accuracies[k] = accuracies[k] / total_predictions
        else:
            final_accuracies[k] = 0.0
    
    return final_accuracies, total_predictions

# Calculate and display top-k accuracy
print("\n" + "="*50)
print("CALCULATING TOP-K ACCURACY")
print("="*50)

top_k_accuracies, total_preds = calculate_top_k_accuracy(
    model, tokenizer, tokenized["validation"], k_values=[1, 5, 10], max_samples=500
)

print(f"\nTop-k Accuracy Results (evaluated on {total_preds} predictions):")
for k, accuracy in top_k_accuracies.items():
    print(f"Top-{k} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Interactive Next Word Prediction
def predict_next_words(model, tokenizer, text, num_predictions=5, temperature=0.8):
    """Predict next words for a given text"""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits[0, -1, :] / temperature  # Apply temperature
        
        # Get top predictions
        top_predictions = torch.topk(predictions, num_predictions)
        
        predicted_words = []
        for i in range(num_predictions):
            token_id = top_predictions.indices[i].item()
            probability = torch.softmax(predictions, dim=-1)[token_id].item()
            word = tokenizer.decode([token_id])
            predicted_words.append((word, probability))
    
    return predicted_words

