"""
Mini Project: Text Sentiment Classifier
=======================================

A complete end-to-end text classification project that combines:
- Text preprocessing and tokenization
- Custom dataset creation
- LSTM-based neural network
- Training loop with validation
- Model evaluation and inference

This project demonstrates practical application of NLP concepts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import random

class TextPreprocessor:
    """Handles text cleaning and preprocessing"""
    
    def __init__(self, max_vocab_size=10000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_counts = Counter()
        
        for text in texts:
            cleaned = self.clean_text(text)
            words = cleaned.split()
            word_counts.update(words)
        
        # Filter by frequency and limit size
        filtered_words = [(word, count) for word, count in word_counts.items() 
                         if count >= self.min_freq]
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        filtered_words = filtered_words[:self.max_vocab_size - 2]  # Reserve space for special tokens
        
        # Build vocabulary
        for word, _ in filtered_words:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
        
        print(f"Built vocabulary with {self.vocab_size} words")
        print(f"Most common words: {[word for word, _ in filtered_words[:10]]}")
    
    def text_to_indices(self, text, max_length=None):
        """Convert text to sequence of indices"""
        cleaned = self.clean_text(text)
        words = cleaned.split()
        
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # Truncate or pad to max_length
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices.extend([self.word2idx['<PAD>']] * (max_length - len(indices)))
        
        return indices

class SentimentDataset(Dataset):
    """Custom dataset for sentiment classification"""
    
    def __init__(self, texts, labels, preprocessor, max_length=128):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_length = max_length
        
        # Convert texts to indices
        self.encoded_texts = [
            preprocessor.text_to_indices(text, max_length)
            for text in texts
        ]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class SentimentLSTM(nn.Module):
    """LSTM model for sentiment classification"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, 
                 num_layers=2, num_classes=2, dropout=0.3):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state from both directions
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)
        # Take last layer, concatenate forward and backward
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # Classification
        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        return logits

def generate_synthetic_reviews():
    """Generate synthetic movie reviews for demonstration"""
    
    positive_templates = [
        "This movie was {adj}! The {aspect} was {quality}.",
        "I {verb} this film. {reason}.",
        "Amazing {aspect}! {reason}.",
        "What a {adj} {type}! Highly recommend.",
        "The {aspect} in this movie is {quality}. {reason}.",
    ]
    
    negative_templates = [
        "This movie was {adj}. The {aspect} was {quality}.",
        "I {verb} this film. {reason}.",
        "Terrible {aspect}. {reason}.",
        "What a {adj} {type}. Don't waste your time.",
        "The {aspect} in this movie is {quality}. {reason}.",
    ]
    
    positive_words = {
        'adj': ['fantastic', 'amazing', 'brilliant', 'outstanding', 'excellent', 'wonderful'],
        'aspect': ['acting', 'plot', 'cinematography', 'story', 'direction', 'soundtrack'],
        'quality': ['superb', 'magnificent', 'outstanding', 'brilliant', 'exceptional'],
        'verb': ['loved', 'enjoyed', 'adored', 'appreciated'],
        'reason': ['Great character development', 'Engaging storyline', 'Beautiful visuals', 
                  'Perfect pacing', 'Memorable scenes'],
        'type': ['masterpiece', 'film', 'movie', 'experience']
    }
    
    negative_words = {
        'adj': ['awful', 'terrible', 'boring', 'disappointing', 'horrible', 'dreadful'],
        'aspect': ['acting', 'plot', 'cinematography', 'story', 'direction', 'soundtrack'],
        'quality': ['poor', 'awful', 'disappointing', 'weak', 'unconvincing'],
        'verb': ['hated', 'disliked', 'regretted watching'],
        'reason': ['Poor character development', 'Confusing plot', 'Bad acting', 
                  'Slow pacing', 'Forgettable scenes'],
        'type': ['disaster', 'failure', 'disappointment', 'waste of time']
    }
    
    reviews = []
    labels = []
    
    # Generate positive reviews
    for _ in range(500):
        template = random.choice(positive_templates)
        review = template.format(**{key: random.choice(values) 
                                  for key, values in positive_words.items()})
        reviews.append(review)
        labels.append(1)  # Positive
    
    # Generate negative reviews
    for _ in range(500):
        template = random.choice(negative_templates)
        review = template.format(**{key: random.choice(values) 
                                  for key, values in negative_words.items()})
        reviews.append(review)
        labels.append(0)  # Negative
    
    return reviews, labels

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """Train the sentiment classification model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = batch['input_ids']
            labels = batch['labels']
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids']
                labels = batch['labels']
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
        print()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids']
            labels = batch['labels']
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = 100 * test_correct / test_total
    
    # Calculate precision, recall, F1 for each class
    tp_pos = sum((p == 1 and l == 1) for p, l in zip(all_predictions, all_labels))
    fp_pos = sum((p == 1 and l == 0) for p, l in zip(all_predictions, all_labels))
    fn_pos = sum((p == 0 and l == 1) for p, l in zip(all_predictions, all_labels))
    
    precision_pos = tp_pos / (tp_pos + fp_pos) if (tp_pos + fp_pos) > 0 else 0
    recall_pos = tp_pos / (tp_pos + fn_pos) if (tp_pos + fn_pos) > 0 else 0
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0
    
    return {
        'accuracy': test_accuracy,
        'precision': precision_pos,
        'recall': recall_pos,
        'f1': f1_pos,
        'predictions': all_predictions,
        'labels': all_labels
    }

def predict_sentiment(model, preprocessor, text):
    """Predict sentiment for a single text"""
    
    model.eval()
    
    # Preprocess text
    indices = preprocessor.text_to_indices(text, max_length=128)
    input_tensor = torch.tensor([indices], dtype=torch.long)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': {
            'negative': probabilities[0][0].item(),
            'positive': probabilities[0][1].item()
        }
    }

def plot_training_history(history):
    """Plot training and validation metrics"""
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_losses'], label='Train Loss', marker='o')
    ax1.plot(history['val_losses'], label='Validation Loss', marker='s')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_accuracies'], label='Train Accuracy', marker='o')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy', marker='s')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/training_history.png', dpi=150, bbox_inches='tight')
    print("Training history plot saved to /tmp/training_history.png")

def main():
    """Main function to run the complete project"""
    
    print("🎬 Text Sentiment Classifier Project")
    print("=" * 50)
    
    # 1. Generate synthetic data
    print("1. Generating synthetic movie reviews...")
    reviews, labels = generate_synthetic_reviews()
    print(f"Generated {len(reviews)} reviews")
    print(f"Sample positive review: {reviews[labels.index(1)]}")
    print(f"Sample negative review: {reviews[labels.index(0)]}")
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = TextPreprocessor(max_vocab_size=5000, min_freq=2)
    preprocessor.build_vocab(reviews)
    
    # 3. Create datasets
    print("\n3. Creating datasets...")
    dataset = SentimentDataset(reviews, labels, preprocessor, max_length=64)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # 4. Create model
    print("\n4. Creating model...")
    model = SentimentLSTM(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=64,
        hidden_dim=32,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Train model
    print("\n5. Training model...")
    history = train_model(model, train_loader, val_loader, num_epochs=8, lr=0.001)
    
    # 6. Evaluate model
    print("\n6. Evaluating model...")
    test_results = evaluate_model(model, test_loader)
    
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"Precision: {test_results['precision']:.3f}")
    print(f"Recall: {test_results['recall']:.3f}")
    print(f"F1 Score: {test_results['f1']:.3f}")
    
    # 7. Test on custom examples
    print("\n7. Testing on custom examples...")
    test_reviews = [
        "This movie was absolutely fantastic! Great acting and amazing story.",
        "Terrible film. Boring plot and poor acting. Complete waste of time.",
        "The cinematography was beautiful and the soundtrack was perfect.",
        "I fell asleep halfway through. Very disappointing and slow paced."
    ]
    
    for review in test_reviews:
        result = predict_sentiment(model, preprocessor, review)
        print(f"\nReview: {review}")
        print(f"Predicted: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print(f"Probabilities: Negative={result['probabilities']['negative']:.3f}, "
              f"Positive={result['probabilities']['positive']:.3f}")
    
    # 8. Plot training history
    print("\n8. Plotting training history...")
    plot_training_history(history)
    
    print("\n" + "=" * 50)
    print("Project completed successfully! 🎉")
    
    return model, preprocessor, test_results

if __name__ == "__main__":
    main()