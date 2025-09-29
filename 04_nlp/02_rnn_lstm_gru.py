"""
PyTorch NLP: RNN, LSTM, and GRU Models
======================================

This file covers recurrent neural networks for NLP:
- Vanilla RNN implementation and usage
- LSTM (Long Short-Term Memory) networks
- GRU (Gated Recurrent Unit) networks
- Bidirectional and stacked RNNs
- Sequence-to-sequence models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class VanillaRNN(nn.Module):
    """Vanilla RNN implementation from scratch"""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # RNN forward pass
        output, hidden = self.rnn(x, hidden)
        
        # Apply output layer to last time step
        output = self.fc(output[:, -1, :])  # Take last output
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

class LSTMClassifier(nn.Module):
    """LSTM-based text classifier"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 num_layers=1, dropout=0.2, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last output for classification
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]  # Last layer
        
        # Classification
        output = self.dropout(hidden)
        output = self.fc(output)
        
        return output

class GRULanguageModel(nn.Module):
    """GRU-based language model"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(GRULanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        # Embedding
        embedded = self.embedding(x)
        
        # GRU
        output, hidden = self.gru(embedded, hidden)
        
        # Reshape for output layer
        output = output.contiguous().view(-1, self.hidden_dim)
        output = self.dropout(output)
        output = self.output(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

class Seq2SeqEncoder(nn.Module):
    """Encoder for sequence-to-sequence model"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(Seq2SeqEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Seq2SeqDecoder(nn.Module):
    """Decoder for sequence-to-sequence model"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(Seq2SeqDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.output(output)
        return output, hidden, cell

class Seq2Seq(nn.Module):
    """Complete sequence-to-sequence model"""
    
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.vocab_size
        
        # Encode source sequence
        hidden, cell = self.encoder(src)
        
        # Prepare decoder inputs and outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        
        # First input to decoder is SOS token
        decoder_input = tgt[:, 0].unsqueeze(1)  # (batch_size, 1)
        
        for t in range(1, tgt_len):
            # Forward pass through decoder
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output.squeeze(1)
            
            # Teacher forcing: use target as next input with probability teacher_forcing_ratio
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs

def generate_synthetic_data():
    """Generate synthetic data for RNN examples"""
    print("=== Generating Synthetic Data ===")
    
    # Generate simple sequence classification data
    def create_sequence_data(n_samples=1000, seq_len=20, n_classes=2):
        """Create sequences where class depends on sequence properties"""
        X = []
        y = []
        
        for _ in range(n_samples):
            # Generate random sequence
            sequence = torch.randint(1, 10, (seq_len,))
            
            # Label based on whether sum is even or odd
            label = sequence.sum().item() % n_classes
            
            X.append(sequence)
            y.append(label)
        
        return torch.stack(X), torch.tensor(y)
    
    X, y = create_sequence_data(n_samples=1000, seq_len=15, n_classes=2)
    print(f"Generated data shapes: X={X.shape}, y={y.shape}")
    print(f"Sample sequence: {X[0]}")
    print(f"Sample label: {y[0]}")
    
    return X, y

def vanilla_rnn_example():
    """Demonstrate vanilla RNN usage"""
    print("\n=== Vanilla RNN Example ===")
    
    # Generate data
    X, y = generate_synthetic_data()
    vocab_size = 10
    
    # Convert to one-hot encoding for RNN input
    X_onehot = F.one_hot(X, num_classes=vocab_size).float()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_onehot, y, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    model = VanillaRNN(input_size=vocab_size, hidden_size=64, output_size=2, num_layers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/5, Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs, _ = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return model

def lstm_classification_example():
    """Demonstrate LSTM for text classification"""
    print("\n=== LSTM Classification Example ===")
    
    # Create text-like data (using integer sequences)
    def create_text_data(n_samples=1000, seq_len=30, vocab_size=100):
        X = []
        y = []
        
        for _ in range(n_samples):
            # Generate sequence
            sequence = torch.randint(1, vocab_size, (seq_len,))
            
            # Simple rule: positive if contains more values > 50
            positive_count = (sequence > 50).sum().item()
            label = 1 if positive_count > seq_len // 2 else 0
            
            X.append(sequence)
            y.append(label)
        
        return torch.stack(X), torch.tensor(y)
    
    X, y = create_text_data(n_samples=1000, seq_len=25, vocab_size=100)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
    
    # Create LSTM model
    model = LSTMClassifier(
        vocab_size=100,
        embedding_dim=64,
        hidden_dim=128,
        output_dim=2,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 2 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    print(f"LSTM Test Accuracy: {accuracy:.2f}%")
    
    return model

def gru_language_model_example():
    """Demonstrate GRU language model"""
    print("\n=== GRU Language Model Example ===")
    
    # Create simple language modeling data
    def create_language_data(sequence_length=20, vocab_size=50, n_sequences=500):
        """Create sequences for language modeling"""
        sequences = []
        
        for _ in range(n_sequences):
            # Generate sequence with some pattern
            seq = torch.randint(1, vocab_size, (sequence_length + 1,))
            sequences.append(seq)
        
        return torch.stack(sequences)
    
    # Generate data
    data = create_language_data(sequence_length=15, vocab_size=30, n_sequences=500)
    
    # Prepare input/target pairs
    X = data[:, :-1]  # Input sequences
    y = data[:, 1:]   # Target sequences (shifted by 1)
    
    # Create data loader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create GRU language model
    model = GRULanguageModel(
        vocab_size=30,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    model.train()
    for epoch in range(15):
        total_loss = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Initialize hidden state
            hidden = model.init_hidden(batch_X.size(0), batch_X.device)
            
            # Forward pass
            outputs, hidden = model(batch_X, hidden)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, model.vocab_size)
            batch_y = batch_y.view(-1)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 3 == 0:
            avg_loss = total_loss / len(dataloader)
            perplexity = torch.exp(torch.tensor(avg_loss))
            print(f"Epoch {epoch+1}/15, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    # Text generation example
    model.eval()
    with torch.no_grad():
        # Start with random token
        generated = torch.randint(1, 30, (1, 1))
        hidden = model.init_hidden(1, generated.device)
        
        generated_sequence = [generated.item()]
        
        for _ in range(10):  # Generate 10 tokens
            output, hidden = model(generated, hidden)
            predicted = output.argmax(dim=-1)
            generated = predicted[:, -1].unsqueeze(1)
            generated_sequence.append(generated.item())
    
    print(f"Generated sequence: {generated_sequence}")
    
    return model

def bidirectional_rnn_comparison():
    """Compare unidirectional vs bidirectional RNNs"""
    print("\n=== Bidirectional RNN Comparison ===")
    
    # Generate data
    X, y = generate_synthetic_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
    
    # Compare unidirectional vs bidirectional
    models = {
        'Unidirectional': LSTMClassifier(
            vocab_size=10, embedding_dim=32, hidden_dim=64, 
            output_dim=2, bidirectional=False
        ),
        'Bidirectional': LSTMClassifier(
            vocab_size=10, embedding_dim=32, hidden_dim=64, 
            output_dim=2, bidirectional=True
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} LSTM...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        model.train()
        for epoch in range(8):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.2f}%")
    
    return results

def sequence_to_sequence_example():
    """Simple sequence-to-sequence example"""
    print("\n=== Sequence-to-Sequence Example ===")
    
    # Create simple seq2seq data (reverse sequences)
    def create_seq2seq_data(n_samples=500, seq_len=10, vocab_size=20):
        src_sequences = []
        tgt_sequences = []
        
        for _ in range(n_samples):
            # Source sequence
            src = torch.randint(1, vocab_size-1, (seq_len,))
            
            # Target sequence (reversed + special tokens)
            # Add SOS (0) at beginning and EOS (vocab_size-1) at end
            tgt = torch.cat([
                torch.tensor([0]),  # SOS
                torch.flip(src, [0]),  # Reversed source
                torch.tensor([vocab_size-1])  # EOS
            ])
            
            src_sequences.append(src)
            tgt_sequences.append(tgt)
        
        return torch.stack(src_sequences), torch.stack(tgt_sequences)
    
    # Generate data
    src_data, tgt_data = create_seq2seq_data(n_samples=400, seq_len=8, vocab_size=15)
    
    # Create data loader
    dataset = TensorDataset(src_data, tgt_data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create encoder and decoder
    encoder = Seq2SeqEncoder(vocab_size=15, embedding_dim=32, hidden_dim=64)
    decoder = Seq2SeqDecoder(vocab_size=15, embedding_dim=32, hidden_dim=64)
    model = Seq2Seq(encoder, decoder)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    model.train()
    for epoch in range(20):
        total_loss = 0
        
        for batch_src, batch_tgt in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_src, batch_tgt, teacher_forcing_ratio=0.7)
            
            # Calculate loss (exclude SOS token from targets)
            outputs = outputs[:, 1:].contiguous().view(-1, outputs.size(-1))
            targets = batch_tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/20, Loss: {avg_loss:.4f}")
    
    # Test inference
    model.eval()
    with torch.no_grad():
        test_src = src_data[0:1]  # Take first sequence
        test_tgt = tgt_data[0:1]
        
        print(f"\nTest example:")
        print(f"Source: {test_src[0].tolist()}")
        print(f"Target: {test_tgt[0].tolist()}")
        
        # Simple greedy decoding
        encoded_hidden, encoded_cell = encoder(test_src)
        decoder_input = torch.tensor([[0]])  # SOS token
        predicted_sequence = [0]
        
        for _ in range(10):  # Max decode length
            output, encoded_hidden, encoded_cell = decoder(
                decoder_input, encoded_hidden, encoded_cell
            )
            predicted_token = output.argmax(dim=-1).item()
            predicted_sequence.append(predicted_token)
            
            if predicted_token == 14:  # EOS token
                break
            
            decoder_input = torch.tensor([[predicted_token]])
        
        print(f"Predicted: {predicted_sequence}")
    
    return model

def main():
    """Run all examples"""
    print("PyTorch RNN/LSTM/GRU Tutorial")
    print("=" * 40)
    
    generate_synthetic_data()
    vanilla_rnn_example()
    lstm_classification_example()
    gru_language_model_example()
    bidirectional_rnn_comparison()
    sequence_to_sequence_example()
    
    print("\n" + "=" * 40)
    print("Tutorial completed successfully!")

if __name__ == "__main__":
    main()