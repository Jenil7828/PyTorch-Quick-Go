"""
PyTorch NLP: Text Processing and Tokenization
=============================================

This file covers text processing fundamentals for NLP:
- Text preprocessing and cleaning
- Tokenization strategies
- Vocabulary building
- Text encoding and decoding
- Handling different text formats
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import string
from collections import Counter, defaultdict
import pickle
import numpy as np

class TextPreprocessor:
    """Comprehensive text preprocessing class"""
    
    def __init__(self, lowercase=True, remove_punctuation=True, remove_numbers=False):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
    
    def clean_text(self, text):
        """Apply basic text cleaning"""
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        return text
    
    def tokenize(self, text):
        """Basic word tokenization"""
        return text.split()
    
    def preprocess(self, text):
        """Full preprocessing pipeline"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return tokens

class Vocabulary:
    """Vocabulary class for managing word-to-index mappings"""
    
    def __init__(self, pad_token='<PAD>', unk_token='<UNK>', sos_token='<SOS>', eos_token='<EOS>'):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        
        # Special tokens
        self.word2idx = {
            pad_token: 0,
            unk_token: 1,
            sos_token: 2,
            eos_token: 3
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.word_counts = Counter()
    
    def build_vocab(self, texts, min_freq=2, max_vocab_size=None):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in texts:
            if isinstance(text, str):
                tokens = text.split()
            else:
                tokens = text
            self.word_counts.update(tokens)
        
        # Filter by frequency and size
        vocab_items = [(word, count) for word, count in self.word_counts.items() 
                      if count >= min_freq]
        vocab_items.sort(key=lambda x: x[1], reverse=True)
        
        if max_vocab_size:
            vocab_items = vocab_items[:max_vocab_size - len(self.word2idx)]
        
        # Add words to vocabulary
        for word, _ in vocab_items:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Most common words: {vocab_items[:10]}")
    
    def encode(self, tokens):
        """Convert tokens to indices"""
        return [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]
    
    def decode(self, indices):
        """Convert indices to tokens"""
        return [self.idx2word.get(idx, self.unk_token) for idx in indices]
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, filepath):
        """Save vocabulary to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'special_tokens': {
                    'pad_token': self.pad_token,
                    'unk_token': self.unk_token,
                    'sos_token': self.sos_token,
                    'eos_token': self.eos_token
                }
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(**data['special_tokens'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_counts = data['word_counts']
        return vocab

class TextDataset(Dataset):
    """Dataset class for text data"""
    
    def __init__(self, texts, labels=None, vocab=None, max_length=None, 
                 add_special_tokens=True):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        # Preprocess texts
        preprocessor = TextPreprocessor()
        self.processed_texts = [preprocessor.preprocess(text) for text in texts]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.processed_texts[idx].copy()
        
        # Add special tokens
        if self.add_special_tokens and self.vocab:
            tokens = [self.vocab.sos_token] + tokens + [self.vocab.eos_token]
        
        # Encode tokens
        if self.vocab:
            token_ids = self.vocab.encode(tokens)
        else:
            token_ids = tokens
        
        # Truncate or pad
        if self.max_length:
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                pad_id = self.vocab.word2idx[self.vocab.pad_token] if self.vocab else 0
                token_ids.extend([pad_id] * (self.max_length - len(token_ids)))
        
        result = {'input_ids': torch.tensor(token_ids, dtype=torch.long)}
        
        if self.labels is not None:
            result['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return result

class CharacterTokenizer:
    """Character-level tokenizer"""
    
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
    
    def fit(self, texts):
        """Build character vocabulary"""
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Add special tokens
        chars.add('<PAD>')
        chars.add('<UNK>')
        
        # Create mappings
        chars = sorted(list(chars))
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(chars)
        
        print(f"Character vocabulary size: {self.vocab_size}")
    
    def encode(self, text):
        """Encode text to character indices"""
        return [self.char2idx.get(char, self.char2idx['<UNK>']) for char in text]
    
    def decode(self, indices):
        """Decode character indices to text"""
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices])

class SubwordTokenizer:
    """Simple subword tokenizer using Byte Pair Encoding (BPE)"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_tokenizer = TextPreprocessor()
        self.bpe_vocab = {}
        self.bpe_merges = []
    
    def get_word_tokens(self, texts):
        """Get word-level tokens from texts"""
        word_freqs = defaultdict(int)
        for text in texts:
            words = self.word_tokenizer.preprocess(text)
            for word in words:
                word_freqs[word] += 1
        return word_freqs
    
    def get_subword_stats(self, word_freqs):
        """Get subword pair statistics"""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair, word_freqs):
        """Merge most frequent pair in vocabulary"""
        new_word_freqs = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in word_freqs:
            new_word = p.sub(''.join(pair), word)
            new_word_freqs[new_word] = word_freqs[word]
        
        return new_word_freqs
    
    def learn_bpe(self, texts, num_merges=500):
        """Learn BPE merges from texts"""
        word_freqs = self.get_word_tokens(texts)
        
        # Initialize with character splits
        word_freqs = {' '.join(word): freq for word, freq in word_freqs.items()}
        
        for i in range(num_merges):
            pairs = self.get_subword_stats(word_freqs)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self.merge_vocab(best_pair, word_freqs)
            self.bpe_merges.append(best_pair)
        
        # Build final vocabulary
        self.bpe_vocab = set()
        for word in word_freqs:
            self.bpe_vocab.update(word.split())
        
        self.bpe_vocab = {token: idx for idx, token in enumerate(sorted(self.bpe_vocab))}
        print(f"BPE vocabulary size: {len(self.bpe_vocab)}")

def text_preprocessing_examples():
    """Demonstrate text preprocessing"""
    print("=== Text Preprocessing Examples ===")
    
    sample_texts = [
        "Hello, World! This is a SAMPLE text with 123 numbers.",
        "Another example with different punctuation: semicolons; and colons:",
        "Some texts have    extra    spaces    and\ttabs.",
        "What about URLs like https://example.com and emails like test@email.com?"
    ]
    
    # Test different preprocessing options
    preprocessors = [
        ("Basic", TextPreprocessor()),
        ("Keep punctuation", TextPreprocessor(remove_punctuation=False)),
        ("Remove numbers", TextPreprocessor(remove_numbers=True)),
        ("Case sensitive", TextPreprocessor(lowercase=False))
    ]
    
    for name, preprocessor in preprocessors:
        print(f"\n{name} preprocessing:")
        for text in sample_texts[:2]:  # Show only first 2 for brevity
            processed = preprocessor.preprocess(text)
            print(f"  Original: {text}")
            print(f"  Processed: {' '.join(processed)}")
    
    return preprocessors[0][1]

def vocabulary_building_example():
    """Demonstrate vocabulary building"""
    print("\n=== Vocabulary Building ===")
    
    # Sample corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "a quick brown dog jumps over a lazy fox",
        "the dog and fox are both quick and lazy",
        "brown animals can be quick or lazy",
        "the lazy fox jumps while the dog runs"
    ]
    
    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(corpus, min_freq=2, max_vocab_size=20)
    
    # Test encoding and decoding
    test_sentence = "the quick fox runs fast"
    tokens = test_sentence.split()
    encoded = vocab.encode(tokens)
    decoded = vocab.decode(encoded)
    
    print(f"\nOriginal: {test_sentence}")
    print(f"Tokens: {tokens}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {' '.join(decoded)}")
    
    # Show vocabulary statistics
    print(f"\nVocabulary statistics:")
    print(f"Total words: {len(vocab)}")
    print(f"Most common words: {list(vocab.word_counts.most_common(10))}")
    
    return vocab

def text_dataset_example():
    """Demonstrate text dataset usage"""
    print("\n=== Text Dataset Example ===")
    
    # Sample data
    texts = [
        "I love this movie it's amazing",
        "This film is terrible and boring",
        "Great acting and wonderful story",
        "Poor script and bad direction",
        "Excellent cinematography and music"
    ]
    labels = [1, 0, 1, 0, 1]  # 1 = positive, 0 = negative
    
    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(texts, min_freq=1)
    
    # Create dataset
    dataset = TextDataset(texts, labels, vocab, max_length=10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample batch:")
    
    for batch in dataloader:
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Input IDs: {batch['input_ids']}")
        print(f"  Labels: {batch['labels']}")
        break
    
    return dataset

def character_tokenization_example():
    """Demonstrate character-level tokenization"""
    print("\n=== Character Tokenization ===")
    
    texts = [
        "Hello world",
        "Character level tokenization",
        "This works at character level"
    ]
    
    char_tokenizer = CharacterTokenizer()
    char_tokenizer.fit(texts)
    
    # Test encoding/decoding
    test_text = "Hello"
    encoded = char_tokenizer.encode(test_text)
    decoded = char_tokenizer.decode(encoded)
    
    print(f"Original: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"Character mapping sample: {list(char_tokenizer.char2idx.items())[:10]}")
    
    return char_tokenizer

def padding_and_batching():
    """Demonstrate padding and batching strategies"""
    print("\n=== Padding and Batching ===")
    
    # Variable length sequences
    sequences = [
        [1, 2, 3],
        [4, 5],
        [6, 7, 8, 9, 10],
        [11],
        [12, 13, 14, 15]
    ]
    
    def pad_sequences(seqs, pad_value=0, max_len=None):
        """Pad sequences to same length"""
        if max_len is None:
            max_len = max(len(seq) for seq in seqs)
        
        padded = []
        for seq in seqs:
            if len(seq) < max_len:
                padded.append(seq + [pad_value] * (max_len - len(seq)))
            else:
                padded.append(seq[:max_len])
        
        return padded
    
    # Different padding strategies
    print("Original sequences:")
    for i, seq in enumerate(sequences):
        print(f"  Seq {i}: {seq} (length: {len(seq)})")
    
    # Pad to maximum length
    padded_max = pad_sequences(sequences)
    print(f"\nPadded to max length ({max(len(s) for s in sequences)}):")
    for i, seq in enumerate(padded_max):
        print(f"  Seq {i}: {seq}")
    
    # Pad to fixed length
    padded_fixed = pad_sequences(sequences, max_len=6)
    print(f"\nPadded to fixed length (6):")
    for i, seq in enumerate(padded_fixed):
        print(f"  Seq {i}: {seq}")
    
    # Convert to tensors
    tensor_batch = torch.tensor(padded_max)
    print(f"\nTensor batch shape: {tensor_batch.shape}")
    
    return tensor_batch

def advanced_text_processing():
    """Advanced text processing techniques"""
    print("\n=== Advanced Text Processing ===")
    
    # Text normalization
    def normalize_text(text):
        """Advanced text normalization"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    # Test normalization
    messy_text = """
    Check out this link: https://example.com and email me at test@email.com!
    There are    multiple    spaces here.
    """
    
    normalized = normalize_text(messy_text)
    print(f"Original: {repr(messy_text)}")
    print(f"Normalized: {repr(normalized)}")
    
    # N-gram generation
    def generate_ngrams(tokens, n):
        """Generate n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        return ngrams
    
    tokens = ["the", "quick", "brown", "fox", "jumps"]
    print(f"\nTokens: {tokens}")
    
    for n in range(1, 4):
        ngrams = generate_ngrams(tokens, n)
        print(f"{n}-grams: {ngrams}")
    
    return normalize_text, generate_ngrams

def main():
    """Run all examples"""
    print("PyTorch NLP Text Processing Tutorial")
    print("=" * 40)
    
    text_preprocessing_examples()
    vocabulary_building_example()
    text_dataset_example()
    character_tokenization_example()
    padding_and_batching()
    advanced_text_processing()
    
    print("\n" + "=" * 40)
    print("Tutorial completed successfully!")

if __name__ == "__main__":
    main()