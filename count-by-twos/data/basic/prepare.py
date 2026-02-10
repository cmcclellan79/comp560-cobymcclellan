"""
Generates training data for counting by twos experiment.
Example output:
    43 45 47 49
    12 14 16 18
    96 98 100 102
"""
import os
import pickle
import numpy as np
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generate training data
def generate_counting_data(num_sequences=10000):
    """Generate sequences of counting by twos"""
    data = []
    
    for _ in range(num_sequences):
        # Pick a random starting number under 100
        start = random.randint(1, 99)
        
        # Generate 4 numbers counting by 2s
        sequence = [str(start + i*2) for i in range(4)]
        line = ' '.join(sequence) + '\n'
        data.append(line)
    
    return ''.join(data)

# Generate data
print("Generating counting by twos data...")
train_data = generate_counting_data(num_sequences=8000)
val_data = generate_counting_data(num_sequences=1000)

# Get all unique characters
all_text = train_data + val_data
chars = sorted(list(set(all_text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {chars}")

# Create character to index mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode function
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Encode the data
train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# Export to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete!")
print(f"Files created: train.bin, val.bin, meta.pkl")

# Show sample of training data
print("\nSample training data:")
print(train_data[:200])
