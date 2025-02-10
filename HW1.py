import os
import sys
import time
import numpy as np
import nltk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from datasets import load_dataset

# Download necessary NLTK data
nltk.download("punkt")

# Load the batterydata/pos_tagging dataset
dataset = load_dataset('batterydata/pos_tagging')

# Use the entire training set and the entire test set
train_data = dataset['train']
test_data = dataset['test']

# Path to your GloVe file (update to your actual file path)
glove_file_path = '/Users/ali/Desktop/SP25/CMPSC 497/Test/HW1/glove.6B.100d.txt'
embedding_dim = 100

# Check if the GloVe file exists
if not os.path.exists(glove_file_path):
    raise FileNotFoundError(f"GloVe file not found at {glove_file_path}. Please set the correct path.")

# Load GloVe embeddings
def load_glove_embeddings(glove_file):
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
            # Optional progress print for large files every 100k lines:
            # if (i + 1) % 100000 == 0:
            #     print(f"  Loaded {i+1} lines from GloVe...")
    return embeddings

print("Loading GloVe embeddings (this can take a little while)...\n")
embeddings = load_glove_embeddings(glove_file_path)

# Single-word embedding lookup
def get_word_embedding(word, embeddings_dict, embedding_dim=100):
    return embeddings_dict.get(word.lower(), np.zeros(embedding_dim))

# Morphological feature extraction (basic example)
def get_morph_features(word):
    feats = []
    feats.append(int(word.isupper()))
    feats.append(int(word.istitle()))
    feats.append(int(word.isdigit()))
    
    # Simple 3-char prefix / suffix hashed to [0..1]
    prefix = word[:3].lower()
    suffix = word[-3:].lower()
    feats.append((hash(prefix) % 1000) / 1000.0)
    feats.append((hash(suffix) % 1000) / 1000.0)
    
    return np.array(feats, dtype='float32')

# Context embedding (with zero-padding for out-of-bounds)
def get_context_embedding(sentence_words, index, embeddings_dict, window_size=1, embedding_dim=100):
    vectors = []
    for offset in range(-window_size, window_size + 1):
        ctx_index = index + offset
        if 0 <= ctx_index < len(sentence_words):
            word = sentence_words[ctx_index]
            word_embed = get_word_embedding(word, embeddings_dict, embedding_dim)
            morph_embed = get_morph_features(word)
            # Concatenate word embedding + morphological features
            combined = np.concatenate([word_embed, morph_embed])
        else:
            # Zero-padding for out-of-bounds
            combined = np.zeros(embedding_dim + 5, dtype='float32')
        vectors.append(combined)
    return np.concatenate(vectors)

# Prepare data
def prepare_data(dataset_split, embeddings_dict, use_context=True, window_size=1, embedding_dim=100):
    X, y = [], []
    for item in dataset_split:
        words = item['words']
        labels = item['labels']
        for i, tag in enumerate(labels):
            if use_context:
                features = get_context_embedding(words, i, embeddings_dict, window_size, embedding_dim)
            else:
                # Single word + morphological
                word_embed = get_word_embedding(words[i], embeddings_dict, embedding_dim)
                morph_embed = get_morph_features(words[i])
                features = np.concatenate([word_embed, morph_embed])
            X.append(features)
            y.append(tag)
    return np.array(X), np.array(y)

# Increase context window to 2 on each side
use_context = True
window_size = 2

X_train, y_train = prepare_data(
    train_data,
    embeddings_dict=embeddings,
    use_context=use_context,
    window_size=window_size,
    embedding_dim=embedding_dim
)

X_test, y_test = prepare_data(
    test_data,
    embeddings_dict=embeddings,
    use_context=use_context,
    window_size=window_size,
    embedding_dim=embedding_dim
)

# Label encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Handle unseen labels in test data
unseen_label_index = len(label_encoder.classes_)
y_test_encoded_list = []
for label in y_test:
    if label in label_encoder.classes_:
        y_test_encoded_list.append(label_encoder.transform([label])[0])
    else:
        y_test_encoded_list.append(unseen_label_index)
y_test_encoded = np.array(y_test_encoded_list)

# FUN LOADING BAR BEFORE TRAINING -- Enjoy it :)
print("Ali AlNaseeb is warming up the GPU (shhhh, we won't mention it's just a CPU). Please wait:")

for i in range(0, 101, 10):
    bar = f"[{'=' * (i // 10)}{' ' * (10 - i // 10)}]"
    sys.stdout.write(f"\r  {bar} {i}% ")
    sys.stdout.flush()
    time.sleep(0.4)

print("\nLoading complete! Now actually starting the training...\n")

# Now actually train the Logistic Regression
clf = LogisticRegression(C=10.0, max_iter=1000)
clf.fit(X_train, y_train_encoded)
print("Model training complete.\n")

# Make predictions
print("Making predictions on the test data...\n")
y_pred = clf.predict(X_test)

# Filter out any unseen-label predictions
valid_indices = (y_test_encoded != unseen_label_index)
y_test_filtered = y_test_encoded[valid_indices]
y_pred_filtered = y_pred[valid_indices]

# Accuracy caculation --- it gave me Accuracy: 93.54% so hopfully it is near that number
accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
print(f"Accuracy: {accuracy * 100:.2f}%\n")