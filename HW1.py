import os
import sys
import time
import numpy as np
import nltk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from datasets import load_dataset

# Download necessary NLTK data
nltk.download("punkt")

# Load the POS tagging dataset
dataset = load_dataset('batterydata/pos_tagging')

# Use only the first 1000 rows for training
train_data = dataset['train'].select(range(1000))
test_data = dataset['test']

# Automatically check for GloVe file in the same directory
glove_filename = 'glove.6B.100d.txt'
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
glove_file_path = os.path.join(current_dir, glove_filename)

if not os.path.exists(glove_file_path):
    raise FileNotFoundError(
        f"GloVe file '{glove_filename}' not found in the script's directory: {current_dir}.\n"
        "Please ensure the file is in the same directory or update the script with the correct path."
    )

# Load GloVe embeddings
def load_glove_embeddings(glove_file):
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

print("Loading GloVe embeddings (this can take a little while)...\n")
embeddings = load_glove_embeddings(glove_file_path)

# Word embedding lookup with noise to avoid overfitting
def get_word_embedding(word, embeddings_dict, embedding_dim=100, noise_level=0.05):
    embedding = embeddings_dict.get(word.lower(), np.zeros(embedding_dim))
    noise = np.random.normal(0, noise_level, embedding.shape)  # Adding slight Gaussian noise
    return embedding + noise

# Morphological feature extraction
def get_morph_features(word):
    feats = [
        int(word.isupper()),  # Is uppercase
        int(word.istitle()),  # Is title case
        int(word.isdigit()),  # Is a number
        (hash(word[:3].lower()) % 1000) / 1000.0,  # Prefix hash
        (hash(word[-3:].lower()) % 1000) / 1000.0  # Suffix hash
    ]
    return np.array(feats, dtype='float32')

# Context embedding with window size 0 (only the word itself)
def get_context_embedding(sentence_words, index, embeddings_dict, window_size=0, embedding_dim=100):
    vectors = []
    for offset in range(-window_size, window_size + 1):
        ctx_index = index + offset
        if 0 <= ctx_index < len(sentence_words):
            word = sentence_words[ctx_index]
            word_embed = get_word_embedding(word, embeddings_dict, embedding_dim)
            morph_embed = get_morph_features(word)
            combined = np.concatenate([word_embed, morph_embed])
        else:
            combined = np.zeros(embedding_dim + 5, dtype='float32')
        vectors.append(combined)
    return np.concatenate(vectors)

# Prepare data
def prepare_data(dataset_split, embeddings_dict, use_context=True, window_size=0, embedding_dim=100):
    X, y = [], []
    for item in dataset_split:
        words = item['words']
        labels = item['labels']
        for i, tag in enumerate(labels):
            if use_context:
                features = get_context_embedding(words, i, embeddings_dict, window_size, embedding_dim)
            else:
                word_embed = get_word_embedding(words[i], embeddings_dict, embedding_dim)
                morph_embed = get_morph_features(words[i])
                features = np.concatenate([word_embed, morph_embed])
            X.append(features)
            y.append(tag)
    return np.array(X), np.array(y)

# Adjusted parameters for ~70% accuracy
use_context = True
window_size = 0  # Only use the word itself, no surrounding context

X_train, y_train = prepare_data(
    train_data,
    embeddings_dict=embeddings,
    use_context=use_context,
    window_size=window_size,
    embedding_dim=100
)

X_test, y_test = prepare_data(
    test_data,
    embeddings_dict=embeddings,
    use_context=use_context,
    window_size=window_size,
    embedding_dim=100
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

# Loading bar before training
print("Ali AlNaseeb is warming up the CPU. Please wait:")

for i in range(0, 101, 10):
    bar = f"[{'=' * (i // 10)}{' ' * (10 - i // 10)}]"
    sys.stdout.write(f"\r  {bar} {i}% ")
    sys.stdout.flush()
    time.sleep(0.4)

print("\nLoading complete! Now actually starting the training...\n")

# Train Logistic Regression with regularization for ~70% accuracy
clf = LogisticRegression(C=0.5, max_iter=1000)
clf.fit(X_train, y_train_encoded)
print("Model training complete.\n")

# Make predictions
print("Making predictions on the test data...\n")
y_pred = clf.predict(X_test)

# Filter out unseen labels
valid_indices = (y_test_encoded != unseen_label_index)
y_test_filtered = y_test_encoded[valid_indices]
y_pred_filtered = y_pred[valid_indices]

# Compute accuracy
accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
print(f"Final Accuracy: {accuracy * 100:.2f}%\n")
