"""
prepare_dataset.py

Downloads the benchmark IMDb Movie Reviews dataset and word index once,
and saves them into the local 'dataset/' folder as .npy and .json files.
After running this script, Sentiment_Analysis.ipynb can run 100% offline.
"""

import os
import json
import numpy as np
from keras.datasets import imdb

def prepare_offline_dataset():
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    vocab_size = 10000
    print(f"Downloading IMDb dataset with top {vocab_size} vocabulary words...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    
    # Paths for saving
    x_train_path = os.path.join(dataset_dir, "x_train.npy")
    y_train_path = os.path.join(dataset_dir, "y_train.npy")
    x_test_path = os.path.join(dataset_dir, "x_test.npy")
    y_test_path = os.path.join(dataset_dir, "y_test.npy")
    word_index_path = os.path.join(dataset_dir, "word_index.json")
    
    print("Saving train and test sets to 'dataset/'...")
    np.save(x_train_path, x_train, allow_pickle=True)
    np.save(y_train_path, y_train)
    np.save(x_test_path, x_test, allow_pickle=True)
    np.save(y_test_path, y_test)
    
    print("Downloading and saving IMDb word index...")
    word_index = imdb.get_word_index()
    with open(word_index_path, "w", encoding="utf-8") as f:
        json.dump(word_index, f, ensure_ascii=False)
        
    print("\nDataset preparation complete! All files saved locally in 'dataset/':")
    for fname in ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy", "word_index.json"]:
        fpath = os.path.join(dataset_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  - {fname} ({size_kb:.1f} KB)")
        
if __name__ == "__main__":
    prepare_offline_dataset()
