import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from BackPropogation import  BackPropogation
from tensorflow.keras.preprocessing import sequence

# Load IMDB dataset
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# Optionally, you can pad the sequences if needed
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# Convert labels to binary (0 for negative, 1 for positive)
y_train_binary = (y_train == 0).astype(int)
y_test_binary = (y_test == 0).astype(int)

# Initialize and train the BackPropagation model
backpropagation = BackPropogation(learning_rate=0.01, epochs=5)
backpropagation.fit(X_train, y_train_binary)

# Predictions on the test set
pred = backpropagation.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test_binary, pred)}")
report = classification_report(y_test_binary, pred, digits=2)
print(report)
print(f"Predictions: {pred}")

import pickle
with open("imdb_back_prop.pkl",'wb') as file:
    pickle.dump(backpropagation,file)