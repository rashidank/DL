from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing import sequence
from Perceptron import  Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle

top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


perceptron = Perceptron(epochs=10)

perceptron.fit(X_train, y_train)
pred = perceptron.predict(X_test)

print(f"Accuracy : {accuracy_score(pred, y_test)}")
report = classification_report(pred, y_test, digits=2)

print(report)
print(f"Predictions :{pred}")

with open(r"C:\Users\fayis\Documents\DL\diff_models\Perceptron\model_1_perceptron.pkl","wb") as file:
    pickle.dump(perceptron,file)