import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_review_length=500
# Assume 'new_review' is the new review provided by a customer
new_review = "The movie was really good and I enjoyed it a lot."

# Load the IMDb dataset to get word-to-index mapping
top_words = 5000
(X_train, _), (_, _) = tf.keras.datasets.imdb.load_data(num_words=top_words)

# Create a reverse mapping from index to word
word_index = tf.keras.datasets.imdb.get_word_index()
reverse_word_index = {index: word for word, index in word_index.items()}

# Tokenize and encode the new review
new_review_sequence = tf.keras.preprocessing.text.text_to_word_sequence(new_review)
new_review_encoded = [word_index.get(word, 0) for word in new_review_sequence]
new_review_padded = pad_sequences([new_review_encoded], maxlen=max_review_length)

# Load the trained model
loaded_model = tf.keras.models.load_model("imdb_DNN_model.h5")

# Make predictions
predictions = loaded_model.predict(new_review_padded)

# Interpret the prediction
if predictions[0] > 0.5:
    sentiment = "Positive"
else:
    sentiment = "Negative"

print(f"Predicted Sentiment: {sentiment} (Probability: {predictions[0][0]:.4f})")
