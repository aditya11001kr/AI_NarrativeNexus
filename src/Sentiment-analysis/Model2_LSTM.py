train_path = "amazon_reviews_train.csv"
test_path  = "amazon_reviews_test.csv"


import re
import nltk
import pandas as pd
nltk.download("stopwords")
nltk.download("wordnet")
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Basic cleaning: lowercasing, removing special chars, stopwords, lemmatization"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)


train_df = pd.read_csv(train_path, nrows=50000)
test_df  = pd.read_csv(test_path, nrows=10000)


train_df["text"] = (train_df["title"].astype(str) + " " + train_df["content"].astype(str)).apply(clean_text)
test_df["text"]  = (test_df["title"].astype(str) + " " + test_df["content"].astype(str)).apply(clean_text)

X_train, y_train = train_df["text"], train_df["label"]
X_test,  y_test  = test_df["text"],  test_df["label"]

print("Train size:", len(X_train), " Test size:", len(X_test))



# MODEL 2: LSTM
import joblib

#Save in Colab
'''joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")'''

#Download to local machine
'''from google.colab import files
files.download("random_forest_model.pkl")
files.download("tfidf_vectorizer.pkl")'''


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Tokenization
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200)
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=200)

# Build model
model_lstm = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=200),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model_lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train (use smaller epochs first!)
history = model_lstm.fit(
    X_train_seq, np.array(y_train),
    epochs=3, batch_size=128,
    validation_data=(X_test_seq, np.array(y_test))
)

# Evaluate
y_pred_lstm = (model_lstm.predict(X_test_seq) > 0.5).astype("int32").flatten()
print(" LSTM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lstm))
print(classification_report(y_test, y_pred_lstm))

cm = confusion_matrix(y_test, y_pred_lstm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("LSTM Confusion Matrix")
plt.show()
model_lstm.save("lstm_model.h5")
joblib.dump(tokenizer, "lstm_tokenizer.pkl")

