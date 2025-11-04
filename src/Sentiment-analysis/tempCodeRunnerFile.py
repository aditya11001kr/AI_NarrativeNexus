model_lstm.save("lstm_model.h5")
joblib.dump(tokenizer, "lstm_tokenizer.pkl")
# Download to local machine
from google.colab import files
files.download("lstm_model.h5")
files.download("lstm_tokenizer.pkl")