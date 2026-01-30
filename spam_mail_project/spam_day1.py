import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv("SMSSpamCollection", sep="\t", header=None)
data.columns = ["label", "message"]

# Stopwords
stop_words = set(stopwords.words("english"))

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Apply cleaning
data["clean_message"] = data["message"].apply(clean_text)

# -------- DAY 3 PART --------
# Convert text to numeric features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_message"])

# Labels
y = data["label"]

# Show feature shape
print("Feature matrix shape:", X.shape)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Encode labels (spam=1, ham=0)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

print("Model training completed")
from sklearn.metrics import accuracy_score, classification_report

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
# -------- DAY 6 PART --------
# Function to predict spam or ham for new messages
def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_vec = vectorizer.transform([msg_clean])
    prediction = model.predict(msg_vec)
    return "Spam" if prediction[0] == 1 else "Ham"

# Test with sample messages
test_messages = [
   "Congratulations! You won a free lottery ticket",
    "Hi bro, are we meeting tomorrow?",
    "URGENT! Call this number to claim your prize",
    "Can you send me the notes?"
]

print("\nReal-Time Prediction Results:")
for msg in test_messages:
    print(f"Message: {msg}")
    print("Prediction:", predict_message(msg))
    print("-" * 40)
# DAY 6: Test with new messages

def predict_message(message):
    message = clean_text(message)
    message_vector = vectorizer.transform([message])
    result = model.predict(message_vector)
    if result[0] == 1:
        return "Spam"
    else:
        return "Ham"

# Test messages
print("\nTesting new messages:\n")

print("Message: Win free cash now")
print("Prediction:", predict_message("Win free cash now"))

print("\nMessage: Are you coming to college today?")
print("Prediction:", predict_message("Are you coming to college today?"))
