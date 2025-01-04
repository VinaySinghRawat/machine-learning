# Import necessary libraries
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data with error handling
modi_path = r'D:\project\election prediction\tweetsModi.csv'
rahul_path = r'D:\project\election prediction\tweetsRG.csv'

if not os.path.exists(modi_path) or not os.path.exists(rahul_path):
    raise FileNotFoundError("One or both data files are missing. Please check the file paths.")

modi_data = pd.read_csv(modi_path)
rahul_data = pd.read_csv(rahul_path)

# Add 'candidate' column
modi_data['candidate'] = 'Modi'
rahul_data['candidate'] = 'Rahul'

# Combine datasets
data = pd.concat([modi_data, rahul_data], ignore_index=True)

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

data['cleaned_tweet'] = data['Tweet'].apply(preprocess_text)

# Sentiment labeling
positive_words = ['good', 'great', 'positive', 'win', 'success', 'happy']
negative_words = ['bad', 'worse', 'negative', 'fail', 'loss', 'sad']

def label_sentiment(text):
    positive_count = sum(word in text for word in positive_words)
    negative_count = sum(word in text for word in negative_words)
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

data['sentiment'] = data['cleaned_tweet'].apply(label_sentiment)

# Filter neutral tweets
data = data[data['sentiment'] != 'neutral']

# Visualize sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='sentiment', hue='candidate')
plt.title("Sentiment Distribution by Candidate")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.legend(title="Candidate")
plt.show()

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['cleaned_tweet'])
y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Binary classification
X[2]
y[2]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Predict winner
data['prediction'] = model.predict(X)
modi_positive = data[(data['candidate'] == 'Modi') & (data['prediction'] == 1)].shape[0]
rahul_positive = data[(data['candidate'] == 'Rahul') & (data['prediction'] == 1)].shape[0]

# Winner prediction
if modi_positive > rahul_positive:
    print(f"Prediction: Modi is likely to win based on sentiment analysis. (Positive Tweets: {modi_positive})")
else:
    print(f"Prediction: Rahul is likely to win based on sentiment analysis. (Positive Tweets: {rahul_positive})")

# Bar chart for positive sentiment
candidate_positive = pd.DataFrame({
    'Candidate': ['Modi', 'Rahul'],
    'Positive Tweets': [modi_positive, rahul_positive]
})

plt.figure(figsize=(8, 6))
sns.barplot(
    data=candidate_positive,
    x='Candidate',
    y='Positive Tweets',
    hue='Candidate',  # Assign x to hue
    dodge=False,      # Ensure single bars
    palette='viridis'
)
plt.title("Positive Sentiment Comparison")
plt.xlabel("Candidate")
plt.ylabel("Number of Positive Tweets")
plt.legend([], [], frameon=False)  # Remove redundant legend
plt.show()
