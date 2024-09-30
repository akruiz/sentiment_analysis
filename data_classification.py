import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Loading preprocessed review data...")
# Load the preprocessed review data
with open('Dataset/preprocessed_review_data.json') as f:
    data = json.load(f)

print("Extracting review texts and sentiment scores...")
# Extract the review texts and sentiment scores
reviews = [item['preprocessed_text'] for item in data]
sentiments = [item['stars'] for item in data]

print("Splitting data into training and testing sets...")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)

print("Converting review texts into numerical features using TF-IDF...")
# Convert the review texts into numerical features using TF-IDF
vectorizer = TfidfVectorizer()

# Join the tokens together into a single string for each review
X_train_joined = [" ".join(review) for review in X_train]
X_test_joined = [" ".join(review) for review in X_test]

X_train = vectorizer.fit_transform(X_train_joined)
X_test = vectorizer.transform(X_test_joined)

# Define the classifiers
classifiers = {
    "LinearSVC": LinearSVC(),
    "SGDClassifier": SGDClassifier(loss='hinge'),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB()
}

print("Training classifiers and printing their accuracies, precision, recall, and F1 scores...")
# Train the classifiers and print their accuracies, precision, recall, and F1 scores
for name, classifier in classifiers.items():
    print(f"Training {name}...")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Metrics for {name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

print("All tasks completed.")