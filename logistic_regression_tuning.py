import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading preprocessed review data...")
# Load the preprocessed review data
with open('Dataset/preprocessed_review_data.json') as f:
    data = json.load(f)

print("Extracting review IDs, texts, and sentiment scores...")
# Extract the review IDs, texts, and sentiment scores
review_ids = [item['review_id'] for item in data]
reviews = [item['preprocessed_text'] for item in data]
sentiments = [item['stars'] for item in data]

print("Splitting data into training and testing sets...")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(reviews, sentiments, review_ids, test_size=0.3, random_state=42)

print("Converting review texts into numerical features using TF-IDF...")
# Convert the review texts into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_joined = [" ".join(review) for review in X_train]
X_test_joined = [" ".join(review) for review in X_test]
X_train = vectorizer.fit_transform(X_train_joined)
X_test = vectorizer.transform(X_test_joined)

print("\nExample of review text conversion to numerical features using TF-IDF:")
for i in range(3):
    print(f"Review Text: {X_train_joined[i]}")
    print(f"Numerical Features: {X_train[i].toarray()}\n")

# Define the Logistic Regression model and parameters for GridSearchCV
model = LogisticRegression(max_iter=1000)
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear']
}

print("Performing hyperparameter tuning for Logistic Regression...")
# Initialize and perform the grid search
grid_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print("Evaluating the best model on the test data...")
# Evaluate the best model on the test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Best parameters found by the hyperparameter tuning:")
print(grid_search.best_params_)

print("\nExample predictions of review sentiments:")
for i in range(3):  # Print first 3 examples
    print(f"Review ID: {ids_test[i]}")
    print(f"Review Text: {X_test_joined[i]}")
    print(f"Actual Stars: {y_test[i]}, Predicted Stars: {y_pred[i]}\n")

# Combine review IDs, actual sentiments, and predicted sentiments
print("Combining review IDs, actual sentiments, and predicted sentiments...")
results = []
for id, actual, pred in zip(ids_test, y_test, y_pred):
    result = {
        "review_id": id,
        "actual_stars": actual,
        "predicted_stars": pred
    }
    results.append(result)

print("Saving the results to a JSON file...")
# Save the results to a JSON file
with open('predicted_sentiments2.json', 'w') as f:
    json.dump(results, f, indent=4)
print("Predicted sentiments saved to predicted_sentiments.json.")

# Evaluation metrics for the best Logistic Regression model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Evaluation metrics for the best Logistic Regression model:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(1,6), yticklabels=range(1,6))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(1, 6)]))

print("Example of incorrect predictions:")
incorrect_indices = [i for i, (actual, pred) in enumerate(zip(y_test, y_pred)) if actual != pred][:3]
for i in incorrect_indices:
    print(f"Review ID: {ids_test[i]}")
    print(f"Review Text: {X_test_joined[i]}")
    print(f"Actual Stars: {y_test[i]}, Predicted Stars: {y_pred[i]}\n")

feature_names = vectorizer.get_feature_names_out()

# For each class (star rating), print the top 10 features contributing to that class
for class_index in range(len(best_model.classes_)):
    coefs = best_model.coef_[class_index]
    top_features = sorted(zip(coefs, feature_names), reverse=True)[:10]
    print(f"Top 10 features contributing to {best_model.classes_[class_index]} stars:")
    for coef, feature in top_features:
        print(f"{feature}: {coef}")
    print("\n")

print("All tasks completed.")