import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

print("Loading data...")
with open('predicted_sentiments.json') as f:
    data = json.load(f)

# Extract the actual and predicted stars
actual_stars = [entry['actual_stars'] for entry in data]
predicted_stars = [entry['predicted_stars'] for entry in data]

print("Creating histogram...")
# Create a histogram with bins [1, 2, 3, 4, 5]
bins = [1, 2, 3, 4, 5, 6]  # Adding 6 to include the upper edge of the last bin
plt.hist(actual_stars, bins=bins, alpha=0.5, label='Actual Stars', align='left')
plt.hist(predicted_stars, bins=bins, alpha=0.5, label='Predicted Stars', align='left')
plt.xticks([1, 2, 3, 4, 5])
plt.xlabel('Stars')
plt.ylabel('Frequency')
plt.title('Distribution of Actual and Predicted Stars')
plt.legend()
plt.savefig('histogram.png')
plt.close()

print("Binarizing stars for precision-recall and ROC curves...")
# Binarize the actual and predicted stars
actual_stars_bin = label_binarize(actual_stars, classes=[1, 2, 3, 4, 5])
predicted_stars_bin = label_binarize(predicted_stars, classes=[1, 2, 3, 4, 5])

print("Creating precision-recall curves...")
# Plot precision-recall curve for each class
for i in range(5):
    precision, recall, _ = precision_recall_curve(actual_stars_bin[:, i], predicted_stars_bin[:, i])
    plt.plot(recall, precision, label=f'{i+1} Star')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('precision_recall_curve.png')
plt.close()

print("Creating ROC curves...")
# Plot ROC curve for each class
for i in range(5):
    fpr, tpr, _ = roc_curve(actual_stars_bin[:, i], predicted_stars_bin[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{i+1} Star (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()

print("Creating box plot...")
# Create a DataFrame for easier plotting
df = pd.DataFrame({'Actual Stars': actual_stars, 'Predicted Stars': predicted_stars})

# Plot box plot
plt.figure(figsize=(10, 7))
sns.boxplot(x='Actual Stars', y='Predicted Stars', data=df)
plt.xlabel('Actual Stars')
plt.ylabel('Predicted Stars')
plt.title('Box Plot of Predicted Stars for Each Actual Star Rating')
plt.xticks([1, 2, 3, 4, 5])
plt.savefig('box_plot.png')
plt.close()

print("Creating error distribution plot...")
# Calculate the prediction errors
errors = [actual - predicted for actual, predicted in zip(actual_stars, predicted_stars)]

# Plot the error distribution
plt.hist(errors, bins=10, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.savefig('error_distribution.png')
plt.close()

print("All plots have been created and saved.")