import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("Starting script...")

# Initialize NLTK resources
print("Downloading and initializing NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess the text field
preprocessed_data = []

print("Loading and processing JSON file...")
i = 1
# Load the JSON file
with open('Dataset/first_half_data.json', 'r') as file:
    for line in file:
        review = json.loads(line)
        text = review['text']
        
        # Tokenization
        tokens = nltk.word_tokenize(text)
        
        # Lowercasing
        tokens = [token.lower() for token in tokens]
        
        # Stopwords removal
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Store the review id, stars, and preprocessed text
        preprocessed_review = {
            'review_id': review['review_id'],
            'stars': review['stars'],
            'preprocessed_text': tokens
        }
        preprocessed_data.append(preprocessed_review)
        print(f"Processed review {i}")
        i += 1

print("Saving preprocessed data...")
# Save the preprocessed data
with open('Dataset/preprocessed_first_half_review_data.json', 'w') as file:
    json.dump(preprocessed_data, file)

print("Script completed.")