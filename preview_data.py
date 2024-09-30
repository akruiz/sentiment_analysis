import json

# Open the JSON file
with open('Dataset/preprocessed_review_data.json', 'r') as file:
    data = json.load(file)

# Print the first few records
for record in data[:5]:
    print(record)