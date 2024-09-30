import json
import matplotlib.pyplot as plt

def summarize_reviews(file_path):
    # Read the data from the JSON file
    with open(file_path) as file:
        data = [json.loads(line) for line in file]
    # Extract the "stars" values
    stars = [review['stars'] for review in data]

    # Summarize the data
    total_reviews = len(stars)
    average_stars = sum(stars) / total_reviews
    min_stars = min(stars)
    max_stars = max(stars)

    # Print the summary data
    print('Total reviews:', total_reviews)
    print('Average stars:', average_stars)
    print('Minimum stars:', min_stars)
    print('Maximum stars:', max_stars)

    # Visualize the distribution of "stars" values
    plt.hist(stars, bins=[1, 2, 3, 4, 5, 6], align='left')
    plt.xlabel('Stars')
    plt.ylabel('Frequency')
    plt.title('Distribution of Stars')
    plt.show()
    return

# Call the function with the path to your JSON file
summarize_reviews('Dataset/first_half_data.json')