from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load the book data from a local CSV file
csv_file = "data/books.csv"
books_data1 = pd.read_csv(csv_file)

# Replace np.nan with empty strings in the 'description' column
books_data1['description'] = books_data1['Author'] + ' ' + books_data1['Genre'] + ' ' + books_data1['Publisher']
books_data1['description'] = books_data1['description'].fillna('')

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Apply the vectorizer to the book descriptions
tfidf_matrix = vectorizer.fit_transform(books_data1['description'])

# Compute the cosine similarity matrix
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get book recommendations based on book title
def get_book_recommendations(title, cosine_similarities, books_data1, top_n=5):
    # Get the index of the book with the given title
    book_indices = books_data1[books_data1['Title'] == title].index

    if len(book_indices) == 0:
        # No matching records found for the given book title
        return []

    book_index = book_indices[0]

    # Rest of the code...


    # Get the pair-wise similarity scores of all books with the given book
    similarity_scores = list(enumerate(cosine_similarities[book_index]))

    # Sort the books based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the top N most similar books (excluding itself)
    top_books_indices = [index for index, _ in similarity_scores[1:top_n + 1]]

    # Return the top N book titles
    return books_data1['Title'].iloc[top_books_indices]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    book_title = request.form['book_title']
    recommendations = get_book_recommendations(book_title, cosine_similarities, books_data1)
    return render_template('recommendations.html', book_title=book_title, recommendations=recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
