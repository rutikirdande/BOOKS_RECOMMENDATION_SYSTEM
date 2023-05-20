import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_books_data(csv_file):
    """
    Load the book data from a CSV file and return a DataFrame.
    """
    return pd.read_csv(csv_file)

def preprocess_data(books_data):
    """
    Preprocess the book data by replacing missing values and creating the 'description' column.
    """
    books_data['description'] = books_data['Author'] + ' ' + books_data['Genre'] + ' ' + books_data['Publisher']
    books_data['description'] = books_data['description'].fillna('')
    return books_data

def create_tfidf_matrix(books_data):
    """
    Create a TF-IDF matrix based on the book descriptions.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(books_data['description'])
    return tfidf_matrix

def compute_cosine_similarities(tfidf_matrix):
    """
    Compute the cosine similarity matrix based on the TF-IDF matrix.
    """
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_similarities

def get_book_recommendations(title, cosine_similarities, books_data, top_n=5):
    """
    Get book recommendations based on a given book title.
    """
    book_index = books_data[books_data['Title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_similarities[book_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_books_indices = [index for index, _ in similarity_scores[1:top_n+1]]
    return books_data['Title'].iloc[top_books_indices]
