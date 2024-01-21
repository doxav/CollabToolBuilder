
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Assuming nltk's stopwords and tokenizer are already installed
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocesses the text by tokenizing, removing stopwords, and lowercasing.
    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    stopwords = set(nltk.corpus.stopwords.words('english'))

    tokens = tokenizer.tokenize(text.lower())
    filtered_words = [w for w in tokens if w not in stopwords]
    return ' '.join(filtered_words)

def check_plagiarism(input_text, database_texts):
    """
    Checks for plagiarism by comparing the input text with a database of texts.
    """
    documents = [input_text] + database_texts
    preprocessed_docs = [preprocess_text(doc) for doc in documents]

    # Vectorizing documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

    # Calculating cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    # Checking for potential plagiarism
    plagiarism_results = {}
    for idx, sim_score in enumerate(cosine_similarities[0]):
        if sim_score > 0.5:  # Threshold for similarity; can be adjusted
            plagiarism_results[f'Document {idx+1}'] = sim_score

    return plagiarism_results