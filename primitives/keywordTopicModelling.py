from env.IR_CPS_TechSynthesis.env import SynthesisManager, Section
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

# Ensure required NLTK datasets are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

def keyword_topic_modeler(bot: SynthesisManager, documents, num_topics=5):
    """
    Analyzes the given documents to extract topics using LDA.

    Args:
        bot (SynthesisManager): The synthesis manager instance.
        documents (list of str): List of documents to be analyzed.
        num_topics (int): Number of topics to extract.

    Returns:
        dict: A dictionary of extracted topics and their keywords.
    """
    def preprocess(document):
        """
        Preprocesses the document by lowercasing, removing punctuation, stopwords, and lemmatizing.
        """
        stop = set(stopwords.words('english'))
        exclude = set(string.punctuation)
        stop_free = " ".join([word for word in document.lower().split() if word not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        lemmatizer = WordNetLemmatizer()
        normalized = " ".join(lemmatizer.lemmatize(word) for word in punc_free.split())
        return normalized

    def extract_topics(lda_model, vectorizer):
        """
        Extracts the topics along with their keywords from the LDA model.
        """
        words = vectorizer.get_feature_names_out()
        topics = {}
        for idx, topic in enumerate(lda_model.components_):
            topics[f"Topic {idx+1}"] = [words[i] for i in topic.argsort()[:-10 - 1:-1]]
        return topics

    # Preprocessing documents
    preprocessed_docs = [preprocess(doc) for doc in documents]

    # Vectorizing documents
    vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(preprocessed_docs)

    # Fitting LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    # Extracting topics
    topics = extract_topics(lda, vectorizer)
    
    # # Update results in resources
    # bot.add_or_update_results_in_resources(topics)

    return topics
