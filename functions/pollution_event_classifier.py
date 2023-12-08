
def pollution_event_classifier(reports, new_reports=None):
    """
The function classifies pollution event reports based on their severity levels using a machine learning model and returns a JSON response with the classification criteria, classification report, and predicted severity for each report.
:param reports: A list of dictionaries containing the text and severity level of pollution event reports
:param new_reports: A list of dictionaries containing additional pollution event reports to update the model (optional, default: None)
:return: A JSON string containing the classification criteria, classification report, and predicted severity for each report
"""
    import json
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    
    # Sample classification criteria for severity levels
    severity_levels = {
        'Low': 'AQI: 0-50',
        'Moderate': 'AQI: 51-100',
        'Unhealthy for Sensitive Groups': 'AQI: 101-150',
        'Unhealthy': 'AQI: 151-200',
        'Very Unhealthy': 'AQI: 201-300',
        'Hazardous': 'AQI: 301-500'
    }

    # Extract features and labels from reports
    texts = [report['text'] for report in reports]
    labels = [report['severity'] for report in reports]
    
    # Encoding labels to ensure matching with target names in classification report
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    severity_level_names = label_encoder.classes_
    
    # Split the dataset into training and test sets
    texts_train, texts_test, encoded_labels_train, encoded_labels_test = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42
    )
    
    # Create a machine learning pipeline with a CountVectorizer and a Naive Bayes classifier
    model = make_pipeline(CountVectorizer(), MultinomialNB())

    # Train the model
    model.fit(texts_train, encoded_labels_train)
    
    # Optionally, update the model with new reports
    if new_reports:
        new_texts = [report['text'] for report in new_reports]
        new_labels = [report['severity'] for report in new_reports]
        new_encoded_labels = label_encoder.transform(new_labels)
        model.fit(new_texts, new_encoded_labels)
    
    # Predict the severity of new reports
    predicted_encoded_labels = model.predict(texts_test)
    predicted_labels = label_encoder.inverse_transform(predicted_encoded_labels)
    
    # Evaluate the model
    report = classification_report(
        encoded_labels_test, predicted_encoded_labels, target_names=severity_level_names
    )
    
    # Create a JSON response
    response = {
        'classification_criteria': severity_levels,
        'classification_report': report,
        'predicted_severity': dict(zip(texts_test, predicted_labels))
    }
    
    return json.dumps(response, indent=2)