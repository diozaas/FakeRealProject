import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle
import json

# Load data
hoax_data = pd.read_csv('datasets/hoax_clean.csv')
real_data = pd.read_csv('datasets/satker_clean.csv')

# Add labels
hoax_data['label'] = 0  # 0 for fake news
real_data['label'] = 1  # 1 for real news

# Combine datasets
data = pd.concat([hoax_data, real_data], ignore_index=True)

# Preprocess data: Replace NaN values with empty strings
data = data.assign(Title=data['Title'].fillna(''))
data = data.assign(Description=data['Description'].fillna(''))

X_title = data['Title']
X_description = data['Description']
X_both = data['Title'] + " " + data['Description']
y = data['label']

# Function to evaluate models
def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Vectorize data
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Define models
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Random Forest': RandomForestClassifier()
    }
    
    results = {}
    for model_name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()
        results[model_name] = {'accuracy': accuracy, 'report': report, 'confusion_matrix': cm}

    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train_vec, y_train)
    # Save the model and vectorizer
    with open('random_forest_model.pkl', 'wb') as model_file:
        pickle.dump(random_forest_model, model_file)

    with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    
    return results

# Evaluate models with different feature sets
results_title = evaluate_models(X_title, y)
results_description = evaluate_models(X_description, y)
results_both = evaluate_models(X_both, y)

# Prepare data for JSON output
output = {
    'Title Only': results_title,
    'Description Only': results_description,
    'Title and Description': results_both
}

# Save results to JSON file
with open('results.json', 'w') as json_file:
    json.dump(output, json_file)

# Prepare data for chart
accuracy_data = {
    'Feature Set': ['Title Only', 'Description Only', 'Title and Description'],
    'Naive Bayes': [results_title['Naive Bayes']['accuracy'], results_description['Naive Bayes']['accuracy'], results_both['Naive Bayes']['accuracy']],
    'Logistic Regression': [results_title['Logistic Regression']['accuracy'], results_description['Logistic Regression']['accuracy'], results_both['Logistic Regression']['accuracy']],
    'Random Forest': [results_title['Random Forest']['accuracy'], results_description['Random Forest']['accuracy'], results_both['Random Forest']['accuracy']]
}

accuracy_df = pd.DataFrame(accuracy_data)

# Plot accuracy comparison chart using matplotlib
plt.figure(figsize=(10, 6))
ax = accuracy_df.plot(x='Feature Set', kind='bar', rot=0)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Feature Set')
plt.legend(loc='lower right')

# Add accuracy values on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() * 1.005, p.get_height() * 1.005))

plt.tight_layout()
plt.savefig('static/FakeRealResultChart.png')
plt.close()
