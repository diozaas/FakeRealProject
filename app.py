from flask import Flask, request, render_template, redirect, url_for
import pickle
import subprocess
import json

# Load the trained model and vectorizer
with open('random_forest_model.pkl', 'rb') as model_file:
    random_forest_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    description = request.form['description']
    combined_text = title + " " + description
    vectorized_text = vectorizer.transform([combined_text])
    prediction = random_forest_model.predict(vectorized_text)
    result = 'Real' if prediction[0] == 1 else 'Fake'
    return render_template('index.html', prediction_text=f'The news is {result}')

@app.route('/process_and_training', methods=['GET'])
def process_and_training():
    subprocess.run(['python', 'fakeReal.py'])
    return redirect(url_for('results'))

@app.route('/results')
def results():
    # Read results from JSON file
    with open('results.json', 'r') as json_file:
        results = json.load(json_file)
    return render_template('results.html', results=results, chart_url='static/FakeRealResultChart.png')

if __name__ == "__main__":
    app.run(debug=True)
