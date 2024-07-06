from flask import Flask, request, render_template
import pickle

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

if __name__ == "__main__":
    app.run(debug=True)
