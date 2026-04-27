import os
import re
import pickle
import nltk
from flask import Flask, request, jsonify, render_template


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

MODEL_PATH = 'trained_model.sav'
VECTORIZER_PATH = 'vectorizer.sav'

model = None
vectorizer = None

def load_resources():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = pickle.load(open(MODEL_PATH, 'rb'))
        vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
        print("Model and vectorizer loaded successfully.")
    else:
        print(f"Error: Could not find {MODEL_PATH} or {VECTORIZER_PATH}. Make sure to run training first.")


port_stem = PorterStemmer()
negation_words = {"not", "no", "nor", "never", "neither", "nobody", "nothing", "nowhere", "hardly", "barely", "scarcely"}
stop_words = set(stopwords.words('english')) - negation_words 

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

load_resources()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if not model or not vectorizer:
        return jsonify({'error': 'Model or vectorizer not loaded.'}), 500
        
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided.'}), 400
        
    text = data['text']
    
    processed_text = stemming(text)
    
 
    vectorized_text = vectorizer.transform([processed_text])
    
    
    proba = model.predict_proba(vectorized_text)[0]
    
    neg_prob = proba[0]
    pos_prob = proba[1]
    
    # Define thresholds
    if pos_prob > 0.6:
        sentiment = 'Positive'
        confidence = pos_prob
    elif neg_prob > 0.6:
        sentiment = 'Negative'
        confidence = neg_prob
    else:
        sentiment = 'Neutral'
        confidence = max(pos_prob, neg_prob) 
        
    return jsonify({
        'sentiment': sentiment,
        'confidence': float(confidence),
        'probabilities': {
            'positive': float(pos_prob),
            'negative': float(neg_prob)
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
