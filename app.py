import numpy as np
from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

try:
    model = joblib.load('model_nb.pkl')
    cv = joblib.load('cv.pkl')
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure model_nb.pkl and cv.pkl are in the root directory.")
    model = None
    cv = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        if model is None or cv is None:
            return render_template('index.html', 
                                 prediction_text='Error: Model files not found. Please check server logs.')
        
        text = request.form.get('Review', '').strip()
        
        if not text:
            return render_template('index.html', 
                                 prediction_text='Please enter a review.')
        
        data = [text]
        vectorizer = cv.transform(data).toarray()
        prediction = model.predict(vectorizer)
        
        if prediction[0] == 1:
            result = 'The review is Positive 😊'
        else:
            result = 'The review is Negative 😞'
        
        return render_template('index.html', 
                             review_text=text, 
                             prediction_text=result)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
