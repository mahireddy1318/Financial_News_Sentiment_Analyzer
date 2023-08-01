# app.py
from flask import Flask, render_template, request
import  pickle
app = Flask(__name__)


file_path = r'C:\Users\Siva Reddy\Desktop\projects\Financial_News_Sentiment_Analyzer\Finanial_news_headline.pkl' 

with open(file_path , 'rb') as f:
    loaded_model = pickle.load(f)

# Load and prepare your NLTK model (same as before)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            text = request.form['text']  # Get the input text from the form

            # Process the input text using your NLTK model
            result = loaded_model.predict([text])

            # Render the template with the prediction
            return render_template('index.html', prediction=result[0])
        except Exception as e:
            return render_template('index.html', error=str(e))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
