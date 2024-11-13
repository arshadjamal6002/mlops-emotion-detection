from flask import Flask, render_template, request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import mlflow
import pickle

mlflow.set_tracking_uri('https://dagshub.com/arshadjamal6002/mlops-emotion-detection.mlflow')
dagshub.init(repo_owner='arshadjamal6002', repo_name='mlops-emotion-detection', mlflow=True)

# to load trained vectorizer to perform bow on input text
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

app = Flask(__name__)

# load the model from model registry
# need model name and version of the model
model_name = 'my_model'
model_version = 1

# build the model uri using the above
model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)  # use the uri to fetch the model

# dynamically fetch

@app.route('/')
def home():
    return render_template('index.html') # load the front end

@app.route('/predict', methods = ['POST'])   # tell the method how the data is coming in
def predict():

    text = request.form['text']    # whatever is being typed the flask backend is able to receive that
    # PREDICTION


    # clean the text coming from user
    text = normalize_text(text)

    # apply feature making by applying bag of words
    features = vectorizer.transform([text])     # needs list of objects not just normal text thus []

    # make the prediction using the models
    result = model.predict(features)

    # show to user
    return str(result[0])    # stored in array of 0 or 1


app.run(debug = True)