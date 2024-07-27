from flask import Flask, request, jsonify, render_template

import pickle
import numpy as np

model_path ='place.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['post'])

def preditct():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)
    output = 'placed' if prediction[0] == 1 else 'Not Placed'

    return render_template('index.html', prediction_text = 'prediction: {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)
