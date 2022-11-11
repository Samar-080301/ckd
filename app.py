# flask route for ml model
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import keras
import ast

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data_raw = request.get_json()
        print(data_raw)
        new_dict = ast.literal_eval(data_raw)
        data=[]
        for i in new_dict.values():
            data.append(i)
        data = np.array(data)
        data = np.array(data.reshape(1, -1))
        print(data)
        # load model
        model = keras.models.load_model('model.pkl', 'rb')
        # make prediction
        prediction = model.predict(data)
        print(prediction)
        return jsonify({'prediction': prediction.tolist()})
    else:
        return jsonify({'prediction': 'error'})

# run flask app
if __name__ == '__main__':
    app.run(debug=True)