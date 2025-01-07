from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    
    # Run your prediction logic here (replace with your actual model prediction)
    prediction = "Sample Prediction"
    
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
