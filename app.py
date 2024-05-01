from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the model using the pipeline API for simplicity.
# Replace 'model-name' with the actual model name you want to use.
model = pipeline('text-generation', model='gpt-2', device=0)  # device=0 for CPU

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the POST request.
        data = request.get_json(force=True)
        text = data['text']

        # Generate a response from the model
        model_response = model(text, max_length=50, num_return_sequences=1)

        # Return the model's response as JSON
        return jsonify(model_response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
