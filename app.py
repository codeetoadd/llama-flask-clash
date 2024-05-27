from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import firebase_admin
from firebase_admin import credentials, firestore
import torch
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize Firestore
cred = credentials.Certificate("C:\\Users\\swati\\Downloads\\safebite-9c259-firebase-adminsdk-s30oq-896ede60e9.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

@app.route('/process', methods=['POST'])
def process():
    app.logger.info("Processing request...")
    data = request.json
    if not data or 'keywords' not in data:
        return jsonify({"error": "Invalid input"}), 400
    app.logger.info(f"Received data: {data}")
    
    keywords = data['keywords']
    custom_instruction = f"Show recipes with alternatives of {', '.join(keywords)} used in them but same nutrition and similar taste."

    # Generate the output using the model
    inputs = tokenizer(custom_instruction, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)  # Adjust max_length as needed

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    app.logger.info(f"Response from model: {response_text}")

    # Store result in Firestore
    doc_ref = db.collection('processed_data').add({'keywords': keywords, 'result': response_text})

    return jsonify({"result": response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
