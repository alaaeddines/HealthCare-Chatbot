# -*- coding: utf-8 -*-
"""
Created on Fri May 21 23:32:56 2021

@author: ALAA
"""

from flask import Flask, render_template, jsonify, request
import random
import json
import torch
from nltk_utils import bag_of_words, tokenize
from model import NeuralNet
from gevent.pywsgi import WSGIServer


app = Flask(__name__)

try:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	with open('intents.json', 'r') as f:
		intents = json.load(f)

	FILE = "data.pth"
	data = torch.load(FILE)

	input_size = data["input_size"]
	hidden_size = data["hidden_size"]
	output_size = data["output_size"]
	all_words = data["all_words"]
	tags = data["tags"]
	model_state = data["model_state"]

	model = NeuralNet(input_size, hidden_size, output_size).to(device)
	model.load_state_dict(model_state)
	model.eval()

except Exception as e:
	print(e)


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/', methods=["POST"])
def chatbot_msg():
	if request.method == "POST":
		user_data = request.json

		sentence = user_data['msg']
	
		sentence = tokenize(sentence)
		X = bag_of_words(sentence, all_words)
		X = X.reshape(1, X.shape[0])
		X = torch.from_numpy(X)

		output = model(X)
		_, predicted = torch.max(output, dim=1)
		tag = tags[predicted.item()]
		probs = torch.softmax(output, dim=1)
		prob = probs[0][predicted.item()]

		if prob.item() > 0.75:
			for intent in intents["intents"]:
				if tag == intent["tag"]:
			
					return jsonify(msg=random.choice(intent['responses']))
		else:
			return jsonify(msg="I do not understand...")



if __name__ == '__main__':
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()
	# app.run(port=5000, debug=True)
     app.run('localhost', 5000)