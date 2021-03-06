import string

from models.resnet_rnn import ResNetRNN
from flask import Flask, jsonify, request
from captcha_solver import  CaptchaSolver

app = Flask(__name__)

config = {
    'input_shape': (3,32,100),
    "map_to_seq_hidden": 512,
    "rnn_hidden": 128,
    'checkpoint':'checkpoints/20211109_5879_resnet_v2_rnn_ctc.pt',
    'characters': "-"+ string.digits + string.ascii_lowercase,
    'decode_method':'beam_search',
    'beam_size':10,
}

model = CaptchaSolver(config)
def check_file(file_name:str):
    return file_name.endswith('png') or file_name.endswith('jpg') or file_name.endswith('jpeg')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if not check_file(file.filename):
            return 'File must be end with png or jpg or jpeg!'
        label = model.predict(file.read())
        return jsonify({'name': file.filename, 'text': label})

@app.route('/')
def index():
	return '''
	<!doctype html>
	<html>
	<body>
	<form action='/predict' method='post' enctype='multipart/form-data'>
  		<input type='file' name='file'>
	<input type='submit' value='Upload'>
	</form>
	'''

if __name__ == '__main__':
    app.run(port= 8080)