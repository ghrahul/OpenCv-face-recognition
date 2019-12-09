from flask import Flask
from datacreator import data_collect
from trainer import train
from detect import detector

app = Flask(__name__)

@app.route('/register')
def register():
    print("Collecting data")
    data_collect()
    print("Training")
    train()
    return 'Training done'

@app.route('/recognize')
def recognition():
    detector()

app.run(host='0.0.0.0', debug=True)