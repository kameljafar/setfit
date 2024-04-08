import torch
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import evaluate
import os
from waitress import serve
from config import *
from models.test2.labels_information import *
LABELS = TRAINING_LABELS
from datasets import Dataset
from flask_httpauth import HTTPBasicAuth

from setfit import SetFitModel, SetFitTrainer
port = ID_PORT
host = ID_HOST
auth = HTTPBasicAuth()

model_directory = MODEL_NAME_OR_PATH
model = SetFitModel.from_pretrained(model_directory)
first_category_dict = FIRST_CATEGORY_DICT
sec_category_dict = SECOND_CATEGORY_DICT
authorized_users = {
    'admindemo': 'admindemo#101info',
}
app = Flask(__name__, template_folder='templates')
CORS(app)
def check_auth(username, password):
    return username in authorized_users and password == authorized_users[username]

auth.verify_password(check_auth)

@app.route ('/')
@auth.login_required

def homepage ():
    username = request.authorization.username
    password = request.authorization.password
    if username in authorized_users and password == authorized_users[username]:
        print (username)
        return render_template('index.html')
    else:
        return jsonify(['Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'}])
@app.after_request
def add_cache_control_header(response):
    response.headers['Cache-Control'] = 'max-age=60'
    return response

@app.route('/predict', methods=['POST'])


def predict():
    final_output =[]
    data = request.get_json(force=True)
    texts = data['texts']
    preds = model([texts])
    output = [[f for f, p in zip(LABELS, ps) if p] for ps in preds]
    scor = model.predict_proba([texts])
    preds_score=max(scor[0])
    first_category=[]
    second_category=[]
    first_false = []
    second_false = []
    if len(output[0]) == 2:
        en_first_cat = output[0][0]
        en_second_cat = output[0][1]
        print (en_first_cat, "1111111111111111111111")
        print (en_second_cat, "2222222222222222222")
        for opt in output[0]:
            for firat_item in first_category_dict:
                if opt == firat_item['en_className']:
                    first_category.append(firat_item["ru_className"])

            for sec_item in sec_category_dict:
                if opt == sec_item['en_className']:
                    second_category.append(sec_item["ru_className"])
        catigories = {"Тип_нормы": first_category[0],
                      "Профиль_нормы": second_category[0],
                      "score": preds_score.item()}
    elif len(output[0]) == 1:
        for opt in output:
            for firat_item in first_category_dict:
                if opt == firat_item['en_className']:
                    first_category.append(firat_item["ru_className"])
                else:
                    first_false.append("First category is empty .. ")
            for sec_item in sec_category_dict:
                if opt[0] == sec_item['en_className']:
                    second_category.append(sec_item["ru_className"])
                else:
                    second_false.append("Second category is empty .. ")
        catigories = {
            "Тип_нормы": first_category[0] if first_category and first_category[0] else first_false[0] if first_false else None,
            "Профиль_нормы": second_category[0] if second_category and second_category[0] else second_false[0] if second_false else None,
            "score": preds_score.item()
        }

    else:
        first_category.append("First category is empty .. ")
        second_category.append("Second category is empty .. ")

        catigories = {"Тип_нормы": first_category[0],
                      "Профиль_нормы": second_category[0],
                      "score": preds_score.item()}
    final_output.append({'predictions': catigories})
    return final_output
if __name__ == '__main__':
    print (f"server running on {host}/{port}")
    # serve(app, host=host, port=port)

    app.run(host=host, port=port, debug=True)