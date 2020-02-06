#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >

from argparse import ArgumentParser
from flask import Flask, request, jsonify

from dnnhelper import load_model, classify_text


app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():

    text_dict_list = request.json
    preds = classify_text(app.config['model'],
                          app.config['labels'],
                          app.config['vocab'],
                          text_dict_list)
    return jsonify(preds)


def parse_args():
    parser = ArgumentParser(description='SilkNOW Text Classifier Server')

    parser.add_argument('--model-load', type=str, help='model path',
                        required=True)
    parser.add_argument('--port', type=int, help='internet port',
                        default=5000)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model, labels, vocab = load_model(args.model_load)
    app.config['model'] = model
    app.config['labels'] = labels
    app.config['vocab'] = vocab
    app.run(port=args.port)
