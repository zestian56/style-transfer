#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Dependencies
from flask import Flask, render_template, url_for
from flask import request, redirect
from flask_restful import Api
from flask_restful import Resource
from torchvision import transforms

from resources import Transfer


app = Flask(__name__)
api = Api(app)

@app.route("/")
def hello():
    return "<h1> Hello this is a test page </h1>"

api.add_resource(Transfer, '/transfer')

if __name__ == "__main__":
    app.run(host='0.0.0.0')