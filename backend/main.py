#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Dependencies
from flask import Flask, render_template, url_for
from flask import request, redirect
from flask_cors import CORS
from flask_restful import Api
from flask_restful import Resource
from torchvision import transforms
from flask_socketio import SocketIO, emit

from resources import Transfer


app = Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*")
transfer = Transfer()

@socketio.on('startProcess')
def test_connect(req):
    transfer.startProcess(req)

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0')