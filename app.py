from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
import requests
import json

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

ma = Marshmallow(app)

class Users(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(200), unique= True)

    def __init__(self, name, email):
        self.name  = name
        self.email = email

class UsersSchema(ma.Schema):
    class Meta:
        fields = ('id', 'name', 'email')

user_schema = UsersSchema()
users_schema = UsersSchema(many=True)

@app.route('/api/users', methods=['POST'])
def users():
    name = request.json['name']
    email = request.json['email']

    new_user = Users(name, email)

    db.session.add(new_user)
    db.session.commit()

    return user_schema.jsonify(new_user)

@app.route('/api/user', methods=['POST'])
def userExists():
    id = request.json['id']
    print(id)
    userExists = Users.query.get(id)

    if(userExists is not None):
        return jsonify({"user": True})

    else:
        return jsonify({"user": False})

@app.route('/api/sentiment', methods=['POST'])
def sentiment():
    id = request.json['id']
    text = request.json['text']

    r = requests.post(url = 'http://localhost:5000/api/user', json = { "id": id })
    r=r.json()

    if r["user"] == True:
        from textblob import TextBlob
        testimonial = TextBlob(text)
        return jsonify({ "result": testimonial.sentiment.polarity })

    else:
        return jsonify({ "result": "User does not exist" })

@app.route('/api/tags', methods=['POST'])
def tags():
    id = request.json['id']
    text = request.json['text']

    r = requests.post(url = 'http://localhost:5000/api/user', json = { "id": id })
    r=r.json()

    if r["user"] == True:
        from textblob import TextBlob
        testimonial = TextBlob(text)
        print(testimonial.noun_phrases)
        return jsonify({ "result": testimonial.noun_phrases })

    else:
        return jsonify({ "result": "User does not exist" })

@app.route('/api/images', methods=['POST'])
def ocr():
    id = request.json['id']
    # text = request.json['text']

    r = requests.post(url = 'http://localhost:5000/api/user', json = { "id": id })
    r=r.json()

    if r["user"] == True:
        from PIL import Image
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\91961\.virtualenvs\bce-_gukpind\lib\site-packages'
        res = pytesseract.image_to_string(Image.open('test.jpg'))
        return jsonify({ "result": res })

    else:
        return jsonify({ "result": "User does not exist" })

if __name__ == '__main__':
    app.run(debug=True)
