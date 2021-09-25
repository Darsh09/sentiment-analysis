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

@app.route('/api/lstm', methods=['POST'])
def lstm():
    id = request.json['id']
    text = request.json['text']

    r = requests.post(url = 'http://localhost:5000/api/user', json = { "id": id })
    r=r.json()

    if r["user"] == True:

        # import re
        # import nltk
        # nltk.download('stopwords')
        # from nltk.corpus import stopwords
        # from nltk.stem.porter import PorterStemmer
        # corpus = []
        # review = re.sub('[^a-zA-Z]', ' ', text)
        # review = review.lower()
        # review = review.split()
        # ps = PorterStemmer()
        # review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        # review = ' '.join(review)
        # corpus.append(review)

        # from sklearn.feature_extraction.text import CountVectorizer
        # cv = CountVectorizer(max_features = 1500,ngram_range=(1,1))
        # X = cv.fit_transform(corpus).toarray()
        # # y = dataset.iloc[:, 1].values

        # msg=corpus[0]
        # ms0=cv.transform([msg])
        # print(ms0)

        # # from sklearn.externals import joblib
        # import joblib
        # nb = joblib.load('./models/nb.pkl')
        # return jsonify({ "result": nb.predict(X) })

        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        model = load_model('./lstm/best_model.h5')
        import pickle
        max_words = 5000
        max_len=50
        tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
        with open('./lstm/preprocess.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        # tokenizer.fit_on_texts(text)
        # tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
        def predict_class(text):
            '''Function to predict sentiment class of the passed text'''
            
            sentiment_classes = ['Negative', 'Neutral', 'Positive']
            max_len=50
            # print(text)
            # Transforms text to a sequence of integers using a tokenizer object
            # tokenizer = Tokenizer()
            
            xt = tokenizer.texts_to_sequences(text)
            print(xt)
            # Pad sequences to the same length
            xt = pad_sequences(xt, padding='post', maxlen=max_len)
            print(xt)
            # Do the prediction using the loaded model
            yt = model.predict(xt).argmax(axis=1)
            print(model.predict(xt))
            # Print the predicted sentiment
            k= sentiment_classes[yt[0]]
            # print(k)
            return k

        return jsonify({ "result": predict_class([text])})

    else:
        return jsonify({ "result": "User does not exist" })

if __name__ == '__main__':
    app.run(debug=True)
