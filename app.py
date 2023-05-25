from flask import Flask, render_template, request
import numpy as np
import pickle
import instaloader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
# ct = pickle.load(open('ct.pkl', 'rb'))

def load_model():
    with open("model.pkl", "rb") as file:
        data = pickle.load(file)
    return data


def load_transform():
    with open("ct.pkl", "rb") as file:
        data = pickle.load(file)
    return data


model = load_model()

ct = load_transform()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = str(request.form['category'])
    val2 = int(request.form['time_posted'])
    val3 = int(request.form['num_followers'])
    val4 = int(request.form['num_posts'])
    # print(val1)
    # print(val2)
    # print(val3)
    # print(val4)
    # val1 = input("category")
    # val2 = int(input("time"))
    # val3 = int(input("followers"))
    # val4 = int(input("posts"))
    category_encoded = ct.transform([[val1, val2, val3, val4]])
    predicted_likes = model.predict(category_encoded)
    # print("Predicted number of likes on the post", ":", predicted_likes * 10)
    # arr = np.array([val1, val2, val3, val4])
    # arr = arr.astype(np.float64)
    # pred = model.predict([arr])

    return render_template('index.html', data=int(predicted_likes))


if __name__ == '__main__':
    app.run(debug=True)
