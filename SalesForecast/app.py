from flask import Flask , render_template , request , jsonify
import pickle
import pandas as pd
from pandas import read_csv
import csv
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
app = Flask(__name__)
# df=pd.read_csv('monthly-milk-production1.csv')
# filename = 'monthly-milk-production1.csv'
# data = read_csv(filename)
# data.head()
# data.isnull().sum()
# X = data.iloc[:, 0:2]
# y = data.iloc[:, 2]
# from sklearn.linear_model import LinearRegression
# clf = LinearRegression()
# clf.fit(X, y)
# clf.score(X, y)
# model = pickle.load(open('model.pkl', 'rb'))

@app.route("/getPredictions",methods=["POST"])
def postPrediction():
    if request.method=="POST":
        period = request.json['period']
        numeric =request.json['numeric']
    filename = 'monthly-milk-production1.csv'
    # print(period,numeric)
    data = read_csv(filename)
    data.head()
    data.isnull().sum()
    X = data.iloc[:, 0:2]
    y = data.iloc[:, 2]
    from sklearn.linear_model import LinearRegression
    clf = LinearRegression()
    clf.fit(X, y)
    clf.score(X, y)
    # pickle.dump(clf, open('model.pkl', 'wb'))
    model = pickle.load(open('model.pkl', 'rb'))
    if (period == "Yearly"):
      l=[]
      year=2023
      for i in range(numeric):
        y=year
        x=list(model.predict([[year, i]]))
        l.append(x)
        year += 1
    elif (period=='Monthly'):
      l = []
      for i in range(numeric):
          x = list(model.predict([[2023, i]]))
          l.append(x)

    with open('src/assets/GFG', 'w') as f:
      write = csv.writer(f)
      write.writerows(l)

    return {'message':'received'}

if __name__=='__main__':
    app.run(debug=True)


