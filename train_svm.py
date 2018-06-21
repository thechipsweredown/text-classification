from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np
x_train = []
y_train = []
x_temp = []
def readfile():
    label = 0
    arr = ['greeting','orther','wh_weather','yesno_weather']
    with open('./data.txt','w') as f1:
        for i in range(len(arr)):
            with open('./data/'+arr[i]) as f:
                lines = f.readlines()
                for j in range(len(lines)):
                    f1.write('__label__'+str(label) +" "+lines[j])
                    y_train.append(label)
                    x_temp.append(lines[j])
                label += 1

def Vectorizer(x_temp):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(x_temp)
    joblib.dump(vectorizer,"tf.pkl")
    vector = vectorizer.transform(x_temp)
    return vector.toarray()

def Vector(str):
    vectorizer = joblib.load("tf.pkl")
    vector = vectorizer.transform([str])
    return vector.toarray()

readfile()    
x_train = Vectorizer(x_temp)
clf = SVC(kernel='linear', degree = 3, gamma=1, C = 100)
clf.fit(x_train, y_train)
joblib.dump(clf,"model.pkl")
y_pred = clf.predict(Vector('mình là Tùng'))
print(y_pred)