import logisticreg
import csv
import numpy as np



n_test = 80
X = []
y = []
with open ('iris.data') as fp:
    for row in csv.reader(fp):
        if row[4] == "Iris-setosa":
            y.append(0)
        else:
            y.append(1)
        X.append(row[:4])

     
y = np.array(y,dtype = np.float64)
X = np.array(X,dtype = np.float64)

#100~150行目のデータを削除し、二値データに変換
y = np.delete(y,slice(100,150),0)
X = np.delete(X,slice(100,150),0)

#yをreshape
y = np.reshape(y,(100,1))
#リストの結合
data = np.block([y,X])

#行をシャッフル
np.random.shuffle(data)

#リストの分割
y, X = np.hsplit(data, [1])

#yを再度reshape
y = np.reshape(y,(100,))


y = np.array(y,dtype = np.float64)
X = np.array(X,dtype = np.float64)
y_train = y[:-n_test]
X_train = X[:-n_test]
y_test = y[-n_test:]
X_test = X[-n_test:]
model = logisticreg.LogisticRegression()
model.fit(X_train,y_train)

y_predict = model.predict(X_test)
n_hits  = (y_test == y_predict).sum()
print('Accuracy:{}/{} = {}'.format(n_hits,n_test,
                                   n_hits/n_test))
