import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.svm import SVC

# load the dataset
df = pd.read_csv("iris.csv")

# descrive the dataset
print (df.describe())

# separate the data and the labels 
data = df.iloc[:, 0:4]
labels = df.iloc[:, 4]

# split into test and train
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.2)

# Standardisation scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
score = lr.score(X_train, y_train)
print(f"Training data accuracy {score}")
score = lr.score(X_test, y_test)
print(f"Testing data accuracy {score}")

# RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
score = rf.score(X_train, y_train)
print(f"Training data accuracy {score}")
score = rf.score(X_test, y_test)
print(f"Testing data accuracy {score}")

# DecisionTreeClassifier
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)
score = dt.score(X_train, y_train)
print(f"Training data accuracy {score}")
score = dt.score(X_test, y_test)
print(f"Testing data accuracy {score}")

# KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
score = knn.score(X_train, y_train)
print(f"Training data accuracy {score}")
score = knn.score(X_test, y_test)
print(f"Testing data accuracy {score}")

#SVC
svm = SVC(random_state=0, gamma='auto', C=1.0)
svm.fit(X_train, y_train)
score = svm.score(X_train, y_train)
print(f"Training data accuracy {score}")
score = svm.score(X_test, y_test)
print(f"Testing data accuracy {score}")

# make some predictions using SVC
iris = datasets.load_iris()
clf = SVC()
#clf.fit(iris.data, iris.target)
#print (list(clf.predict(iris.data[:3])))
clf.fit(iris.data, iris.target_names[iris.target])
print (list(clf.predict(iris.data[:3])))

# predict using the svm variable
print (list(svm.predict(data[:1])))

# predict using new data
df_predict = pd.DataFrame([[5.9, 3.0, 5.1, 1.8]], columns=['sepal.length','sepal.width','petal.length','petal.width'])
print (svm.predict(df_predict))
print (knn.predict(df_predict))
print (lr.predict(df_predict))
print (rf.predict(df_predict))
print (dt.predict(df_predict))

