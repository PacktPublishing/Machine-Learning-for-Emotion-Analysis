import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

spam = pd.read_csv("spam.csv", encoding_errors="ignore")
labels = spam["v1"]
data = spam["v2"]
X_train,X_test,y_train,y_test = train_test_split(data, labels, test_size = 0.2)

count_vectorizer = CountVectorizer()
X_train_features = count_vectorizer.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_features,y_train)
X_test_features = count_vectorizer.transform(X_test)
score = knn.score(X_test_features,y_test)
print(f"Training data accuracy {score}")