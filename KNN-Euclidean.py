df = pd.read_csv("Yuz.csv", sep=",")
data = df[['Cumle', 'Sinif']]

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

comment_words = ''
stopwords = set(STOPWORDS)

for val in data["Cumle"]:

    val = str(val)

    tokens = val.split()

    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens) + " "

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(comment_words)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()
tokenizer = Tokenizer(split=' ', num_words=40000)
tokenizer.fit_on_texts(data['Cumle'].values)
X = tokenizer.texts_to_sequences(data['Cumle'].values)
X = pad_sequences(X, maxlen=40)
X[-1]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
knn_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_euclidean = knn_euclidean.fit(X_test, Y_test)
y_pred = knn_euclidean.predict(X_test)
print("KneighborsClassifier: ", accuracy_score(Y_test, y_pred))

from sklearn.metrics import f1_score

print("KNN-euclidean Distance F1 Score:", f1_score(Y_test, knn_euclidean.predict(X_test), average='weighted'))
print("KNN-euclideanDistance Accuracy:", accuracy_score(Y_test, y_pred))