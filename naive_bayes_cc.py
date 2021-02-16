from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import *

x = cc_df[['amnt_avg_ratio']].copy()
y = cc_df['isFradulent'].copy()


def split_test_train(x, y, test_size):
    return train_test_split(x, y, test_size=test_size, random_state=0)


X_train, X_test, y_train, y_test = split_test_train(x, y, 0.3)



def create_naive_bayes_classifier(X, y):
    model = GaussianNB()
    model.fit(X, y)
    return model


model = create_naive_bayes_classifier(X_train, y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(len(y_pred) == len(y_test))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='2')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



ratios = np.arange(0.1, 0.9, 0.1)
accuracy = []
for ratio in ratios:
    X_train, X_test, y_train, y_test = split_test_train(x, y, ratio)
    model = create_naive_bayes_classifier(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))

plt.grid(True)
plt.plot(ratios, accuracy, 'r--')
plt.xlabel('Size of Test Set')
plt.ylabel('Accuracy')
plt.title('Accuracy over Different Sizes of Train Set', fontsize=15)
plt.show()

predicted_data = X_test.copy()
predicted_data["isFradulent"] = y_pred
print(predicted_data)

graph = sns.FacetGrid(cc_df, col='isFradulent')
graph.map(plt.hist, 'amnt_avg_ratio', bins=10)
plt.show()
graph = sns.FacetGrid(predicted_data, col='isFradulent')
graph.map(plt.hist, 'amnt_avg_ratio', bins=10)
plt.show()




