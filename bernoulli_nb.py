from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import BernoulliNB
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import *

x = cc_df[['isHighRiskCountry','isForeignTransaction']].copy()
y = cc_df['isFradulent'].copy()


def split_test_train(x, y, test_size):
    return train_test_split(x, y, test_size=test_size, random_state=0)


X_train, X_test, y_train, y_test = split_test_train(x, y, 0.4)



def create_naive_bayes_classifier(X, y):
    model = BernoulliNB()
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

print(model.predict_proba(x))

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

fraud_test_data = X_test.copy()
fraud_test_data["isFradulent"] = y_test
fraud_predicted_data = X_test.copy()
fraud_predicted_data["isFradulent"] = y_pred

#
labels = ['Fraud', 'Not Fraud']
real_not_fraud = fraud_test_data["isForeignTransaction"][fraud_test_data["isFradulent"] == 0].count()
real_fraud = fraud_test_data["isForeignTransaction"][fraud_test_data["isFradulent"] == 1].count()
values_real = [real_not_fraud, real_fraud]
predict_not_fraud = fraud_predicted_data["isForeignTransaction"][fraud_predicted_data["isFradulent"] == 0].count()
predict_fraud = fraud_predicted_data["isForeignTransaction"][fraud_predicted_data["isFradulent"] == 1].count()
values_predict = [predict_not_fraud, predict_fraud]
width = 0.5
fig = plt.figure(facecolor="white")
ax = fig.add_subplot(1, 1, 1)
ax1 = ax.bar(labels, values_real, width, color='seagreen', edgecolor='springgreen', linewidth=0.5,
             label='Real data')
ax2 = ax.bar(labels, values_predict, width, bottom=values_real, color='teal',
             linewidth=0.5, label='Predict data')
for r1, r2 in zip(ax1, ax2):
    h1 = r1.get_height()
    h2 = r2.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="center", color="grey",
             fontsize=10, fontweight="bold")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="center", color="grey",
             fontsize=10, fontweight="bold")
plt.title('Bar plot for fraud and not fraud transaction \n based on foreign country', weight='bold')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#
labels = ['Fraud', 'Not Fraud']
real_not_fraud = fraud_test_data["isHighRiskCountry"][fraud_test_data["isFradulent"] == 0].count()
real_fraud = fraud_test_data["isHighRiskCountry"][fraud_test_data["isFradulent"] == 1].count()
values_real = [real_not_fraud, real_fraud]
predict_not_fraud = fraud_predicted_data["isHighRiskCountry"][fraud_predicted_data["isFradulent"] == 0].count()
predict_fraud = fraud_predicted_data["isHighRiskCountry"][fraud_predicted_data["isFradulent"] == 1].count()
values_predict = [predict_not_fraud, predict_fraud]
width = 0.5
fig = plt.figure(facecolor="white")
ax = fig.add_subplot(1, 1, 1)
ax1 = ax.bar(labels, values_real, width, color='lightseagreen', edgecolor='lightcyan', linewidth=0.5,
             label='Real data')
ax2 = ax.bar(labels, values_predict, width, bottom=values_real, color='mistyrose', edgecolor='salmon',
             linewidth=0.5, label='Predict data')
for r1, r2 in zip(ax1, ax2):
    h1 = r1.get_height()
    h2 = r2.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="center", color="grey",
             fontsize=10, fontweight="bold")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="center", color="grey",
             fontsize=10, fontweight="bold")
plt.title('Bar plot for fraud and not fraud transaction \n based on risky country', weight='bold')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print(metrics.classification_report(y_test, y_pred))