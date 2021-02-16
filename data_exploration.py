
from preprocessing import *
import seaborn as sns
import matplotlib.pyplot as plt

print(cc_df.shape[0], cc_df.shape[1])
print(cc_df.columns)
print(cc_df.describe())



#
print(len(cc_df[(cc_df["isFradulent"] == 1) & (cc_df["Is declined"] == 1)]) / len(
    cc_df["Is declined"][cc_df["Is declined"] == 1]))
print(len(cc_df[(cc_df["isFradulent"] == 1) & (cc_df["amnt_avg_ratio"] > 30)]) / len(
    cc_df["isFradulent"][cc_df["isFradulent"] == 1]))



# Bar plot for fraud and not fraud transaction below and above 30 transaction amount/average amount ratio
labels = ['Fraud', 'Not Fraud']
below_30_not_fraud = cc_df[(cc_df['amnt_avg_ratio'] <= 30) & (cc_df['isFraudulent'] == 0)]['isFraudulent'].count()
below_30_fraud = cc_df[(cc_df['amnt_avg_ratio'] <= 30) & (cc_df['isFraudulent'] == 1)]['isFraudulent'].count()
values_below_30 = [below_30_fraud, below_30_not_fraud]
above_30_not_fraud = cc_df[(cc_df['amnt_avg_ratio'] > 30) & (cc_df['isFraudulent'] == 0)]['isFraudulent'].count()
above_30_fraud = cc_df[(cc_df['amnt_avg_ratio'] > 30) & (cc_df['isFraudulent'] == 1)]['isFraudulent'].count()
values_above_30 = [above_30_fraud, above_30_not_fraud]
width = 0.5
fig = plt.figure(facecolor="white")
ax = fig.add_subplot(1, 1, 1)
ax1 = ax.bar(labels, values_below_30, width, color='#6282EA', edgecolor='black', linewidth=0.5,
             label='Below 30')
ax2 = ax.bar(labels, values_above_30, width, bottom=values_below_30, color='#CDD9EC', edgecolor='black',
             linewidth=0.5, label='Above 30')
for r1, r2 in zip(ax1, ax2):
    h1 = r1.get_height()
    h2 = r2.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="center", color="black",
             fontsize=8, fontweight="bold")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="center", color="black",
             fontsize=8, fontweight="bold")
plt.title('Bar plot for fraud and not fraud transaction below and above 30 transaction\n amount/average amount ratio ', weight='bold')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Fraudulent orders
plt.pie(cc_df['isFraudulent'].value_counts(),
        colors=['#b8a9c9', '#622569'],
        explode=[0, 0.1],
        labels=['Not Fraud', 'Fraud'],
        shadow=True,
        autopct='%.2f%%')
plt.title('Fraudulent orders', fontsize=20, fontweight="bold")
plt.axis('off')
plt.legend()
plt.show()

#Foreign Transaction Ratio Of Fraudulent Transactions
plt.pie(cc_df['isForeignTransaction'][cc_df['isFraudulent'] == 1].value_counts(),
        colors=['#bf73a4', '#e8bad8'],
        explode=[0, 0.1],
        labels=['Foreign Transaction', 'Not Foreign Transaction'],
        shadow=True,
        autopct='%.2f%%')
plt.title('Foreign Transaction Ratio Of Fraudulent Transactions', fontsize=20, fontweight="bold")
plt.axis('off')
plt.legend()
plt.show()

#Correlation Matrix Of Selected Columns
plt.figure(figsize=(8,8))
sns.set(font_scale=1)
heatmap = sns.heatmap(
cc_df[["Transaction_amount","amnt_avg_ratio" , "Is declined", "isForeignTransaction", "isHighRiskCountry", "isFraudulent"]].corr(),
    annot=True, cmap= 'coolwarm',
    linewidths=0.8,
    vmax=1,
    annot_kws={"size": 10})
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=15)
plt.title('Correlation Matrix Of Selected Columns', fontsize=20, fontweight="bold")
plt.show()

#Transaction amount box plot
sns.set_style("whitegrid")
sns.boxplot(x='isFraudulent', y='Transaction_amount', hue="Is declined", data=cc_df, palette='Reds')
plt.title('Transaction amount box plot', size =20, fontweight="bold")
plt.show()

#Distribution of Transaction Amount
sns.set(style = 'darkgrid')
sns.distplot(cc_df['Transaction_amount'], color='#0a69ad')
plt.title('Distribution of Transaction Amount', fontsize = 20, fontweight="bold")
plt.xlabel('Amounts')
plt.ylabel('Count')
plt.show()

#Transaction Amount Distribution
fig,ax = plt.subplots(1,1,figsize=(9,5))
sns.kdeplot(cc_df[cc_df['isFraudulent']==0]['Transaction_amount'],ax = ax, color='#4040a1')
sns.kdeplot(cc_df[cc_df['isFraudulent']==1]['Transaction_amount'],ax=ax, color= '#ad220a')
sns.set_style("darkgrid")
plt.legend(['is Fraudulent=0','is Fraudulent=1'])
plt.title('Transaction Amount Distribution', fontsize = 20, fontweight="bold")
plt.xlabel('Transaction Amount')
plt.ylabel('Count')
plt.show()

