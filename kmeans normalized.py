from preprocessing import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

sum_squared = []
silhouette = []
for i in range(2, 12):
    kmeans = KMeans(n_clusters=i,
                    init='k-means++')
    kmeans.fit(normalized_data)
    sum_squared.append(kmeans.inertia_)
    silhouette.append(silhouette_score(normalized_data, kmeans.labels_))

#Sum of Squred Error
plt.plot(range(2, 12), sum_squared, '#00cc00')
plt.title(r'Sum of Squred Error ${R_2}$', fontsize=15, fontweight="bold")
plt.xlabel('No. of Clusters')
plt.ylabel(r'${R_2}$')
plt.show()

#Silhouette
plt.plot(range(2, 12), silhouette, color='#8c1aff')
plt.title('Silhouette', fontsize=15,fontweight="bold")
plt.xlabel('No. of Clusters')
plt.ylabel('Silhouette')
plt.show()

kmeans = KMeans(n_clusters=2, init='k-means++')
kmeans.fit(normalized_data)

print(silhouette_score(normalized_data, kmeans.labels_))

normalized_data["label"] = pd.Series(kmeans.labels_)
clusters = normalized_data.groupby("label")

cc_df_kmeans["label"] = normalized_data["label"]
print(cc_df_kmeans.loc[cc_df_kmeans['label'] == 0].describe().loc[['count','mean','std','max','min']])
print(cc_df_kmeans.loc[cc_df_kmeans['label'] == 1].describe().loc[['count','mean','std','max','min']])

#Transaction Amount Divided By Labels And Fraudulen
sns.swarmplot(cc_df_kmeans.label, cc_df_kmeans.Transaction_amount, hue=cc_df_kmeans.isFraudulent, color='#e6b800')
plt.title('Transaction Amount Divided By Labels And Fraudulent', fontsize=14, fontweight="bold")
plt.show()


#Avg Amount Ratio By High Risk Country Divided To Labels And Fraudulent
ax = sns.catplot(x="isHighRiskCountry", y="amnt_avg_ratio",
                hue="label", col="isFraudulent",
                data=cc_df_kmeans, kind="swarm", palette='Set1',
                height=4, aspect=.7)
ax.fig.suptitle('Avg Amount Ratio By High Risk Country Divided To Labels And Fraudulent', fontsize=14, fontweight="bold")
plt.show()

#Transaction Amount Distribution After Clustering
fig,ax = plt.subplots(1,1,figsize=(9,5))
sns.kdeplot(cc_df_kmeans[cc_df_kmeans['label'] == 0]['Transaction_amount'],ax = ax, color='#0099cc')
sns.kdeplot(cc_df_kmeans[cc_df_kmeans['label'] == 1]['Transaction_amount'],ax=ax, color='#ff00ff')
sns.set_style("darkgrid")
plt.legend(['label == 0','label == 1'])
plt.title('Transaction Amount Distribution After Clustering', fontsize=14, fontweight="bold")
plt.xlabel('Transaction Amount')
plt.ylabel('Count')
plt.show()