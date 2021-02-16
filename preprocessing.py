from sklearn import preprocessing
from read_csv import *

desired_width = 280
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 12)

# drop transaction date (all NAn):

cc_df = cc_df.drop('Transaction date', 1)

cc_df = cc_df.dropna()

# changing categorical column to numeric:
encoders = {"Is declined": preprocessing.LabelEncoder(), "isForeignTransaction": preprocessing.LabelEncoder(),
            "isHighRiskCountry": preprocessing.LabelEncoder(), "isFradulent": preprocessing.LabelEncoder()}

cc_df['Is declined'] = encoders['Is declined'].fit_transform(cc_df['Is declined'])
cc_df['isHighRiskCountry'] = encoders['isHighRiskCountry'].fit_transform(cc_df['isHighRiskCountry'])
cc_df['isFradulent'] = encoders['isFradulent'].fit_transform(cc_df['isFradulent'])
cc_df['isForeignTransaction'] = encoders['isForeignTransaction'].fit_transform(cc_df['isForeignTransaction'])

#adding column
cc_df = cc_df.rename(columns={'Average Amount/transaction/day' : 'AverageAmount_transaction_day'})
cc_df = cc_df.rename(columns={'isFradulent' : 'isFraudulent'})
cc_df['amnt_avg_ratio'] = cc_df.Transaction_amount / cc_df.AverageAmount_transaction_day

#dropping id for the KMEANS
cc_df_kmeans = cc_df.copy()
cc_df_kmeans = cc_df_kmeans.drop('Merchant_id', 1)


normalized_data = pd.DataFrame()
# Normalize the data:
columns = ['AverageAmount_transaction_day', 'Transaction_amount', 'Total Number of declines/day', 'Daily_chargeback_avg_amt',
           '6_month_avg_chbk_amt', '6-month_chbk_freq','amnt_avg_ratio']
for col in columns:
    normalized_data[col] = cc_df_kmeans[col].apply(
        lambda x: (x - cc_df_kmeans[col].mean() / cc_df_kmeans[col].std()))

# add binery
normalized_data["Is declined"] = cc_df_kmeans["Is declined"]
normalized_data["isForeignTransaction"] = cc_df_kmeans["isForeignTransaction"]
normalized_data["isHighRiskCountry"] = cc_df_kmeans["isHighRiskCountry"]
normalized_data["isFraudulent"] = cc_df_kmeans["isFraudulent"]


