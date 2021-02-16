import pandas as pd

cc_df = None

if cc_df is None:
    try:
        cc_df = pd.read_csv('creditcardcsvpresent.csv')
    except:
        print("an error occurred")
