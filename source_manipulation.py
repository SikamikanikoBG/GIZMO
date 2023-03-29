import pandas as pd
import numpy as np


df = pd.read_parquet("./input_data/input_source_original.parquet")

# Remove duplicates
print(f"Data set before deduplication: {len(df)}")
df = df.drop_duplicates(subset=['FKCUSTOMER', 'ObservationDate'], keep='last')
print(f"Data set after deduplication: {len(df)}")

# Create flags and additional columns
df["ObservationDate_str"] = df["ObservationDate"].astype(str)
df['criterion_NbDirectApp_2M'] = np.where((df['NbDirectApp_2M'] > 0), 1, 0)
df['criterion_NbDirectLoan_2M'] = np.where((df['NbDirectLoan_2M'] > 0), 1, 0)
df['criterion_NbConsoApp_2M'] = np.where((df['NbConsoApp_2M'] > 0), 1, 0)
df['criterion_NbConsoLoan_2M'] = np.where((df['NbConsoLoan_2M'] > 0), 1, 0)
df['criterion_NbPlusApp_2M'] = np.where((df['NbPlusApp_2M'] > 0), 1, 0)
df['criterion_NbPlusLoan_2M'] = np.where((df['NbPlusLoan_2M'] > 0), 1, 0)
df['criterion_NbretailApp_1M'] = np.where((df['NbretailApp_1M'] > 0), 1, 0)
df['criterion_NbRetailLoan_1M'] = np.where((df['NbRetailLoan_1M'] > 0), 1, 0)
df['criterion_NbVATApp_1M'] = np.where((df['NbVATApp_1M'] > 0), 1, 0)
df['criterion_NbVATLoan_1M'] = np.where((df['NbVATLoan_1M'] > 0), 1, 0)

df['flag_active_customer'] = np.where(((df['Direct_Active_CRD'] > 0)
                                      | (df['Retail_Active_CRD'] > 0)
                                      | (df['Card_Active_CRD'] > 0)), 1, 0)

df['flag_direct_active_customer'] = np.where(((df['Direct_Active_CRD'] > 0)
                                      | (df['Card_Active_CRD'] > 0)), 1, 0)

# ---------------------------------------------------------- cut into subsets
# Then split active into sub-segments
df_org = df.copy() # keep original for checks
df = df[df["flag_active_customer"] == 1].copy()

df_direct_active = df[df["flag_direct_active_customer"] == 1].copy()
df_retail_active = df[~df["FKCUSTOMER"].isin(df_direct_active["FKCUSTOMER"])].copy()

# df_new_12M_active = df[df["Direct_Active"] > 0].copy() #todo: add acquisition date + months since acquisition

# Check
print(f"Total active {len(df)}, Direct active {len(df_direct_active)}, Retail active {len(df_retail_active)}, "
      f"sum {len(df_direct_active)+len(df_retail_active)}")

# Split the in-active
df_inactive = df_org[~df_org["FKCUSTOMER"].isin(df_direct_active["FKCUSTOMER"])].copy()
df_inactive = df_inactive[~df_inactive["FKCUSTOMER"].isin(df_retail_active["FKCUSTOMER"])].copy()
print(f"In-active {len(df_inactive)}")
# ---------------------------------------------------------- cut into subsets

# store files
df_direct_active.to_parquet("./input_data/IDA_PL_Active_Direct/input_source.parquet")
# df_new_12M_active.to_parquet("./input_data/IDA_Active_New_12M/input_source.parquet")
df_inactive.to_parquet("./input_data/IDA_InActiveAndProspects/input_source.parquet")
df_retail_active.to_parquet("./input_data/IDA_Retail_Active/input_source.parquet")

# 100
df_direct_active.sample(250).to_csv("./input_data/df_direct_active.csv", index=False)
df_inactive.sample(250).to_csv("./input_data/df_inactive.csv", index=False)
df_retail_active.sample(250).to_csv("./input_data/df_retail_active.csv", index=False)

