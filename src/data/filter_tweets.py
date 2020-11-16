#%%
import pandas as pd

df = pd.read_csv('data/raw/full_dataset_clean.tsv', sep = "\t")

#%%
df.columns
# %%
df_eng = df[df.lang == 'en']
# %%
del df
# %%
df_eng['date'] = pd.to_datetime(df_eng['date'], format = '%Y-%m-%d')
# %%

df_eng0316 = df_eng[df_eng.date == "2020-03-16"]
# %%
df_eng0316.to_csv('data/interim/Tweet200316.tsv')
# %%
