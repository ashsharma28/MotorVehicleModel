# read two data: policy and claims from /content/drive/MyDrive/Colab Notebooks/Allianz Data/Motor vehicle insurance data.csv and  /content/drive/MyDrive/Colab Notebooks/Allianz Data/sample type claim.csv
import pandas as pd
# Increase pandas option max_for columns
pd.set_option('display.max_columns', None)
df_policies = pd.read_csv('data/Motor vehicle insurance data.csv', sep=";")
df_claims  = pd.read_csv('data/sample type claim.csv' , sep = ';')
# Convert dates to datetime objects
date_cols = ['Date_start_contract', 'Date_last_renewal', 'Date_next_renewal', 'Date_birth', 'Date_driving_licence']
for col in date_cols:
    df_policies[col] = pd.to_datetime(df_policies[col], format='%d/%m/%Y', errors='coerce')


# Call the Agent code
from Agent import app
config = {"configurable": {"thread_id": "1"}}
turn_1 = app.invoke(
    input={"input": "Hello, my name is John. I am 30 years old and I have a car with 4 doors."},
    config=config
)