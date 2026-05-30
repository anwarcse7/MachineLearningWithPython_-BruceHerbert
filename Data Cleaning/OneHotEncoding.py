from sklearn.preprocessing import OneHotEncoder
import pandas as pd 

df = pd.read_csv('Titanic-Dataset.csv')

# Using pandas(quickest for EDA)
df_encoded = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# print(df_encoded.head())

# Using Scikit-Learn (use inside pipelines) 
ohe = OneHotEncoder(drop='first', sparse_output=False) 
embarked_encoded = ohe.fit_transform(df[['Embarked']])