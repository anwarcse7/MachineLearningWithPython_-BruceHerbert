import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

df = pd.read_csv('Titanic-Dataset.csv')

# Step 1: Diagnose missing values
print(df.isnull().sum())

# Step 2: Impute missing values using SimpleImputer (median strategy)
simple_imputer = SimpleImputer(strategy='median')
df[['Age']] = simple_imputer.fit_transform(df[['Age']])
# print(df[['Age']])

# Step 3: Mode imputation for categorical variables
simple_imputer_cat = SimpleImputer(strategy='most_frequent')
df[['Embarked']] = simple_imputer_cat.fit_transform(df[['Embarked']])

# Step 4: Add a missingness indicator
df['Cabin_was_missing'] = df['Cabin'].isnull().astype(int)
print(df['Cabin_was_missing'])

print(df.isnull().sum())