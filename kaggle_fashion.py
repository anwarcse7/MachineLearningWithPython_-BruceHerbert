import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset/fashion-mnist_train.csv')
# print(df.head())
# print('Missing values per column:')
# print(df.isnull().sum())
# print(df.describe())

#   Plot histograms for all numeric columns
# fig, axes = plt.subplots(3, 3, figsize=(12, 10))
# axes = axes.flatten()

# for i, col in enumerate(df.columns[:-1]):
#     df[col].hist(bins=50, ax=axes[i])
#     axes[i].set_title(col)
#     plt.tight_layout()
#     plt.show()
#Create a correlation heatmap with seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()