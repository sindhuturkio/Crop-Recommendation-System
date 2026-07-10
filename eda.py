import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/raw/Crop_recommendation.csv")

# Display first 5 rows
print(df.head())

# Check dataset shape (rows, columns)
print("\nShape of dataset:", df.shape)

# Check column info and data types
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Check how many samples per crop
print("\nCrop distribution:\n", df['label'].value_counts())

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.drop('label', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Boxplots to check outliers
plt.figure(figsize=(12,8))
for i, col in enumerate(['N','P','K','temperature','humidity','ph','rainfall']):
    plt.subplot(3,3,i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.savefig("boxplots.png")
plt.show()

# Check pH range for each crop
print("\nAverage pH per crop:\n", df.groupby('label')['ph'].mean().sort_values())