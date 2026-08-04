# data cleaning
import pandas as pd

df=pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\kaggle data\pima_diabetes_dataset\diabetes_prediction_dataset.csv")

# 1.first 5 rows
print(df.head())

# 2.dataset information
print(df.info())

# 3.check for null values
print(df.isnull().sum())

# 4.remove duplicates
df.drop_duplicates(inplace=True)

# 5.fill missing values,numerical columns->median
numerical_cols=df.select_dtypes(['int64', 'float64']).columns
for col in numerical_cols:
    df[col].fillna(df[col].median(),inplace=True)

# categorical columns->mode
categorical_cols=df.select_dtypes(['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0],inplace=True)

# 6.Remove outliers
for col in categorical_cols:
    df[col]=df[col].str.strip()

# 7.convert text to lowercase
for col in categorical_cols:
    df[col]=df[col].str.lower()

# 8.check missing values again
print(df.isnull().sum())
    
# 9.save cleaned dataset
df.to_csv(r"C:\Users\LENOVO\OneDrive\Documents\kaggle data\pima_diabetes_dataset\diabetes_prediction_dataset.csv", index=False)

# 10. Display success message
print("Data cleaning completed successfully!")