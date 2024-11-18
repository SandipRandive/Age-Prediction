# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("C:/Users/malik/OneDrive/Desktop/DS Python/Animal Dataset.csv")

# Show basic info and the first few rows
print(df.info())
print(df.head())

# 4. Visualize Data

# Box Plot for features
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Height (cm)', 'Weight (kg)', 'Age (years)']])
plt.title('Boxplot of Features')
plt.show()

# Histogram for Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Age (years)'], kde=True, color='blue')
plt.title('Distribution of Species Age')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot of Height vs Age
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Height (cm)', y='Age (years)')
plt.title('Scatter Plot: Height vs Age')
plt.xlabel('Height (cm)')
plt.ylabel('Age (years)')
plt.show()

# 5. Pearson Correlation

# Filter only numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=[float, int])

# Handle missing values in numeric columns
# Option 1: Drop rows with NaN values in numeric columns (if too many NaNs)
numeric_df = numeric_df.dropna()

# Option 2: Fill NaN values with the mean of the column
# numeric_df = numeric_df.fillna(numeric_df.mean())

# Check if numeric_df is empty after handling missing data
if numeric_df.empty:
    print("Error: No numeric data available for correlation.")
else:
    # Calculate the Pearson correlation between numeric features
    correlation_matrix = numeric_df.corr()
    print("Correlation Matrix:")
    print(correlation_matrix)

    # Plot the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Heatmap')
    plt.show()

# 6. Identify Dependent and Independent Features

# Dependent variable is Age (years), independent variables are features like Height, Weight
X = df[['Height (cm)', 'Weight (kg)', 'Age (years)']]  # Independent features
y = df['Age (years)']  # Dependent feature (target)

# 7. Split Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Evaluation:")
print(f'Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}')
print(f'Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}')
print(f'R-squared: {r2_score(y_test, y_pred)}')

# Example of predicted vs actual values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.head())
