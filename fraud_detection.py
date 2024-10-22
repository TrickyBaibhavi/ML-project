# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Step 2: Load Dataset
df = pd.read_csv('Admission_Predict.csv')
print(df.head(2))

# Step 3: Data Exploration
print(df.info())
print(df.describe())

# Step 4: Feature Selection
x = df[['GRE Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
y = df['Chance of Admit ']

# Step 5: Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9598)

# Step 6: Model Training
model = LinearRegression()
model.fit(x_train, y_train)

# Step 7: Prediction
y_pred = model.predict(x_test)
print(y_pred)

# Step 8: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAE: {mae}, MAPE: {mape}')

# Step 9: Accuracy Calculation
accuracy = (1 - mape)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
