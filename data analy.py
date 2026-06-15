import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",
    database="air_pollution"
)

query = "SELECT * FROM pollution_data"
df = pd.read_sql(query, conn)

print(df.head())

print("Average AQI:", df['aqi'].mean())
print("Max AQI:", df['aqi'].max())

plt.figure()
df.groupby('city')['aqi'].mean().plot(kind='bar')
plt.title("Average AQI by City")
plt.xlabel("City")
plt.ylabel("AQI")
plt.show()

X = df[['pm25', 'pm10', 'no2', 'co']]
y = df['aqi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)


new_data = [[100, 150, 50, 1.2]]
predicted_aqi = model.predict(new_data)

print("Predicted AQI:", predicted_aqi[0])