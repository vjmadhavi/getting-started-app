import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

data = {
    'SquareFeet': [1000, 1200, 1400, 1600, 1800],
    'Price': [200000, 260000, 280000, 310000, 350000]
}
df = pd.DataFrame(data)

#feature and Targets
X = df[['SquareFeet']]
Y = df[['Price']]

#Train model
mod = LinearRegression()
mod.fit(X,Y)

#Predict price for given sqft
sft = int(input("enter a required sft: "))
pred2=mod.predict([[sft]])


plt.scatter(df['SquareFeet'], df['Price'], color='Blue', label='Actual Data')
plt.plot(df['SquareFeet'], mod.predict(X), color='Red', label='Regression')
plt.scatter(sft, pred2, color='green', s=150, edgecolors='black', label=f'Prediction: {sft} sqft')
plt.xlabel('SquareFeet')
plt.ylabel('Price')
plt.title('SquareFeet vs Price')
plt.show()