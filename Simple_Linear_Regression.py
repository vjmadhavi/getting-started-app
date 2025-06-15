import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
cost_per_sft = 4600
data = {
    'SquareFeet': [1500, 1600, 1700, 1800, 1900, 2000, 2100],
    'Price': [300000, 320000, 340000, 360000, 380000, 400000, 420000]
}
df = pd.DataFrame(data)

#features and targets
X = df[['SquareFeet']]
Y = df[['Price']]

#Train model
model = LinearRegression()
model.fit(X,Y)

#predict
sft = [3500]
predict1 = model.predict([sft])
print(f"Predicted price for {sft} sqft = ${predict1[0][0]:,.2f}")

#plot 
plt.scatter(df['SquareFeet'], df['Price'], color='blue')
plt.plot(df['SquareFeet'], model.predict(X), color='red')
plt.xlabel('SquareFeet')
plt.ylabel('Price')
plt.show()