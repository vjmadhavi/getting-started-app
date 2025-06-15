import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

DATA = pd.read_csv('/Users/venu/getting-started-app/LR/personality_dataset.csv')
df = pd.DataFrame(DATA)

# features and targets
X = 