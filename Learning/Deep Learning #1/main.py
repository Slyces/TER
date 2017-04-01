import panda as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# On récupère les données grâce à panda
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

# Modèle de régression linéaire
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# visualisaton des résultats
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
