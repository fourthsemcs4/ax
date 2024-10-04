import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_set=pd.read_csv(r"C:\Users\TRISHA MAITHRI\Documents\Dataset\Dataset\Salary_Data.csv")
x=data_set.iloc[:,:-1].values
y=data_set.iloc[:,1].values
from sklearn.linear_model import LinearRegression
lin_regs=LinearRegression()
lin_regs.fit(x,y)
from sklearn.preprocessing import PolynomialFeatures
poly_regs=PolynomialFeatures(degree=2)
x_regs=poly_regs.fit_transform(x)
lin_regs_2=LinearRegression()
lin_regs_2.fit(x_regs,y)
plt.scatter(x,y,color="red")
plt.plot(x,lin_regs.predict(x),color="blue")
plt.title("bluff detection model(linear regression)")
plt.xlabel("postion level")
plt.ylabel("salary")
plt.show()
plt.scatter(x,y,color="blue")
plt.plot(x,lin_regs_2.predict(poly_regs.fit_transform(x)),color="red")
plt.title("bluff detection model(polynomial regression)")
plt.xlabel("postion level")
plt.ylabel("salary")
plt.show()

