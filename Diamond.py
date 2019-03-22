import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#Importing data
data = pd.read_csv("diamonds.csv")

p=data["price"]
carat = data["carat"]

#Removing outliers
Q1 = p.quantile(.25)
Q3 = p.quantile(.75)
q1 = Q1-1.5*(Q3-Q1)
q3 = Q3+1.5*(Q3-Q1)
df = data[p.between(q1, q3)]
plt.plot(df["price"].values, 'o')
plt.show()

plt.boxplot(p)
plt.show()

#Histogram of price
plt.title("Graph 2 - Histogram of price")
plt.ylabel("price")
plt.xlabel("count")
plt.hist(p)
plt.show()

#Histogram of price log(10)
plt.title("Graph 3 - Histogram of price log(10)")
plt.ylabel("log$_{10}$price")
plt.xlabel("count")
plt.hist(np.log(p))
plt.show()

#Scatter plot of price vs carat
plt.title("Graph 4 - Scatter plot of price vs carat")
plt.xlabel("carat")
plt.ylabel("price")
plt.scatter(carat, p)
plt.show()

#Scatter plot of price log(10) vs carat 1/2
x = np.power(carat,1/2).values[:,np.newaxis]
y = np.log10(p).values
model = LinearRegression()
model.fit(x, y)
plt.scatter(x, y,color='g')
plt.plot(x, model.predict(x),color='k')
plt.title("Graph 5 - Scatter plot of log$_{10}$price vs $\sqrt{carat}$")
plt.ylabel("log$_{10}$price")
plt.xlabel("$\sqrt{carat}$")
plt.show()

#Scatter plot of price log(10) vs carat 1/3
x = np.power(carat,1/3).values[:,np.newaxis]
y = np.log10(p).values
model = LinearRegression()
model.fit(x, y)
plt.scatter(x, y,color='g')
plt.plot(x, model.predict(x),color='k')
plt.title("Graph 6 - Scatter plot of log$_{10}$price vs $\sqrt[3]{carat}$")
plt.ylabel("log$_{10}$price")
plt.xlabel("$\sqrt[3]{carat}$")
plt.show()

#Scatter plot of price log(10) vs carat 1/4
x = np.power(carat,1/4).values[:,np.newaxis]
y = np.log10(p).values
model = LinearRegression()
model.fit(x, y)
plt.scatter(x, y,color='g')
plt.plot(x, model.predict(x),color='k')
plt.title("Graph 7 - Scatter plot of log$_{10}$price vs $\sqrt[4]{carat}$")
plt.ylabel("log$_{10}$price")
plt.xlabel("$\sqrt[4]{carat}$")
plt.show()

#Scatter plot of price log(10) vs carat 1/5
x = np.power(carat,1/5).values[:,np.newaxis]
y = np.log10(p).values
model = LinearRegression()
model.fit(x, y)
plt.scatter(x, y,color='g')
plt.plot(x, model.predict(x),color='k')
plt.title("Graph 8 - Scatter plot of log$_{10}$price vs $\sqrt[5]{carat}$")
plt.ylabel("log$_{10}$price")
plt.xlabel("$\sqrt[5]{carat}$")
plt.show()

#Scatter plot of price log(10) vs carat log(10)
x = np.log(carat).values[:,np.newaxis]
y = np.log(p).values
model = LinearRegression()
model.fit(x, y)
plt.scatter(x, y,color='g')
plt.plot(x, model.predict(x),color='k')
plt.title("Graph 9 - Scatter plot of log$_{10}$price vs log$_{10}$carat")
plt.ylabel("log$_{10}$price")
plt.xlabel("log$_{10}$carat")
plt.show()

#Scatter plot of price log(10) vs carat 2
x = np.power(carat,2).values[:,np.newaxis]
y = np.log10(p).values
model = LinearRegression()
model.fit(x, y)
plt.scatter(x, y,color='g')
plt.plot(x, model.predict(x),color='k')
plt.title("Graph 10 - Scatter plot of log$_{10}$price vs carat$^2$")
plt.ylabel("log$_{10}$price")
plt.xlabel("carat$^2$")
plt.show()

#Scatter plot of price log(10) vs carat 3
x = np.power(carat,3).values[:,np.newaxis]
y = np.log(p).values
model = LinearRegression()
model.fit(x, y)
plt.scatter(x, y,color='g')
plt.plot(x, model.predict(x),color='k')
plt.title('Graph 11 - Scatter plot of log$_{10}$price vs carat$^3$')
plt.ylabel("log$_{10}$price")
plt.xlabel("carat$^3$")
plt.show()

#Scatter plot of price log(10) vs carat 1/4 grouped by clarity
groups = data.groupby('clarity')
fig, ax = plt.subplots()
for name, group in groups:    
    ax.plot(np.power(group.carat,1/4), np.log10(group.price), marker='o', linestyle='', ms=6, label=name)
ax.legend() 
plt.title("Graph 12 - Scatter plot of log$_{10}$price vs $\sqrt[4]{carat}$ grouped by clarity")  
plt.ylabel("log$_{10}$price")
plt.xlabel("$\sqrt[4]{carat}$")

#Scatter plot of price log(10) vs carat 1/4 grouped by cut
groups = data.groupby('cut')
fig, ax = plt.subplots()
for name, group in groups:    
    ax.plot(np.power(group.carat,1/4), np.log10(group.price), marker='o', linestyle='', ms=6, label=name)
ax.legend()    
plt.title("Graph 13 - Scatter plot of log$_{10}$price vs $\sqrt[4]{carat}$ grouped by cut")  
plt.ylabel("log$_{10}$price")
plt.xlabel("$\sqrt[4]{carat}$") 
plt.show()

#Scatter plot of price log(10) vs carat 1/4 grouped by color
groups = data.groupby('color')
fig, ax = plt.subplots()
for name, group in groups:    
    ax.plot(np.power(group.carat,1/4), np.log10(group.price), marker='o', linestyle='', ms=6, label=name)
ax.legend()
plt.title("Graph 14 - Scatter plot of log$_{10}$price vs $\sqrt[4]{carat}$ grouped by color")    
plt.ylabel("log$_{10}$price")
plt.xlabel("$\sqrt[4]{carat}$")
plt.show()