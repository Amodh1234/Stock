from pandas_datareader import DataReader
import datetime as dti
import seaborn as sb
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


a=input("Enter any Company Name    ")
no_days=int(input("Enter Number of Days      "))
def Stock_Prices(a,no_days):
    dt=DataReader(a,'tiingo',start='1/1/2000',end=dti.datetime.now(),api_key='5820dd91615951808c4e36626e5c1cfa3bf139c0')
    dt.reset_index(inplace=True)
    dt.set_index("date",inplace=True)
    dt=dt[['adjClose','adjHigh', 'adjLow', 'adjOpen', 'adjVolume']] 
    algo=LinearRegression()
    ip=dt.drop('adjClose',axis=1)
    dt['adjClose'].shift(-no_days)
    out=dt['adjClose'].shift(-no_days).dropna()
    train_inp=ip[:-no_days]
    prd_in=ip[-no_days:]
    algo.fit(train_inp,out)
    y=algo.predict(prd_in)
    x=mean_squared_error(out,algo.predict(train_inp))
    return y-x,y+x

print(Stock_Prices(a,no_days))