import pandas as pd
import yfinance as yf
import datetime
import numpy as np

tickers = "^N225"
#tickers = "^DJI"
start="2002-06-10"
end=datetime.date.today()
interval="1d"

#日足
data=yf.download(tickers=tickers, start=start, end=end, auto_adjust=True)

#1h足
#data=yf.download(tickers=tickers, period="max", interval=interval, auto_adjust=True)

#print(data)
#print(data.loc[:,["Close", "Volume"]])
print(data.shape)

data.to_csv("finance_"+interval+".csv")
#data.loc[:,["Close", "Volume"]].to_csv("data/finance_N225.csv")
