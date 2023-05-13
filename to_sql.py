from prices import ticker_history
import pandas as pd
import numpy as np
import pymysql
import sqlalchemy as alc
import datetime as dt

tickers = ['SPY', 'AMZN']

start = dt.datetime(2023, 5, 1, 4 - 3, 0, 0, 0)
end = dt.datetime(2023, 5, 6, 12 - 3, 0, 0, 0)

tick_hist = ticker_history(tickers)
price, price_filt = tick_hist.ohlcv(start, end)

price_filt['SPY']['Open']

# Open connection to the database
connection = pymysql.connect(
    host='stockprices.cqjvudkwowr9.us-east-1.rds.amazonaws.com',
    user='',
    password='',
    port=3306,
    database='prices_1m'
)

# Create the tables if they are not already created
cursor = connection.cursor()
sql_code = "CREATE TABLE IF NOT EXISTS time_id( "
sql_code += "time_id INT AUTO_INCREMENT NOT NULL PRIMARY KEY, "
sql_code += "time VARCHAR(255) NOT NULL);"
cursor.execute(sql_code)
connection.commit()

# 
cursor = connection.cursor()
sql_code = "SELECT * FROM time_id"
cursor.execute(sql_code)

for i in cursor.fetchall():
    print(i)

# Create the tables if they are not already created
cursor = connection.cursor()
sql_code = "INSERT INTO time_id (time) VALUES (1300)"
_ = cursor.execute(sql_code)
connection.commit()

host = "stockprices.cqjvudkwowr9.us-east-1.rds.amazonaws.com"
schema = "time_id"
port = 3306
user = "nickeisenberg"
p_w = ""
cnx = alc.create_engine(f'mysql+pymysql://{user}:{p_w}@{host}:{port}/{schema}', echo=False)

