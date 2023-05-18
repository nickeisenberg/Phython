from prices import ticker_history
import pandas as pd
import numpy as np
import pymysql
import sqlalchemy as alc
import datetime as dt

# Open connection to the database
connection = pymysql.connect(
    host='stockprices.cqjvudkwowr9.us-east-1.rds.amazonaws.com',
    user='nickeisenberg',
    password='',
    port=3306,
    database='prices_1m'
)
#--------------------------------------------------

# Tickers to get prices
tickers = ['SPY', 'AMZN', 'GOOG', 'AAPL', 'QQQ']
#--------------------------------------------------

# Create the ticker table if not exists
cursor = connection.cursor()
sql_code = "CREATE TABLE IF NOT EXISTS ticker_id( "
sql_code += "ticker_id INT AUTO_INCREMENT NOT NULL PRIMARY KEY, "
sql_code += "ticker VARCHAR(255) NOT NULL);"
cursor.execute(sql_code)
connection.commit()
#--------------------------------------------------

# Add the tickers to the table if they arent already there
for t in tickers:
    cursor = connection.cursor()
    sql_code = "INSERT INTO ticker_id (ticker) "
    sql_code += f"SELECT '{t}' FROM DUAL "
    sql_code += "WHERE NOT EXISTS (SELECT * FROM  ticker_id "
    sql_code += f"WHERE ticker='{t}' LIMIT 1);"
    cursor.execute(sql_code)
    connection.commit()
#--------------------------------------------------

# Create the tables if they are not already created
cursor = connection.cursor()
sql_code = "CREATE TABLE IF NOT EXISTS time_id( "
sql_code += "time_id INT AUTO_INCREMENT NOT NULL PRIMARY KEY, "
sql_code += "time VARCHAR(255) NOT NULL);"
cursor.execute(sql_code)
connection.commit()
#--------------------------------------------------

start = dt.datetime(2023, 5, 1, 4 - 3, 0, 0, 0)
end = dt.datetime(2023, 5, 6, 12 - 3, 0, 0, 0)

tick_hist = ticker_history(tickers)
price, price_filt = tick_hist.ohlcv(start, end)

price_filt['SPY']['Open']

# Create the time ids
# This wy is very slow, need to find a better way
# for t in price_filt['SPY']['Open'].index.values:
#     cursor = connection.cursor()
#     sql_code = "INSERT INTO time_id (time) "
#     sql_code += f"SELECT '{t}' FROM DUAL "
#     sql_code += "WHERE NOT EXISTS (SELECT * FROM  time_id "
#     sql_code += f"WHERE time='{t}' LIMIT 1);"
#     cursor.execute(sql_code)
#     connection.commit()

# host = "stockprices.cqjvudkwowr9.us-east-1.rds.amazonaws.com"
# schema = "time_id"
# port = 3306
# user = "nickeisenberg"
# p_w = ""
# cnx = alc.create_engine(f'mysql+pymysql://{user}:{p_w}@{host}:{port}/{schema}', echo=False)

