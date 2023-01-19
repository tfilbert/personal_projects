#A Program to Calcualte an Equal Weight S&P 500
#The goal of this section of the course is to create a Python script that will accept the value of your portfolio 
# and tell you how many shares of each S&P 500 constituent you should purchase to get an equal-weight version of the index fund

from turtle import clear
import numpy as np
import pandas as pd
import requests
import xlsxwriter
import math
import os
from secrets import IEX_CLOUD_API_TOKEN

file_path = os.path.dirname(__file__)

#Reading in list of S&P500 stocks
stocks = pd.read_csv(file_path + '\\sp_500_stocks.csv')

#Create a dataframe to hold stock values
my_columns = ['Ticker', 'Stock Price', 'Market Capitialization', 'Number of Shares to Buy']
final_dataframe = pd.DataFrame(columns = my_columns)

#Create groupings of 100 stocks to call api in batch (IEX Cloud maxes out at 100 symbols per batch api request)
symbol_groups =  list([stocks['Ticker'][x:x+100] for x in range(0, len(stocks['Ticker']), 100)])
symbol_strings = []
for i in range(0, len(symbol_groups)):
    symbol_strings.append(','.join(symbol_groups[i]))

#Call api in batches, and parse to get stock: ticker, price, market cap.
for symbol_string in symbol_strings:
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=quote&token={IEX_CLOUD_API_TOKEN}'
    data = requests.get(batch_api_call_url)
    try:
        data = requests.get(batch_api_call_url).json()
    except:
        print("An error occured.")
    for symbol in symbol_string.split(','):
        data_to_add = [symbol, data[symbol]['quote']['latestPrice'], data[symbol]['quote']['marketCap'], 'N/A']
        final_dataframe.loc[len(final_dataframe)] = data_to_add
           
#Get user inputted data on their portfolio size
while True:
    try:
        portfolio_size = float(input('Please enter your portfolio size:'))

    except ValueError:
        #Input error
        print("That's not a number!")
        continue
    else:
        #Process was successful
        break

position_size = math.floor(portfolio_size/len(final_dataframe.index))

#Add number of shares to buy for each stock to have a portfolio consiting of an equal weight S&P500
for i in range(0, len(final_dataframe.index)):
    final_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size/final_dataframe.loc[i, 'Stock Price'])

    
writer = pd.ExcelWriter('C:\\Python\\Recommended Trades.xlsx', engine = 'xlsxwriter')
final_dataframe.to_excel(writer, 'Recommended Trades', index = False)

#Formating xlsx output file
background_color = '#0a0a23'
font_color = '#ffffff'

string_format = writer.book.add_format(
    {
        'font_color': font_color,
        'bg_color': background_color,
        'border' : 1
    }
)

dollar_format = writer.book.add_format(
    {
        'num_format': '$0.00',
        'font_color': font_color,
        'bg_color': background_color,
        'border' : 1
    }
)

integer_format = writer.book.add_format(
    {
        'num_format': '0',
        'font_color': font_color,
        'bg_color': background_color,
        'border' : 1
    }
)

column_formats = {
    'A': ['Ticker', string_format],
    'B': ['Stock Price', dollar_format],
    'C': ['Market Capitialization', dollar_format],
    'D': ['Number of Shares to Buy', integer_format]
}

for column in column_formats.keys():
    writer.sheets['Recommended Trades'].set_column(f'{column}:{column}', 18, column_formats[column][0])
    writer.sheets['Recommended Trades'].set_column(f'{column}:{column}', 18, column_formats[column][1])
    
writer.save()





