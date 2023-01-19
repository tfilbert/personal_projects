#A momentum based trading stategy
#Momentum investing is investing in stocks that have increased in price the most

import os
import numpy as np
import pandas as pd
import requests
import math
from scipy import stats
import xlsxwriter
from secrets import IEX_CLOUD_API_TOKEN

#Read list of stocks in S&P500 from .csv file
file_path = os.path.dirname(__file__)
stocks = pd.read_csv(os.path.join(file_path, 'sp_500_stocks.csv'))

#Creating final dataframe to hold resultant data
my_columns = ['Ticker', 'Price', 'One-Year Price Return', 'Number of Shares to Buy']
final_dataframe = pd.DataFrame(columns=my_columns)


# Function sourced from 
# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]   

#Split stocks into groups of 100 for batch api calls        
symbol_groups = list(chunks(stocks['Ticker'], 100))
symbol_strings = []
for i in range(0, len(symbol_groups)):
    symbol_strings.append(','.join(symbol_groups[i]))


#API requests to get data from IEX cloud
for symbol_string in symbol_strings:
    batch_api_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=stats,quote&token={IEX_CLOUD_API_TOKEN}'
    try:
        my_data = requests.get(batch_api_url).json()
    except:
        print('Error with request')
    
    #Adding invidivual stock info's price and 1 year change (percent)
    for symbol in symbol_string.split(','):
        price = my_data[symbol]['quote']['latestPrice']
        year_one_change = my_data[symbol]['stats']['year1ChangePercent']
        final_dataframe.loc[len(final_dataframe.index)] = [symbol, price, year_one_change, 'N/A']
        

#Sort in descending order by one year price return, and only keep top 50
final_dataframe = final_dataframe.sort_values(by=['One-Year Price Return'], ascending=False).reset_index(drop=True)
final_dataframe = final_dataframe.drop(final_dataframe.index[50:len(final_dataframe)])


#A function to get and return an entered value for a portfolio size
def portfolio_input():
    while(True):
        print("Enter the size of your portfolio:")
        portfolio_size = input()

        try:
            portfolio_size = float(portfolio_size)
            return portfolio_size
        except:
            print("Please enter a number")


#Calcualte and add number of shares to buy for each stock (integer values for num shares)
position_size = portfolio_input() / 50
for index, row in final_dataframe.iterrows():
    price = row['Price']
    num_shares = position_size // price
    final_dataframe.at[index,'Number of Shares to Buy'] = num_shares

#Output dataframe to excel
writer = pd.ExcelWriter('C:/Python/Momentum_Trades_SP500.xlsx', engine ='xlsxwriter')

final_dataframe.to_excel(writer, sheet_name='Trades Report', index = False)

workbook = writer.book
worksheet = writer.sheets['Trades Report']

#Add cell formats
dollar_format = workbook.add_format({'num_format': '$#,##0.00'})
percent_format = workbook.add_format({'num_format': '0%'})
integer_format = workbook.add_format({'num_format': '#,##0'})

#Set column width and format
worksheet.set_column('B:B', 10, dollar_format)
worksheet.set_column('C:C', 22, percent_format)
worksheet.set_column('D:D', 24, integer_format)

writer.save()
