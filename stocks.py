from tempfile import tempdir
from tkinter import Y
from turtle import clear
import requests
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math
import os
plt.style.use('_mpl-gallery')

def main():
    current_dir = os.path.dirname(__file__)
    output_dir = os.path.join(current_dir, 'Tables\\')
    now = datetime.now().strftime("%m-%d-%Y %H.%M.%S")
    api_key =  "HR0GMGCA4BFBDLZVEUAWYJINZS4AGORW"
    #Available markets NASDAQ: $COMPX, DOW: $DJI, SPY: $SPX.X
    market = "$COMPX"
    try:
        url = "https://api.tdameritrade.com/v1/marketdata/"+ market + "/movers"
        r = requests.get(url, params={'apikey' : api_key})
        #print("HTML:\n", r.text)
        print("Success!")
        print('\n')
    except:
        print("Fetching data failed")
        print('\n')

    stocks = r.text.split("},{")
    #Remove "[{" from first entry
    stocks[0] = stocks[0][2:len(stocks[0])]
    #Remove "}]" from last entry
    stocks[len(stocks)-1] = stocks[len(stocks)-1][0:len(stocks[len(stocks)-1])-3]
    
    stocks_descriptions = []
    for x in stocks:
        #Break up values for each stock
        stock_description = x.split(",\"")

        #Remove equation quotation marks
        for y in range (0,len(stock_description)):
            stock_description[y] = stock_description[y].replace('"','')

        #Add all cleaned up values of stocks (each having their own list) to a list
        stocks_descriptions.append(stock_description)
        
        


    lookup_symbol = "symbol"
    stocks_values = {}
    #Create a dictionary and store all symbols
    for x in stocks_descriptions:
        #Swap symbol and change to have symboles the start of each list
        tmp = x[0]
        x[0] = x[4]
        x[4] = tmp

        #Add all symbols as keys to dictionaries
        symbol = x[0].split(":")[1]
        stocks_values[symbol] = {}

        #add all data entires under each stock's value as a dictionary in form: data name, value
        for y in range (1, len(x)):
            data = x[y].split(":")
            stocks_values[symbol][data[0]] = data[1]

    #Define figure and axes
    fig, ax = plt.subplots()

    #Getting the data
    tickers = [x for x in stocks_values]
    percent_change = [round(100*float(stocks_values[x]["change"]),2) for x in stocks_values]
    last = [round(float(stocks_values[x]["last"]),2) for x in stocks_values]
    description = [stocks_values[x]["description"] for x in stocks_values]
    table_data = []
    for i in range (len(tickers)):
        table_data.append([description[i], tickers[i], percent_change[i], last[i]])
    
    #Helper function to sort table_data from highest change to lowest (2 signifies the 3 data type in the data fomrat: 0.desc, 1.ticker, 2.perchange, 3. last)
    def byChange(e):
        return e[2]
    #Sort table by highest change to lowest change
    table_data.sort(reverse = True, key=byChange)

    #Creating the table
    table = ax.table(cellText=table_data, loc = 'center', colLabels=["Description", "Symbol", "Percent Change", "Last"])

    #Modify the table
    table.set_fontsize(50)
    table.scale(40,10)
    ax.axis('off')
    ax.set_title("Top Movers: " + market, fontsize = 75, y = 10, pad =-14)

    #Save the table
    filename = 'Top Movers '
    fig.savefig(output_dir + filename + market + ' ' + now + '.png', bbox_inches='tight')
    
    
main()