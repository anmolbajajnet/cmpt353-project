# CMPT353 - Final Report

## Purpose

There are two (2) main Python files for the analysis of data from e-commerce website [BrightWhiteSmilePro.com](https://brightwhitesmilepro.com/).


<img src="https://cdn.shopify.com/s/files/1/0548/5765/files/sig_410x.png?v=1516980032">

### chart.py

This file groups the three (3) different files in the `/data/` directory. Dataframes are grouped in various ways to produce the plots for hourly,
monthly, and yearly trends. Statistical analysis is performed to determine p-values for normality, equal-variance, and t-tests.

### ml.py

This file the code to train machine learning models and output the accuracy scores. The code for generating the optimized parameters for the
Random Forest Regressor (RFR) is also found here. The resulting predictions and actual results are plotted on a line-plot, along with a residual
plot to visualize accuracy.  


## Order of operation

The order of operation is irrelevant as both .py files are independent of each other.


## Commands

Running ml.py 

`$ python3 ml.py data/conversions_2017-01-01_2019-01-01-2.csv data/sales_2017-01-01_2019-01-01.csv data/visits_2017-01-01_2019-01-01.csv`

Running chart.py 

`$ python3 chart.py data/conversions_2017-01-01_2019-01-01-2.csv data/sales_2017-01-01_2019-01-01.csv data/visits_2017-01-01_2019-01-01.csv`

## Expected Output

The following files will be output as the result of successfull operation of `ml.py` and `chart.py`
- DayTrend.png - Plots the mean hourly sales in each hour of the day 
- joined.csv - Resulting CSV after merging the conversion, sales, and visits into one file
- MonthlyAvgSales.png - Plots the mean orders in each month
- PredVsActualOrders.png - Plots the predicted number of orders and the actual number of orders
- ResidualsVsCounts.png - Bar plot to visuale the difference between predicted number of orders vs actual number of orders
- YearlyTrend.png - Plots the monthly sales from each year to determine yearly growth


## Libraries Used
- numpy
- pandas
- seaborn
- scipy
- matplotlib
- sklearn

