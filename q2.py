
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters 
register_matplotlib_converters()
import seaborn as sns
import datetime

def main():

    conversions = sys.argv[1]
    sales = sys.argv[2]
    visits = sys.argv[3]

    conversions_df = pd.read_csv(conversions,parse_dates=['hour'])
    sales_df = pd.read_csv(sales,parse_dates=['hour'])
    visits_df = pd.read_csv(visits,parse_dates=['hour'])


    # Question 2:
    # Which months have the highest average sales?

    month_sales_df = sales_df[['hour','orders']]
    month_sales_df['date'] = month_sales_df['hour'].dt.date
    month_sales_df['month'] = month_sales_df['hour'].dt.month
    month_sales_df['year'] = month_sales_df['hour'].dt.year
    month_sales_df = month_sales_df.groupby(['month','year']).sum().reset_index()
    group_by_month = month_sales_df.groupby(['month']).mean().reset_index()
    del group_by_month['year']
    group_by_month = group_by_month.sort_values(by=['orders'], ascending=False) 

    ax=sns.barplot(x=group_by_month['month'], y=group_by_month['orders'])
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean Orders')
    ax.set_title('Month vs. Mean Orders')    
    plt.show()

if __name__ == '__main__':
    main()
