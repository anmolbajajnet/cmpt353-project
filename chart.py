import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters 
register_matplotlib_converters()
import seaborn


# Run Command
# python3 chart.py data/conversions_2017-01-01_2019-01-01-2.csv data/sales_2017-01-01_2019-01-01.csv data/visits_2017-01-01_2019-01-01.csv

def main():
    conversions = sys.argv[1]
    sales = sys.argv[2]
    visits = sys.argv[3]

    conversions_df = pd.read_csv(conversions,parse_dates=['hour'])
    sales_df = pd.read_csv(sales,parse_dates=['hour'])
    visits_df = pd.read_csv(visits,parse_dates=['hour'])
    # print(conversions_df)
    # print(sales_df)
    # print(visits_df)

    # What time of the day has the highest average sales?
    hour_sales_df = sales_df[['hour','total_sales']]
    hour_sales_df['total_conversion'] = conversions_df['total_conversion']
    hour_sales_df['time'] = [d.time().hour for d in hour_sales_df['hour']]
    hour_sales_df = hour_sales_df.groupby(['time']).mean().sort_values(by=['total_sales'], ascending=False).reset_index()
    print(hour_sales_df)
    print("The time in the day with the most sales is 18:00, followed by 13:00, 19:00, and 12:00")
    k2, p = stats.normaltest(hour_sales_df['total_sales'])
    print(p)
    if (p < 0.05):
        print("The data does not follow a normal distribution")
    else:
        print("Fail to reject H0 => There is not enough evidence to support that data does not follow a normal distribution")

    seaborn.set()
    plt.subplot(2, 1, 1)
    plt.xlabel("Hour in the Day (24 Hour Clock)")
    plt.ylabel("Mean Sales (in $)")
    plt.bar(hour_sales_df['time'],hour_sales_df['total_sales'])

    plt.subplot(2, 1, 2)
    plt.xlabel("Hour in the Day (24 Hour Clock)")
    plt.ylabel("Mean Conversion Rate")
    plt.bar(hour_sales_df['time'],hour_sales_df['total_conversion'],color='r')
    plt.show()
  

    joined_df = conversions_df.merge(sales_df,on='hour').merge(visits_df,on='hour')
    joined_df['new_date'] = [d.date().month for d in joined_df['hour']]
    joined_df['new_time'] = [d.time().hour for d in joined_df['hour']]
    # print(joined_df)
    joined_df.to_csv('joined.csv',index=False)

    # Compute r-squared value
    slope, intercept, r_value, p_value, std_err = stats.linregress(joined_df['total_sales'], joined_df['total_visitors'])
    print("R-squared: %f" % r_value**2)

    # Compute pairwise correlation 
    print(joined_df['total_sales'].corr(joined_df['total_visitors']))

    # Are time and sales correlated?
    time_sales_corr = joined_df['total_sales'].corr(joined_df['new_time'])
    print ('Correlation coefficient of total sales and time in the day is %d' % (time_sales_corr))

    # plt.plot(joined_df['hour'],joined_df['total_sales'], label='total sales')
    # plt.plot(joined_df['hour'], joined_df['total_visitors'], label='total visitors')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()
