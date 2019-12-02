import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters 
register_matplotlib_converters()
import seaborn as sns
import datetime



# Run Command
# python3 chart.py data/conversions_2017-01-01_2019-01-01-2.csv data/sales_2017-01-01_2019-01-01.csv data/visits_2017-01-01_2019-01-01.csv

def normal_test(p):
    if (p < 0.05):
        return "The data does not follow a normal distribution"
    else:
        return "Fail to reject H0 => There is not enough evidence to support that data does not follow a normal distribution"
    

def main():
    conversions = sys.argv[1]
    sales = sys.argv[2]
    visits = sys.argv[3]

    conversions_df = pd.read_csv(conversions,parse_dates=['hour'])
    sales_df = pd.read_csv(sales,parse_dates=['hour'])
    visits_df = pd.read_csv(visits,parse_dates=['hour'])

    # Question 1:
    # What time of the day has the highest average sales?
    hour_sales_df = sales_df[['hour','total_sales']]
    hour_sales_df['time'] = hour_sales_df['hour'].dt.time
    hour_sales_df['time'] = hour_sales_df['hour'].dt.hour 
    hour_sales_df = hour_sales_df.groupby(['time']).mean().reset_index()
    # .sort_values(by=['total_sales'], ascending=False).reset_index()
    # print("The time in the day with the most sales is 18:00, followed by 13:00, 19:00, and 12:00")
    k2, p = stats.normaltest(hour_sales_df['total_sales'])
    print(p)
    print(normal_test(p))

    sns.set()
    fig, ax = plt.subplots()
    sns.barplot(x=hour_sales_df['time'], y=hour_sales_df['total_sales'])
    ax.set_xlabel("Hour in the Day (24 Hour Clock)")
    ax.set_ylabel("Mean Sales (in $)")
    ax.set_title("Hour in the Day vs Sales")
    fig.savefig('DayTrend.png')
    # plt.show()

    # Question 1 Statistics
    print("Normality test for Hourly Sales")
    print(stats.normaltest(hour_sales_df['total_sales']).pvalue)


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
    plt.savefig('MontlyAvgSales.png')    
    # plt.show()
    

    # Question 3:
    # How is the company growing every year? What are the yearly trends?

    yearly_sales_df = sales_df[['hour','orders']]
    yearly_sales_df['date'] = yearly_sales_df['hour'].dt.date
    yearly_sales_df['month'] = yearly_sales_df['hour'].dt.month
    yearly_sales_df['year'] = yearly_sales_df['hour'].dt.year
    yearly_sales_df = yearly_sales_df.groupby(['year','month']).sum().reset_index()

    mask = yearly_sales_df['year'] == 2017
    year_2017 = yearly_sales_df[mask]
    year_2018 = yearly_sales_df[~mask].reset_index()
    frame = { 'months': year_2017['month'], 'orders_2017': year_2017['orders'],'orders_2018': year_2018['orders']  } 
    result = pd.DataFrame(frame) 
    print(result)

    sns.set(style="whitegrid")

    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(result['months'], result['orders_2018'], width, label='2018')
    rects2 = ax.bar(result['months'], result['orders_2017'], width, label='2017')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Orders')
    ax.set_xlabel('Months')
    ax.set_title('Orders by Months (Year 2017-2018)')
    plt.xticks(result['months'])
    ax.legend()

    # Label code from 
    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    def autolabel(rects):  
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    fig.savefig('YearlyTrend.png')
    # plt.show()

    # Question 3 Statistics

    # Normality Test
    print("Normality test for Year 2017 Sales")
    print(stats.normaltest(result['orders_2017']).pvalue)
    print("Normality test for Year 2018 Sales")
    print(stats.normaltest(result['orders_2018']).pvalue)

    # Equal Variance Test
    print("Equal Variance Test for Year 2017 and 2018")
    print(stats.levene(result['orders_2017'], result['orders_2018']).pvalue)

    # T-test 
    print("T-test for Year 2017 and 2018")
    print(stats.ttest_ind(result['orders_2017'], result['orders_2018']).pvalue)

    # joined_df = conversions_df.merge(sales_df,on='hour').merge(visits_df,on='hour')
    # joined_df['new_date'] = [d.date().month for d in joined_df['hour']]
    # joined_df['new_time'] = [d.time().hour for d in joined_df['hour']]
    # joined_df.to_csv('joined.csv',index=False)

    # # Compute r-squared value
    # slope, intercept, r_value, p_value, std_err = stats.linregress(joined_df['total_sales'], joined_df['total_visitors'])
    # print("R-squared: %f" % r_value**2)

    # # Compute pairwise correlation 
    # print(joined_df['total_sales'].corr(joined_df['total_visitors']))

    # # Are time and sales correlated?
    # time_sales_corr = joined_df['total_sales'].corr(joined_df['new_time'])
    # print ('Correlation coefficient of total sales and time in the day is %d' % (time_sales_corr))

    # plt.plot(joined_df['hour'],joined_df['total_sales'], label='total sales')
    # plt.plot(joined_df['hour'], joined_df['total_visitors'], label='total visitors')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()
