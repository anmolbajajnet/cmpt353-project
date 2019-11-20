import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

# Run Command
# python3 ml.py joined.csv

def custom_round(x, base):
    return int(base * round(float(x)/base))


def main():

    conversions = sys.argv[1]
    sales = sys.argv[2]
    visits = sys.argv[3]

    conversions_df = pd.read_csv(conversions,parse_dates=['hour'])
    sales_df = pd.read_csv(sales,parse_dates=['hour'])
    visits_df = pd.read_csv(visits,parse_dates=['hour'])

    joined_df = conversions_df.merge(sales_df,on='hour').merge(visits_df,on='hour')
    joined_df['date'] = [d.date() for d in joined_df['hour']]
    joined_df['year'] = [d.date().year for d in joined_df['hour']]
    joined_df['month'] = [d.date().month for d in joined_df['hour']]
    joined_df['day'] = [d.date().day for d in joined_df['hour']]
    joined_df['time'] = [d.time() for d in joined_df['hour']]

    joined_df.orders = joined_df.orders.apply(lambda x: custom_round(x, 5))

    


    joined_df = joined_df.drop(columns=['hour','date'])
    joined_df = joined_df.groupby(['year','month', 'day']).sum().reset_index().round()

    # print(joined_df)

    # print(joined_df)
    joined_df.to_csv('joined.csv')

    # joined = sys.argv[1]
    # joined_df = pd.read_csv(joined,parse_dates=['hour'])
    # joined_df.orders = joined_df.orders.apply(lambda x: custom_round(x, base=5))

    # X = joined_df[['year','day','orders','total_sessions_x','total_conversion','total_sales']].round(-1)
    X = joined_df[['year','day','month','total_sessions_x','total_sales']]
    
    # print(X)
    
    y = joined_df['orders']
    
    # print(y.dtypes)
    print("Y IS")
    print(y)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.50, random_state=42)
    # print("XTRAIN IS")
    # print(X_train)


    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    model.fit(X_train, y_train)
    print('Using SVC')
    print(model.score(X_valid, y_valid))

    gaus_model =  GaussianNB()

    gaus_model.fit(X_train, y_train)
    print('Using Gauss')
    print(gaus_model.score(X_valid, y_valid))

    knn_model = KNeighborsClassifier(n_neighbors=20)
    knn_model.fit(X_train, y_train)
    print('Using Knn')
    print(knn_model.score(X_valid, y_valid))

    rfc_model = RandomForestClassifier(n_estimators=100,
        max_depth=3, min_samples_leaf=10)
    rfc_model.fit(X_train, y_train)
    print('Using Rfc')
    print(rfc_model.score(X_valid, y_valid))


if __name__ == '__main__':
    main()