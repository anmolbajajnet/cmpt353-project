import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters 
register_matplotlib_converters()
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
import datetime


# Run Command
# python3 ml.py data/conversions_2017-01-01_2019-01-01-2.csv data/sales_2017-01-01_2019-01-01.csv data/visits_2017-01-01_2019-01-01.csv

def custom_round(x, base):
    return int(base * round(float(x)/base))


def get_season(month):
    if month in (3,4,5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    elif month in (9,10,11):
        return "Fall"
    else:   
        return "Winter"

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
    joined_df['season'] = joined_df['month'].apply(lambda x: get_season(x))

    
    joined_df = joined_df.drop(columns=['hour','date'])
    joined_df = joined_df.groupby(['year','month', 'day']).sum().reset_index().round()


    joined_df.to_csv('joined.csv')

    # Predicting orders => ~65% Accuracy using RFR and hyperparameter tuned parameters
    X = joined_df[['year','day','month','total_sessions_x']]
    y = joined_df['orders']

    # Predicting Month => 25% Accuracy with Knn Model Score, 60% Accuracy with customized score that allows for the model to be wrong by couple months
    # X = joined_df[['year','day','orders','total_sessions_x','total_carts','total_checkouts','total_sales','gross_sales','total_visitors']]
    # y = joined_df['month']


    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30, random_state=42)

    svc_model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    svc_model.fit(X_train, y_train)
    print('Score using SVC')
    print(svc_model.score(X_valid, y_valid))

    gaus_model =  GaussianNB()

    gaus_model.fit(X_train, y_train)
    print('Score using Gauss')
    print(gaus_model.score(X_valid, y_valid))

  
    knn_model = KNeighborsClassifier(n_neighbors=20)
    knn_model.fit(X_train, y_train)
    print('Score using KNN')
    print(knn_model.score(X_valid, y_valid))


    rfc_model = RandomForestClassifier(n_estimators=30,
        max_depth=8, min_samples_leaf=10)    

    rfc_model.fit(X_train, y_train)
    print('Score using RFC:')
    print(rfc_model.score(X_valid, y_valid))


    # Default RFR Model
    default_rfr_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
    
    default_rfr_model.fit(X_train, y_train)
    print('Score using Default parameters for RFR')
    print(default_rfr_model.score(X_valid, y_valid))

    # RFR Model with Hypertuned Parameters 
    rfr_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=7,
                      max_features='sqrt', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=33,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)

    rfr_model.fit(X_train, y_train)
    print('Score using RFR Model')
    print('- Valid Data Score')
    print(rfr_model.score(X_valid, y_valid))
    print('- Training Data Score')
    print(rfr_model.score(X_train, y_train))
    

    y_pred = rfr_model.predict(X_valid)
    y_true = y_valid.to_numpy()


    graph_df =  pd.DataFrame({'y_pred':y_pred.round().astype(int), 'y_true': y_true})
    data = X_valid[['year', 'month', 'day']].apply(lambda s : datetime.datetime(*s),axis = 1)
    time_df = pd.DataFrame(data, columns = ['timestamp']).reset_index()
    graph_df['date'] = time_df['timestamp']

    sns.set(style="darkgrid")

    # Plotting ML Prediction graphs
    graph_df = graph_df.sort_values(by=['date'])
    fig, ax = plt.subplots()
    ax.set_xlabel('Date (YYYY-MM)')
    ax.set_ylabel('Orders over Time')
    ax.set_title('Date vs. Orders over Time')      
    ax.plot(graph_df['date'], graph_df['y_pred'], label="Predicted Orders")
    ax.plot(graph_df['date'], graph_df['y_true'], label="Actual Orders")
    ax.legend()
    plt.show()
    fig.savefig('PredVsActualOrders.png')
    # plt.savefig('PredVsActualOrders.png')

    # Plotting the Residual graph
    fig, ax = plt.subplots()
    ax.hist(graph_df['y_pred']-graph_df['y_true'])
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Counts")
    ax.set_title("Residuals vs Counts")
    # plt.show()
    plt.savefig('ResidualsVsCounts.png')



# Hyperparameter Tuning => https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# This model generates the best parameters to use.

#     # Number of trees in random forest
#     n_estimators = [int(x) for x in np.linspace(start = 3, stop = 70, num = 50)]
#     # Number of features to consider at every split
#     max_features = ['auto', 'sqrt']
#     # Maximum number of levels in tree
#     max_depth = [int(x) for x in np.linspace(3, 70, num = 50)]
#     max_depth.append(None)
#     # Minimum number of samples required to split a node
#     min_samples_split = [2, 5, 10]
#     # Minimum number of samples required at each leaf node
#     min_samples_leaf = [1, 2, 4]
#     # Method of selecting samples for training each tree
#     bootstrap = [True, False]# Create the random grid
# # Create the random grid
#     random_grid = {'n_estimators': n_estimators,
#                 'max_features': max_features,
#                 'max_depth': max_depth,
#                 'min_samples_split': min_samples_split,
#                 'min_samples_leaf': min_samples_leaf,
#                 'bootstrap': bootstrap}

#     pprint(random_grid)

#     # Use the random grid to search for best hyperparameters
#     # First create the base model to tune
#     rf = RandomForestRegressor()
#     # Random search of parameters, using 3 fold cross validation, 
#     # search across 100 different combinations, and use all available cores
#     rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 2000, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
#     rf_random.fit(X_train, y_train)
#     print(rf_random.best_params_)
#     print(rf_random.best_estimator_)
            

if __name__ == '__main__':
    main()