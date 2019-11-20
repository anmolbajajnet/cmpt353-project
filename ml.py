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
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint


# Run Command
# python3 ml.py data/conversions_2017-01-01_2019-01-01-2.csv data/sales_2017-01-01_2019-01-01.csv data/visits_2017-01-01_2019-01-01.csv

def custom_round(x, base):
    return int(base * round(float(x)/base))

def custom_score(y_pred,y_true):
    accurate = 0
    for i in range(0,len(y_pred)-1):
        if abs(y_pred[i] - y_true[i]) <= 2:
            accurate += 1
    score = accurate/len(y_pred)
    return score

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


    joined_df.orders = joined_df.orders.apply(lambda x: custom_round(x, 5))

    


    joined_df = joined_df.drop(columns=['hour','date'])
    joined_df = joined_df.groupby(['year','month', 'day']).sum().reset_index().round()

    # print(joined_df)

    # print(joined_df)
    joined_df.to_csv('joined.csv')

    # joined = sys.argv[1]
    # joined_df = pd.read_csv(joined,parse_dates=['hour'])
    # joined_df.orders = joined_df.orders.apply(lambda x: custom_round(x, base=5))

    # Predicting orders => 85% Accuracy using RFR and hyperparameter tuning parameters
    X = joined_df[['year','day','month','total_sessions_x','total_carts','total_checkouts','total_sales','gross_sales','total_visitors']]
    y = joined_df['orders']

    # Predicting Month => 25% Accuracy with Knn Model Score, 60% Accuracy with customized score that allows for the model to be wrong by couple months
    # X = joined_df[['year','day','orders','total_sessions_x','total_carts','total_checkouts','total_sales','gross_sales','total_visitors']]
    # y = joined_df['month']
    
    # print(y.dtypes)
    # print("Y IS")
    # print(y)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30, random_state=42)
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

    # 20 gives 28% accuracy
    knn_model = KNeighborsClassifier(n_neighbors=20)
    knn_model.fit(X_train, y_train)
    print('Using Knn')
    print(knn_model.score(X_valid, y_valid))
    # print("Predictions are..")
    # print(knn_model.predict(X_valid))
    # print("Valid results are..")
    # print(y_valid)


    rfc_model = RandomForestClassifier(n_estimators=30,
        max_depth=8, min_samples_leaf=10)

    

    rfc_model.fit(X_train, y_train)
    print('Using Rfc')
    print(rfc_model.score(X_valid, y_valid))
    print("Predictions are..")
    y_pred = rfc_model.predict(X_valid)
    print(rfc_model.predict(X_valid))
    print("Valid results are..")
    y_true = y_valid.to_numpy()
    print(y_valid.to_numpy())
    print(mean_squared_error(y_true, y_pred))

    # Custom "score" function since the model is ~60% accurate if you include the months it ~almost~ gets right
    print("Custom score is")
    print(custom_score(y_pred,y_true))

    
    rfr_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=70, n_jobs=None,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

    rfr_model.fit(X_train, y_train)
    print('Using Rfr')
    print(rfr_model.score(X_valid, y_valid))

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