Restaurant Rating Predictor


A machine learning project that predicts the aggregate rating of a restaurant
using features like votes, location, cuisine type, cost, and more.

Built as part of an internship data science task.


Dataset
-------
File    : Dataset_internship.csv
Rows    : 9,551 restaurants
Columns : 21 features
Target  : Aggregate rating (0.0 – 4.9)

Source features used:
  - Country Code, City, Longitude, Latitude
  - Cuisines, Currency
  - Average Cost for two, Price range
  - Has Table booking, Has Online delivery
  - Is delivering now, Switch to order menu
  - Votes


Project Structure
-----------------
  restaurant_rating_ml.py   Main ML pipeline (preprocessing + training + evaluation)
  Dataset_internship.csv    Raw dataset (place in the same folder)
  README.txt                This file


Requirements
------------
Python 3.7+





1. Install dependencies
   pip install pandas numpy scikit-learn

2. Run the script
   python restaurant_rating_ml.py


Preprocessing Steps

1. Dropped leakage columns: Rating color, Rating text (direct proxies of target)
2. Dropped ID/address columns: Restaurant ID, Name, Address, Locality
3. Filled 9 missing values in Cuisines column with "Unknown"
4. Binary encoded Yes/No columns -> 1/0
5. Label encoded: City, Cuisines, Currency
6. Train/test split: 80% train, 20% test (random_state=42)


Models Trained
--------------
1. Linear Regression
2. Decision Tree Regressor  (max_depth=8)
3. Random Forest Regressor  (n_estimators=100, max_depth=10)


Results
-------
Model                 MSE     RMSE    MAE     R2
--------------------  ------  ------  ------  ------
Linear Regression     1.5862  1.2594  1.0480  0.3031
Decision Tree         0.1023  0.3199  0.2092  0.9550
Random Forest         0.0879  0.2964  0.1932  0.9614  <-- best

Best model: Random Forest with R2 = 0.9614


Key Finding
-----------
The Votes feature (number of user reviews) accounts for ~97% of the Random
Forest's feature importance. Restaurants with more votes tend to have more
stable and representative aggregate ratings.


Metrics Explained
-----------------
MSE  : Mean Squared Error — average of squared prediction errors
RMSE : Root MSE — error in original rating units (easier to interpret)
MAE  : Mean Absolute Error — average absolute difference from actual
R2   : Proportion of variance explained by the model (1.0 = perfect)


Author
------
[Himanshu Pandey]
[shrikrishna7394849@gmail.com]
