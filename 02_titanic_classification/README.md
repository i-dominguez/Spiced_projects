## 02. Supervised Machine Learning: Classification - Kaggle's Titanic Challenge

The goal of this project was to built a machine learning model to predict the survival of Titanic passenger based on the features in the dataset of Kaggle's Titanic - Machine Learning from Disaster.

Based on the Exploratory Data Analysis (plotted missing values and the correlation between survival and the different data categories) selected the most significant features and dropped the ones which cannot contribute to accurate prediction.

In feature engineering using ColumnTransformer, I applied 1) OneHotEncoder: to convert categorical variables into binary features, 2) SimpleImputer: to fill missing values and 3) MinMaxScaler: to normalize continous numerical variable in range 0.0 - 1.0.

The data was trained on Scikit-learn's LogisticRegression and RandomForestClassifier models. After evaluating different model's accuracy scores and cross validation, I kept the RandomForest model for prediction (cross validation: mean accuracy score 82.06%).

Data source: [Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview).
