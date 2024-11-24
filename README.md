# Used Car Pricing Analysis
## Problem Statement  
The problem statement is to predict used car pricing, given the car features. The dataset contains information on 426K cars, including various car features as well as their price. The goal is to understand what factors impact the price of the car. A regression model is built to predict the price of the used car.

## Exploratory Data Analysis
For understanding the data and relationship between the features and price, an exploratory data analysis is done. This included previewing the data columns, sample rows, data types and descriptive statistics. After that, the distribution of the data columns was examined, identifying any outliers. Finally, a look into nulls in various columns and use of relevant imputation strategy is employed.

<img width="362" alt="Screenshot 2024-11-23 at 9 05 16 PM" src="https://github.com/user-attachments/assets/3707a8f6-ac1f-4b77-93d3-b06a61f536be">

- The dataset has 17 features and price columns and over 420K rows.
- 14 of the independent features are object while id, VIN, odometer and year are the only numerical fields.
- "id", "VIN" columns can be dropped from analysis since they will not impact a pricing model.

- Descriptive statistics of numeric columns highlights outliers of lower and upper end of price and odometer. Outlier trimming required.
- Price of 0 should be removed. Price above 80,000 also can be removed as outliers. Price distrbution is skewed. Using log of price may improve model.
- Lower end of year of 1900 seems incorrect. Removing 1920 and below.
- Odometer reading of 0 seems invalid. There are a number of outlier above 275,000. Zooming in to get better reading on the upper limit -

<img width="593" alt="Screenshot 2024-11-23 at 9 32 14 PM" src="https://github.com/user-attachments/assets/f28213d4-bea7-4dd8-af7e-1fd59a9c1ea3">

- There are number of columns with a high rate of nulls. "size" column with over 70% nulls is being excluded since any imputation does not seem like a good option.
<img width="294" alt="Screenshot 2024-11-23 at 9 33 31 PM" src="https://github.com/user-attachments/assets/73268020-d05a-4a24-b299-e36a1f4fbedb">

# Data Preparation
A 3 tier approach was taken towards preparing the dataset for modeling. First tier was to use just the two numerical features to get a basline model. Second tier is to include categorical features with small cardinality along with numerical features. Finally, the third tier was to use as many categorical features as possible to create a complex model.

numerical columns = ["odometer", "year"]
low cardinality columns = ["fuel", "drive", "cylinders", "title_status", "condition", "transmission"]
high cardinality columns = ["paint_color", "type", "region", "state", "manufacturer", "model"]

Observations on the low cardinality columns - 
- Price is higher for diesel fuel type than other fuel types.
- fwd drive has lower price than other types.
- 12 cylinder cars are pricier and wider in range, 5 cylinders are the lowest price.
- title_status of lien has large range and highest price compared to other status. Similarly, new condition has the highest price and salvage the lowest.
- Upper bound of best cars is at 90,000. Trimmed price down to 90,000

For high cardinality columns, the trailing categories are grouped into "Other" category to reduce the categories. For example, the trailing paint colors of "yellow", "purple", "orange" were grouped into "other" category.

<img width="595" alt="Screenshot 2024-11-23 at 10 01 31 PM" src="https://github.com/user-attachments/assets/9f792585-7a3d-4155-abd6-2789f25bc427">

Strategy for imputing for nulls -
- For columns that have less than 20% nulls, most frequent value is filled in for the nulls post the train/test split
- For columns with high amount of nulls, missing values are filled as "Missing".

Columns dropped because of very high null count and very large cardinality are - "size", "manufacturer", "model", "region", "state", "condition"

# Modeling
- Column transformer is made with numerical columns normalized and categorical columns dummified using OneHotEncoder
- Polynomial features optimized for no overfitting. Only degree 1 and 2 are evalauted in the interest of time, since degree 3 was computationally very slow.
- Cross validation used for regularization paramter optimization
- Model error and R2 score compared to base model

# Results
Baseline model -

LinearRegression with only numerical columns ======== <br/>
RMSE for training set is 11747.66, for test set is 11746.84. Price prediction is off by $11747.0 <br/>
R2 score for the best model is 0.32. 32.0% data points are explained by the model

with log of dependent column
LinearRegression with log price ======== <br/>
RMSE for training set is 1.05, for test set is 1.07. Price prediction is off by $1.0 <br/>
R2 score for the best model is 0.19. 19.0% data points are explained by the model

Log of pricing does not explain data points better than regular price.

Tier 2 model -

Linear Regression ========== <br/>
Best model params - {'poly__degree': 2} <br/>
Best model - RMSE for training set is 8467.75, for test set is 8515.34. Price prediction is off by $8515.0 <br/>
R2 score for the best model is 0.64. 64.0% data points are explained by the model

Ridge Regression =========== <br/> 
Best model params - {'model__alpha': 10, 'poly__degree': 2} <br/>
Best model - RMSE for training set is 8468.43, for test set is 8514.42. Price prediction is off by $8514.0 <br/>
R2 score for the best model is 0.64. 64.0% data points are explained by the model

Lasso Regression ========== <br/>
Best model params - {'model__alpha': 0.01, 'poly__degree': 2} <br/>
Best model - RMSE for training set is 8642.7, for test set is 8706.14. Price prediction is off by $8706.0 <br/>
R2 score for the best model is 0.63. 63.0% data points are explained by the model

Tier 3 model -

Linear Regression ========== <br/>
Best model params - {'poly__degree': 2} <br/>
Best model - RMSE for training set is 8024.91, for test set is 8070.37. Price prediction is off by $8070.0 <br/>
R2 score for the best model is 0.68. 68.0% data points are explained by the model

Ridge Regression =========== <br/>
Best model params - {'model__alpha': 10, 'poly__degree': 2} <br/>
Best model - RMSE for training set is 8026.33, for test set is 8069.16. Price prediction is off by $8069.0 <br/>
R2 score for the best model is 0.68. 68.0% data points are explained by the model

Lasso Regression ========== <br/>
Best model params - {'model__alpha': 0.01, 'poly__degree': 2} <br/>
Best model - RMSE for training set is 8203.34, for test set is 8268.57. Price prediction is off by $8269.0 <br/>
R2 score for the best model is 0.66. 66.0% data points are explained by the model

# Interpretation of the results
- Ridge Regression gives only a very slight improvement in accuracy while explaining the same percent of datapoints as basic Linear Regression. Lasso Regression optimizes to poorer performance than Linear Regression. Use Linear Regression model.
- Complex model gives the lowest RMSE error and explains over 68% of datapoints. Intermediate model without using the high cardinality columns gives R2 score of 64%.
- Going with the best model of Ridge Regression with the tier 3 dataset
- The top features influencing the pricing are fuel and type of car with the highest influence for electric 
<img width="429" alt="Screenshot 2024-11-23 at 10 52 55 PM" src="https://github.com/user-attachments/assets/3dc1bb45-ddac-445c-8086-02c76cee3bff">

Further work with different data imputation strategy could improve the model performance. Feature selection methods could also help trim the features and improve model performance and interpretation. 

