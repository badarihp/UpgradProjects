# Surprise Housing House Prediction
> A US-based housing company named Surprise Housing has decided to enter the Australian market. The company uses data analytics to purchase houses at a price below their actual values and flip them on at a higher price. For the same purpose, the company has collected a data set from the sale of houses in Australia. The data is provided in the CSV file below.

## Project Source link
https://github.com/badarihp/House_price_prediction_adv_regression.git
 
## Table of Contents
* [Problem Statement](#problem-statement)
* [General Info](#general-information)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## Problem Statement

### Business Understanding

The company is looking at prospective properties to buy to enter the market. You are required to build a regression model using regularisation in order to predict the actual value of the prospective properties and decide whether to invest in them or not.

### Business Goal:

You are required to model the price of houses with the available independent variables. This model will then be used by the management to understand how exactly the prices vary with the variables. They can accordingly manipulate the strategy of the firm and concentrate on areas that will yield high returns. Further, the model will be a good way for management to understand the pricing dynamics of a new market.

### Business Risk:

- Customers will not buy a house if the company predicts a sale price that is higher than its value, resulting in a loss for the company.

### Requirement:

The company wants to know:
- Which variables are significant in predicting the price of a house, and
- How well those variables describe the price of a house.

Also, determine the optimal value of lambda for ridge and lasso regression.

## General Information
- Steps :
<ol>
    <li>Data Visualization</li>
      <ol>
        <li>Perform EDA to understand various variables.</li>
        <li>Check the correlation between the variables.</li>
      </ol>
    <li>Data Preparation</li>
      <ol>
        <li>Create dummy variables for all the categorical features.</li>
        <li>Divide the data to train & Test.</li>
        <li>Perform Scaling.</li>
        <li>Divide data into dependent & Independent variables.</li>
      </ol>
    <li>Data Modelling & Evaluation</li>
      <ol>
        <li>Create Linear Regression model </li>
        <li>Create Ridge and Lasso models</li>
        <li>Check the various assumptions.</li>
        <li>Check the Adjusted R-Square for test data.</li>
        <li>Report the final model.</li>
      </ol>
</ol>
- Data Set : train.csv 


## Technologies Used
- pandas - 1.3.4
- numpy - 1.20.3
- matplotlib - 3.4.3
- seaborn - 0.11.2
- plotly - 5.8.0
- sklearn - 1.1.2
- statsmodel - 0.13.2


<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
- https://www.geeksforgeeks.org/
- https://seaborn.pydata.org/
- https://plotly.com/
- https://pandas.pydata.org/
- https://learn.upgrad.com/

## Conclusions

We used lasso and ridge regression for the housing prediction. Please ref to below table :

|	**Model** |	**Alpha** |	**MAE** |	**MSE** |	**RMSE**	| **R2 Score** |	**RMSE - Cross-Validation**|
|-----------|-----------|---------|---------|----------|--------------|----------------------------|
|Ridge	| 10	| 0.08061	| 0.01291	| 0.11362	| 0.91443	| 0.115130 |
|Lasso	|0.001| 0.08236 |	0.01283	| 0.11326	| 0.91497	| 0.116269 |
|LinearRegression	|NA	|3.566540e+07|	2.345172e+17|	4.842697e+08	|-2.270392e+17	| 0.126760|

## Contact
Created by [@badarihp] - feel free to contact me!
