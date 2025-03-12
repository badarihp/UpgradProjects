# Linear Regression Assignment
> This assignment is a programming assignment wherein you have to build a multiple linear regression model for the prediction of demand for shared bikes. You will need to submit a Jupyter notebook for the same.

TODO : Added file which uses standard scaler, work in progress

## Table of Contents
* [Bike Sharing Assignment Intro](#bike-sharing-assignment-intro)
* [Business Goals](#business-goals)
* [Data Preparation](#data-preparation)
* [Model Building](#model-building)
* [Model Evaluation](#model-evaluation)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->
## Bike Sharing Assignment Intro
This is an assignment wherein a multiple linear regression model is expected to be built, to predict demand for bikes depending on the current trend.

In this programming task, you will create a multiple linear regression model that can estimate the demand for shared bikes based on the current trend. Shared bikes are bikes that people can rent and use for a short time, either for free or for a fee. They can pick up a bike from a computerized station, where they enter their payment information and get a code to unlock the bike. They can then drop off the bike at another station that belongs to the same system.

## Business Goals
The goal of this project is to build a multiple linear regression model that can predict the demand for shared bikes using the available explanatory variables. This will help the management to know how the demand changes with different factors. They can then adjust their business strategy to match the demand levels and satisfy the customers. Moreover, the model will be a useful tool for management to understand the demand patterns of a new market.

##  Data Preparation
In the dataset, some of the features like 'weathersit' and 'season' have numbers 1, 2, 3, 4 that represent different labels (you can see what they mean in the data dictionary). These numbers may make you think that there is some order or ranking among them - but that is not true (Look at the data dictionary and think why). So, it is better to change these feature values into categorical text values before you build the model. Please check the data dictionary to understand all the independent variables better. You may also notice the feature 'yr' with two values 0 and 1 that show the years 2018 and 2019 respectively. You may feel like dropping this feature as it only has two values so it may not be useful for the model. But in fact, since these bike-sharing systems are becoming more popular, the demand for these bikes is growing every year which means that the feature 'yr' may be a good predictor. So be careful before you drop it.

## Model Building
The dataset has three features called 'casual', 'registered', and 'cnt'. The feature 'casual' shows the number of casual users who rented a bike. The feature 'registered' shows the number of registered users who booked a bike on a given day. The feature 'cnt' shows the total number of bike rentals, including both casual and registered. You should build the model using this 'cnt' as the target variable.

## Model Evaluation
After you have built the model and analyzed the residuals and made predictions on the test set, please make sure you use these two lines of code to find out the R-squared score on the test set:

```python
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```

where y_test is the test data set for the target variable, and y_pred is the variable containing the predicted values of the target variable on the test set.


## Conclusions
- The target variable is cnt, which is the number of bike rentals in a day.
- There are more than three features that have a positive correlation with the target variable, which means they increase the demand for bike rentals.
- The features that have the highest positive correlation with the target variable are atemp, yr, season_winter and mnth_Sep, with correlation values of 0.4632, 0.2350, 0.0412 and 0.0587 respectively.

We can see that the equation for best fitted line is:
```cnt = 0.2620 + 0.2350 X yr - 0.1028 X holiday + 0.4632 X atemp - 0.1254 X windspeed -0.1167 X season_Spring + 0.0412 X season_Winter - 0.0657 X mnth_July + 0.0587 X mnth_Sep - 0.2872 X weathersit_Light rain_Light snow_Thunderstorm - 0.0837 X weathersit_Mist_cloudy -0.0484 X weekday_Sunday```

Comparision between Training and Testing dataset:

| **Item**          | **Train Data Set** | **Test Data Set** |
| ----------------- | -------------- | ------------- |
| R^2               |   0.836        | 0.8035        |
| Adjusted R^2      |   0.832        | 0.7845        |

## Technologies-Used
- python - version 3.10.9
- pandas - version 1.5.3
- numpy - version 1.23.5
- matplotlib - version 3.7.0
- scikit-learn - version 1.2.1
- seaborn - version 0.12.2
- statsmodels - version 0.13.5

## Contact
Created by [@badarihp] - feel free to contact me!

## Acknowledgements
Give credit here.
- https://seaborn.pydata.org/
- https://pandas.pydata.org/
- https://learn.upgrad.com/
- https://github.com/ContentUpgrad/Linear-Regression/tree/main 
