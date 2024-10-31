# datahow coding challenge

## Implementation
There are two different implementations for the datahow coding challenge.

The first implementation was done with a [random forest regression model](RandomForest.py) and the second with a [LSTM](LSTM.py).

You can run `python3 RandomForest.py` or `python3 LSTM.py` in order to get the descriptive statistics, the metrics, and the plots.

The visualizations can be found [here](visualizations).

The results of the Random Forest Regression model are the following.
Training Set Metrics:
R² Score: 0.660
RMSE: 436.928
MAE: 287.698
MAPE: 30.01%

Test Set Metrics:
R² Score: -0.639
RMSE: 1046.780
MAE: 893.544
MAPE: 112.27%

![Random Forest Training Plots](/home/amohap/github/datahow/visualizations/RandomForest_Train.png)
![Random Forest Testing Plots](/home/amohap/github/datahow/visualizations/RandomForest_Test.png)

The results of the LSTM are the following.
Training Set Metrics:
R² Score: 0.559
RMSE: 588.267
MAE: 393.644
MAPE: 69.11%

Test Set Metrics:
R² Score: 0.775
RMSE: 387.365
MAE: 306.550
MAPE: 16.24%

![LSTM Loss](/home/amohap/github/datahow/visualizations/LSTM_Loss.png)
![LSTM Testing](/home/amohap/github/datahow/visualizations/LSTM_Testing.png)