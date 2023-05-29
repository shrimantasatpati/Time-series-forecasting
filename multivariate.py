#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# Function to process the Excel file
def process_excel_file(file_path):
    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_csv(file_path, encoding="utf-8")
        
        df['TimeSeries_Key'] = df['product_code'] + '_' + df['customer_code']

        df['Week'] = df['calweek'].astype(str).str[-2:].astype(int)

        #df.columns

        df1 = df.drop(['Cluster/Country/Region', 'SubCluster', 'brand', 'dc'], axis = 1)

        timeseries_key_values = list(df1['TimeSeries_Key'].unique())

        # Convert 'calweek' column to datetime format
        df1['calweek'] = pd.to_datetime(df1['calweek'].astype(str) + '1', format='%Y%W%w')

        # Generate new columns for year, month, and quarter
        df1["year"] = df1["calweek"].dt.year
        df1["month"] = df1["calweek"].dt.month
        df1["quarter"] = df1["calweek"].dt.quarter
        # Generate a new column for season
        df1['season'] = df1['calweek'].dt.month.apply(lambda x: (x%12 + 3)//3)
        
        start_week = 16
        end_week = 28

        filtered_df = df[(df['Week'] >= start_week) & (df['Week'] <= end_week)]

        def generate_dataframes():
            grouped_df = filtered_df.groupby('TimeSeries_Key')
            for key, group in grouped_df:
                group['TimeSeries_Key'] = key
                yield group

        combined_df = pd.concat(generate_dataframes(), ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['TimeSeries_Key', 'Week'])
        combined_df = combined_df.drop(["product_code", "customer_code", "Quantity", "Week", "calweek"], axis = 1)
        #combined_df
        
        df_prod1 = df1[df1['TimeSeries_Key']=='PRDCT1_CUSTA']
        #df_prod1
        # Group data by 'calweek' and aggregate 'Quantity'
        ts_data = df_prod1.groupby('calweek')['Quantity'].sum()
        #ts_data
        
#         from statsmodels.tsa.stattools import adfuller
#         passing_data=adfuller(ts_data)
#         def adf_test(sales):
#             result=adfuller(ts_data)
#             labels = ['Test parameters', 'p-value','#Lags Used','Dataset observations']
#             for value,label in zip(result,labels):
#                 print(label+' : '+str(value) )
#             if result[1] <= 0.05:
#                 print("Dataset is stationary")
#             else:
#                 print("Dataset is non-stationary ")
#         adf_test(ts_data)

        # Create an empty dataframe to store MAPE values
        results_df = pd.DataFrame()
        results_mape = pd.DataFrame()
        
        for i in timeseries_key_values:
            df_prod1 = df1[df1['TimeSeries_Key']==i]

            print("----------RESULT FOR",i,"----------")

            window_size = 3  # Define the window size for rolling average

            # Calculate rolling average for Quantity column
            df_prod1['Rolling_Average'] = df_prod1['Quantity'].rolling(window=window_size).mean()
            # Lagged Quantity
            df_prod1['lag1_Quantity'] = df_prod1['Quantity'].shift(1)

            # Split data into training and testing sets
            train_size = int(len(df1) * 0.8)  # Adjust the split ratio as per your requirement
            train_df = df1[:train_size]
            test_df = df1[train_size:]

            # Select the relevant columns for modeling
            X_train = train_df.drop(['calweek', 'product_code', 'customer_code', 'TimeSeries_Key', 'Quantity', 'season','Week'], axis=1)
            y_train = train_df['Quantity']
            X_test = test_df.drop(['calweek', 'product_code', 'customer_code', 'TimeSeries_Key', 'Quantity', 'season','Week'], axis=1)
            y_test = test_df['Quantity']

            # Find optimal p, d, q values using Auto ARIMA
            auto_arima_model = pm.auto_arima(y_train, seasonal=False, suppress_warnings=True)
            p, d, q = auto_arima_model.order

            # ARIMA model
            arima_model = ARIMA(endog=y_train, order=(p, d, q))  # ARIMA(p, d, q)
            arima_fit = arima_model.fit()
            arima_predictions = arima_fit.forecast(steps=len(X_test))

            # SARIMAX model
            sarimax_model = SARIMAX(endog=y_train, exog=X_train, order=(p, d, q))  # SARIMAX(p, d, q)
            sarimax_fit = sarimax_model.fit()
            sarimax_predictions = sarimax_fit.forecast(steps=len(X_test), exog=X_test)

            # Random Forest
            rf = RandomForestRegressor(n_estimators=100)
            rf.fit(X_train, y_train)
            rf_predictions = rf.predict(X_test)

            #ExtraTrees 
            et_extra_model = ExtraTreesRegressor(n_estimators=100)
            et_extra_model.fit(X_train, y_train)
            et_extra_predictions = et_extra_model.predict(X_test)

            # Gradient Boosting with XGBoost
            xgb_model = XGBRegressor(n_estimators=100)
            xgb_model.fit(X_train, y_train)
            xgb_predictions = xgb_model.predict(X_test)



            # Calculate MAPE
            def calculate_mape(actual, forecast):
                return np.mean(np.abs((actual - forecast) / actual)) * 100

            # Calculate MAPE for ARIMA
            arima_mape = calculate_mape(test_df['Quantity'], arima_predictions)

            # Calculate MAPE for SARIMA
            sarima_mape = calculate_mape(test_df['Quantity'], sarimax_predictions)

            # Calculate MAPE for RandomForest
            rf_mape = calculate_mape(test_df['Quantity'], rf_predictions)

            # Calculate MAPE for ExtraTrees
            et_extra_mape = calculate_mape(test_df['Quantity'], et_extra_predictions)

             # Calculate MAPE for Gradient Boosting with XGBoost
            xgb_mape = calculate_mape(test_df['Quantity'], xgb_predictions)    


            #Print MAPE values 
            print("ARIMA - MAPE: {:.2f}%".format(arima_mape))
            print("SARIMA - MAPE: {:.2f}%".format(sarima_mape))
            print("RANDOMFOREST - MAPE: {:.2f}%".format(rf_mape))
            print("ExtraTrees - MAPE: {:.2f}%".format(et_extra_mape))
            print("XGB - MAPE: {:.2f}%".format(xgb_mape))

             #FORECASTINGGG

            # Split data into training and testing sets
            train = df_prod1
            y_train = train['Quantity']

            # ARIMA model
            arima_model = ARIMA(endog=y_train, order=(1, 0, 0))  # ARIMA(p, d, q)
            arima_fit = arima_model.fit()
            arima_predictions = arima_fit.forecast(steps=13)

            #print(arima_predictions)
            # Convert arima_predictions to a list
            arima_predictions = arima_predictions.tolist()

            # Generate date values for the forecasted period
            forecast_dates = pd.date_range(start=df_prod1['calweek'].iloc[-1], periods=13, freq='W').strftime('%Y-%m-%d')

            for j in range(13):
                results_df = pd.concat([results_df, pd.DataFrame({
                    'date': [forecast_dates[j]],
                    'TimeSeries_Key': [i],
                    'ARIMA_Predicted': [arima_predictions[j]]
                     })], ignore_index=True)

                results_mape = pd.concat([results_mape, pd.DataFrame({
                    'date': [forecast_dates[j]],
                    'TimeSeries_Key': [i],
                    'ARIMA_MAPE': [arima_mape],
                    'SARIMA_MAPE': [sarima_mape],
                    'RandomForest_MAPE': [rf_mape],
                    'ExtraTrees_MAPE': [et_extra_mape],
                    'XGB_MAPE': [xgb_mape],
                })], ignore_index=True) 
                
        #results_df

        #results_mape

        # Convert the date column to datetime format
        results_df['date'] = pd.to_datetime(results_df['date'])

        # Convert the date format to 'YYYY-WW'
        results_df['calweek'] = results_df['date'].dt.strftime('%Y%U')

        merged_df = pd.merge(results_df, combined_df, on='TimeSeries_Key')
        merged_df.drop_duplicates(subset=['date', 'TimeSeries_Key'], inplace=True)
        merged_df.reset_index(drop=True, inplace=True)
        #merged_df

        # Select specific columns from each dataframe
        final_df = df[['calweek', 'Cluster/Country/Region', 'SubCluster', 'brand', 'dc', 'TimeSeries_Key', 'Quantity']]
        final_pred_df = merged_df[['calweek', 'Cluster/Country/Region', 'SubCluster', 'brand', 'dc', 'TimeSeries_Key', 'ARIMA_Predicted']]

        # Concatenate the dataframes
        final_multivariate_csv = pd.concat([final_df, final_pred_df], axis=0)
        #final_multivariate_csv
         
        return final_multivariate_csv
    
    except Exception as e:
        print("Error occurred:", str(e))
        return None
    


# Main program
if __name__ == '__main__':
    # Get the file path from the user via command-line input
    file_path = input("Enter the path to the Excel file: ")
    
    # Call the function to process the Excel file
    final_csv = pd.DataFrame(process_excel_file(file_path))
    
    # Save the result as a CSV file
    final_csv.to_csv('final_multivariate.csv', index=False)

