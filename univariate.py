#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
        
        # ARIMA model
        def arima_forecast(train, test):

            # Find optimal p, d, q values using Auto ARIMA
            auto_arima_model = pm.auto_arima(train, seasonal=False, suppress_warnings=True)

            p, d, q = auto_arima_model.order
            model = ARIMA(train, order=(p, d, q))  # ARIMA(p, d, q)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            return forecast

        # SARIMA model
        def sarima_forecast(train, test):

            # Find optimal p, d, q, P, D, Q, and m values using Auto SARIMA
            auto_sarima_model = pm.auto_arima(train, seasonal=True, m=12, suppress_warnings=True)
            p, d, q = auto_sarima_model.order
            P, D, Q, m = auto_sarima_model.seasonal_order


            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))  # SARIMA(p, d, q)(P, D, Q, m)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            return forecast

        # Holt-Winters model
        def holt_winters_forecast(train, test):
            model = ExponentialSmoothing(train, seasonal_periods=12, trend='add', seasonal='add')
            model_fit = model.fit()
            forecast = model_fit.forecast(len(test))
            return forecast

        # Facebook Prophet model
        def prophet_forecast(train, test):
            model = Prophet()
            model.fit(train)
            future = model.make_future_dataframe(periods=len(test), freq='W')
            forecast = model.predict(future)
            forecast = forecast['yhat'][-len(test):].values
            return forecast
        
        for i in timeseries_key_values:
            df_prod1 = df1[df1['TimeSeries_Key']==i]
    
            print("----------RESULT FOR",i,"----------")

            # Convert 'calweek' column to datetime format
            df_prod1['calweek'] = pd.to_datetime(df_prod1['calweek'].astype(str) + '1', format='%Y%W%w')

            # Group data by 'calweek' and aggregate 'Quantity'
            ts_data = df_prod1.groupby('calweek')['Quantity'].sum().reset_index()

            # Rename columns for Prophet model
            ts_data.rename(columns={'calweek': 'ds', 'Quantity': 'y'}, inplace=True)

            # Split data into training and testing sets
            n_train = int(len(ts_data) * 0.8)  # 80% for training, 20% for testing
            train = ts_data.iloc[:n_train]
            test = ts_data.iloc[n_train:]

            # Perform forecasting using ARIMA
            arima_predictions = arima_forecast(train['y'], test['y'])

            # Perform forecasting using SARIMA
            sarima_predictions = sarima_forecast(train['y'], test['y'])

            # Perform forecasting using Holt-Winters
            holt_winters_predictions = holt_winters_forecast(train['y'], test['y'])
    

            # Perform forecasting using Facebook Prophet
            prophet_predictions = prophet_forecast(train, test)

            # Ensemble forecast combining ARIMA and Prophet
            ensemble_forecast = (arima_predictions + prophet_predictions) / 2.0

            # Calculate MAPE
            def calculate_mape(actual, forecast):
                return np.mean(np.abs((actual - forecast) / actual)) * 100

            # Calculate MAPE for ARIMA
            arima_mape = calculate_mape(test['y'], arima_predictions)

            # Calculate MAPE for SARIMA
            sarima_mape = calculate_mape(test['y'], sarima_predictions)

            # Calculate MAPE for Holt-Winters
            holt_winters_mape = calculate_mape(test['y'], holt_winters_predictions)

            # Calculate MAPE for Facebook Prophet
            prophet_mape = calculate_mape(test['y'], prophet_predictions)

            # Calculate MAPE for ensemble forecast
            ensemble_mape = calculate_mape(test['y'], ensemble_forecast)


            # Print MAPE values
            print("ARIMA - MAPE: {:.2f}%".format(arima_mape))
            print("SARIMA - MAPE: {:.2f}%".format(sarima_mape))
            print("Holt-Winters - MAPE: {:.2f}%".format(holt_winters_mape))
            print("Facebook Prophet - MAPE: {:.2f}%".format(prophet_mape))
            print("Ensemble Forecast - MAPE: {:.2f}%".format(ensemble_mape))
    
            #FORECASTINGGG

            # Split data into training and testing sets
            train = ts_data.iloc[:172]

            # Facebook Prophet model
            def prophet_forecast1(train):
                model = Prophet()
                model.fit(train)
                future = model.make_future_dataframe(periods=13, freq='W')
                forecast = model.predict(future)
                forecast = forecast['yhat'][-13:].values
                return forecast

            # ARIMA model
            def arima_forecast1(train):
                model = ARIMA(train, order=(1, 0, 0))  # ARIMA(p, d, q)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=13)
                return forecast
    
            # Perform forecasting using ARIMA
            arima_predictions = arima_forecast1(train['y'])

            # Perform forecasting using Facebook Prophet
            prophet_predictions = prophet_forecast1(train)

            # Ensemble forecast combining ARIMA and Prophet
            ensemble_predictions = np.array((arima_predictions + prophet_predictions) / 2.0)
            print(ensemble_predictions)

            # Generate date values for the forecasted period
            start_date = ts_data['ds'].iloc[-1]
            forecast_dates = pd.date_range(start=start_date, periods=len(test), freq='W').strftime('%Y-%m-%d')
            for j in range(13):
                results_df = pd.concat([results_df, pd.DataFrame({
                    'date': [forecast_dates[j]],
                    'TimeSeries_Key': [i],
                    'Ensemble_Predicted': ensemble_predictions[j],
                    'Prophet_Predicted': [prophet_predictions[j]]
            
            })], ignore_index=True)
            results_mape = pd.concat([results_mape, pd.DataFrame({
                'date': [forecast_dates[j]],
                'TimeSeries_Key': [i],
                'Prophet_MAPE': [prophet_mape],
                'ARIMA_MAPE': [arima_mape],
                'SARIMA_MAPE': [sarima_mape],
                'HoltWinters_MAPE': [holt_winters_mape],
                'Ensemble_MAPE': [ensemble_mape]
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
        final_pred_df = merged_df[['calweek', 'Cluster/Country/Region', 'SubCluster', 'brand', 'dc', 'TimeSeries_Key', 'Ensemble_Predicted']]

        # Concatenate the dataframes
        final_univariate_csv = pd.concat([final_df, final_pred_df], axis=0)
        #final_univariate_csv
         
        return final_univariate_csv
    
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
    final_csv.to_csv('final_univariate.csv', index=False)

