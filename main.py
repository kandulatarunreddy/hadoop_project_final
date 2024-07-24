from mrjob.job import MRJob
import csv
from io import StringIO
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import r2_score,make_scorer
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

class CSVMapperReducer(MRJob):

    def mapper(self, _, line):
        # Parse the CSV line
        rows = next(csv.reader(StringIO(line)))

        # # Convert the row to a DataFrame
        flights = pd.DataFrame([rows])
        flights.columns = ['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route', 'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops', 'Additional_Info', 'Price']

        # Removing empty valued rows from the data frame
        flights.dropna(inplace=True)

        flights['weekday'] = pd.to_datetime(flights['Date_of_Journey']).dt.day
        flights['month'] = pd.to_datetime(flights['Date_of_Journey']).dt.month
        flights['Dep_Time']=pd.to_datetime(flights['Dep_Time']).dt.hour
        flights['Additional_Info']=flights['Additional_Info'].str.replace('No info','No Info')
        flights['Duration']=flights['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
        flights['Duration']=pd.to_numeric(flights['Duration'])

        # Convert the cleaned data to a JSON-compatible representation
        json_data = flights.to_dict(orient='records')
        if len(json_data) >= 1:
            json_data = json_data[0]
        yield None, json.dumps(json_data)

    def reducer(self, _, values):
        test_file=pd.read_excel('C:/Users/tarun_kandula/OneDrive/Desktop/SUNY_NP/fall2023/data_science/project/data/Flight_Ticket_Participant_Datasets/Test_set.xlsx')
        # Create a list to store the final data
        final_data = []

        # Iterate over each JSON-compatible data emitted by the mapper
        for json_data in values:
            # Convert the JSON-compatible data back to a DataFrame
            if json_data is not None and len(json_data) > 0:
                df = pd.DataFrame.from_dict([json.loads(json_data)])

                # Apply the same condition as in the mapper
                df_cleaned = df.replace('', np.nan).dropna()

                if not df_cleaned.empty:
                    # Append the modified DataFrame to the final data list
                    final_data.append(df)

        # Concatenate all DataFrames into one final DataFrame
        if final_data:
            flights = pd.concat(final_data, ignore_index=True)

        # dropping cols from result data frame
        flights.drop(['Route','Arrival_Time','Date_of_Journey'],axis=1,inplace=True)

        var_mod = ['Airline','Source','Destination','Additional_Info','Total_Stops','weekday','month','Dep_Time']
        le = LabelEncoder()
        for i in var_mod:
            flights[i] = le.fit_transform(flights[i])

        flights=self.outlier(flights)
        x=flights.drop('Price',axis=1)#taking all the other columns except price 
        y=flights['Price']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)
        accuracies={}

        rfr=RandomForestRegressor(n_estimators=100)
        rfr.fit(x_train,y_train)
        features=x.columns
        importances = rfr.feature_importances_
        indices = np.argsort(importances)
        predictions=rfr.predict(x_test)

        accuracies['MAE']=metrics.mean_absolute_error(y_test, predictions)
        accuracies['MSE']=metrics.mean_squared_error(y_test, predictions)
        accuracies['RMSE']=np.sqrt(metrics.mean_squared_error(y_test, predictions))
        accuracies['r2_score']=metrics.r2_score(y_test, predictions)

        regg=[LinearRegression(),RandomForestRegressor(),SVR(),DecisionTreeRegressor(), Lasso()]
        mean=[]
        std=[]
        for i in regg:
            cvs=cross_val_score(i,x,y,cv=5,scoring=make_scorer(r2_score))
            mean.append(np.mean(cvs))
            std.append(np.std(cvs))
        
        for i in range(0, len(regg)):
            accuracies[regg[i].__class__.__name__]=mean[i]
        
        test_file['Date_of_Journey']=pd.to_datetime(test_file['Date_of_Journey'], format = "%d/%m/%Y")
        test_file['Dep_Time']=pd.to_datetime(test_file['Dep_Time'],format='%H:%M').dt.time
        test_file['Duration']=test_file['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
        test_file['Duration']=pd.to_numeric(test_file['Duration'])
        test_file['Dep_Time']=test_file['Dep_Time'].apply(lambda x:x.hour)
        test_file['Dep_Time']=pd.to_numeric(test_file['Dep_Time'])
        test_file["month"] = test_file['Date_of_Journey'].map(lambda x: x.month_name())
        test_file['weekday']=test_file[['Date_of_Journey']].apply(lambda x:x.dt.day_name())
        test_file['Additional_Info']=test_file['Additional_Info'].str.replace('No info','No Info')
        test_file.drop(['Date_of_Journey','Route','Arrival_Time'],axis=1,inplace=True)
        for i in var_mod:
            test_file[i]=le.fit_transform(test_file[i])
        test_file = test_file[['Airline', 'Source', 'Destination', 'Dep_Time', 'Duration', 'Total_Stops', 'Additional_Info', 'weekday', 'month']]
        test_price_predictions=rfr.predict(test_file)
        
        # Serializing json
        json_object = json.dumps(accuracies, indent=4)

        # Writing to accuracies.json
        local_output_path = "C:/Users/tarun_kandula/OneDrive/Desktop/SUNY_NP/fall2023/data_science/project/data/accuracies.json"
        with open(local_output_path, "w") as outfile:
            #print(f"Creating temp file at '{local_output_path}'")
            outfile.write(json_object)

        # Specify the desired HDFS path
        hdfs_output_path = "/dynamic_price_prediction/output/accuracies.json"
        # Upload the local CSV file to HDFS
        os.system(f'hadoop fs -rm -f {hdfs_output_path}')
        os.system(f'hadoop fs -put {local_output_path} {hdfs_output_path}')

        # try:
        #     os.remove(local_output_path)
        #     # print(f"Deleted temp file created at '{local_output_path}' successfully.")
        # except FileNotFoundError:
        #     print(f"File '{local_output_path}' not found.")
        # except Exception as e:
        #     print(f"An error occurred: {e}")


    def outlier(self, df):
        for i in df.describe().columns:
            Q1=df.describe().at['25%',i]
            Q3=df.describe().at['75%',i]
            IQR= Q3-Q1
            LE=Q1-1.5*IQR
            UE=Q3+1.5*IQR
            df[i]=df[i].mask(df[i]<LE,LE)
            df[i]=df[i].mask(df[i]>UE,UE)
        return df

    def split_duration(self, df_cleaned):
        # Assuming df_cleaned["Duration"] is a Series
        duration_series = df_cleaned["Duration"]

        # Check if the series has any values
        if not duration_series.empty:
            # Take the first non-empty value, split on 'h', and convert to integer
            first_value = duration_series.dropna().iloc[0]

            # Check if 'h' is present in the value
            if 'h' in str(first_value):
                d_hour = int(str(first_value).split('h')[0])
            else:
                d_hour = 0

            # Extract minutes
            if 'm' in str(first_value):
                d_min = int(str(first_value).split('m')[0].split()[-1])
            else:
                d_min = 0

            return [d_hour, d_min]
        else:
            return [0, 0]  # Return default values if the series is empty


if __name__ == '__main__':
    CSVMapperReducer.run()
