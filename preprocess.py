import pandas as pd
import numpy as np
#Use .env file to store API key
from dotenv import load_dotenv
import os

import googlemaps

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import time


#Load the API key from .env file
gmaps = googlemaps.Client(key="""KEY""")

def retain_columns(df):
    col = ("ID", "INCIDENT", "DATE", "TIME", "LOCATION", "ADDRESS","CALL_TYPE", "CATEGORY", "DESCRIPTION", "CITY", "STATE")
    for c in df.columns:
        c = c.upper()

        #If the string has any of the words in col, keep it
        if any(x in c for x in col):
            continue
        else:
            df = df.drop(c, axis=1)
    return df


def get_coords(row):
    import googlemaps

    #Load the API key from .env file
    gmaps = googlemaps.Client(key="AIzaSyAasL8vUMb6lrZctKylJc9AMQLFpZddrrQ")

    geocode_result = gmaps.geocode(row['ADDRESS']+","+row['CITY']+","+row['STATE'])
    lat = geocode_result[0]["geometry"]["location"]["lat"]
    lng = geocode_result[0]["geometry"]["location"]["lng"]

    row['LAT'] = str(lat)
    row['LNG'] = str(lng)

    print(row)
    return row

def main():
    sanjose = pd.read_csv("sanjose_policecalls2023.csv")
    sanjose = retain_columns(sanjose)
    print(sanjose.shape)


    for index, row in sanjose.iterrows():

        #Get the coordinates
        row = get_coords(row)

        sanjose.at[index, 'LAT'] = row['LAT']
        sanjose.at[index, 'LNG'] = row['LNG']
        #Wait 0.5 seconds to avoid overloading the API
        print(sanjose.columns)
        time.sleep(0.5)

        if index == 5:
            break
        
        

    print(sanjose.columns)

    #Print how many rows and columns
    print(sanjose.shape)

    #Display the first 5 rows
    print(sanjose.head())


    #Save the dataframe to a csv file
    sanjose.to_csv("updated_sanjose_policecalls2023.csv", index=True)




main()
