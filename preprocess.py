import pandas as pd
import numpy as np
#Use .env file to store API key
from dotenv import load_dotenv
import os

import googlemaps

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import time


def retain_columns(df):
    col = ("ID", "INCIDENT", "DATE", "TIME", "LOCATION", "ADDRESS","TYPE", "CATEGORY", "DESCRIPTION", "CITY", "STATE")
    for c in df.columns:
        c = c.upper()
        #If the string has any of the words in col, keep it
        if any(x in c for x in col):
            continue
        else:
            df = df.drop(c, axis=1)
    #Find the column with TYPE within the name, and find all unique values
    types = list()
    colname = ""
    for c in df.columns:
        c = c.upper()
        if "TYPE" in c and "CODE" not in c:
            #print(df[c].unique())
            types = list(df[c].unique())
            colname = c

    #Drop all rows that have a type that is not in the list
    typeslist = ["BURGLARY", "DISTURBANCE", "THEFT", "STOLEN", "SUSPICIOUS", "ASSAULT", "MISCHIEF", "BATTERY", "THREATS", "STAB", "FIGHT", "ROBBERY", "DISTURBANCE", "HATE", "KIDNAPPING"]
    for t in types:
        #If t is not similar to typeslist, drop it
        if not any(x in t for x in typeslist):
            #remove from types
            types.remove(t)
    
    #Go through the dataframe and drop all rows that have a type that is not in the list
    for index, row in df.iterrows():
        if row[colname] not in types:
            df = df.drop(index)
            print(index)

    return df


def get_coords(row):
    import googlemaps

    #Load the API key from .env file
    gmaps = googlemaps.Client(key=os.getenv("GOOGLE"))

    geocode_result = gmaps.geocode(row['ADDRESS']+","+row['CITY']+","+row['STATE'])
    lat = geocode_result[0]["geometry"]["location"]["lat"]
    lng = geocode_result[0]["geometry"]["location"]["lng"]

    row['LAT'] = str(lat)
    row['LNG'] = str(lng)

    print(row)
    return row

def main():
    df = pd.read_csv("sanjose_policecalls2023.csv")
    df = retain_columns(df)
    print(df.shape)


    for index, row in df.iterrows():

        #Get the coordinates
        row = get_coords(row)

        df.at[index, 'LAT'] = row['LAT']
        df.at[index, 'LNG'] = row['LNG']
        #Wait 0.5 seconds to avoid overloading the API
        print(df.columns)
        time.sleep(0.5)
        

    print(df.columns)

    #Print how many rows and columns
    print(df.shape)

    #Display the first 5 rows
    print(df.head())


    #Save the dataframe to a csv file
    df.to_csv("updated_sanjose_policecalls2023.csv", index=True)




main()
