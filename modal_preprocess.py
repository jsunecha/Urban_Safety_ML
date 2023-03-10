import os
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import time
import modal


image=(modal.Image.debian_slim()
    .apt_install("curl")
    .run_commands(
        "apt-get update",
        "curl -O https://raw.githubusercontent.com/TruthQuestWeb/ml-model/main/train.csv",
    ).pip_install(
        "pandas",
        "scikit-learn",
        "numpy",
        "requests",
        "googlemaps",
    ))

stub = modal.Stub(image=image)  

@stub.function(timeout=10000)
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
            print(df[c].unique())
            types = list(df[c].unique())
            colname = c


    
    print(types)
    #Drop all rows that have a type that is not in the list
    
    

    time.sleep(10)

    return df

@stub.function(timeout=10000,secret=modal.Secret.from_name("my-google-maps"))
def get_coords(row):
    import googlemaps

    #Load the API key from .env file
    gmaps = googlemaps.Client(key=os.environ["GOOGLE_MAPS"])

    geocode_result = gmaps.geocode(str(row[-3])+","+str(row[-2])+","+str(row[-1]))
    lat = geocode_result[0]["geometry"]["location"]["lat"]
    lng = geocode_result[0]["geometry"]["location"]["lng"]

    row.append(str(lat))
    row.append(str(lng))

    return row


@stub.local_entrypoint()
def main():
    sanjose = pd.read_csv("sanjose_policecalls2023.csv")
    sanjose = retain_columns.call(sanjose)

    #Convert df to list
    list_sanjose = sanjose.values.tolist()

    # #Adding the new columns
    columns_sanjose = sanjose.columns.tolist()
    columns_sanjose.append("LAT")
    columns_sanjose.append("LNG")

    templist = list()

    #Find the biggest whole number that divides evenly, but is smaller or equal to 50 for the number len(list_sanjose)
    #This is to avoid overloading the API
    counter = 1
    for i in range(50, 0, -1):
        if len(list_sanjose)%i == 0:
            counter = i
            break

    for x in range(0, len(list_sanjose)-counter, counter):
        templist += list(get_coords.map(list_sanjose[x:x+counter]))
        print(len(templist))
        time.sleep(1)
    
    #print(len(templist[0]))
    time.sleep(3)

    sanjose = pd.DataFrame(templist, columns=columns_sanjose)
        

    print(sanjose.columns)

    #Print how many rows and columns
    print(sanjose.shape)

    #Display the first 5 rows
    print(sanjose.head())

    #Save the dataframe to a csv file
    sanjose.to_csv("updated_sanjose_policecalls2023.csv", index=True)

