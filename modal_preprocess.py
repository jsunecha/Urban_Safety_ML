import os
import time

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
    col = ("ID", "INCIDENT", "DATE", "TIME", "LOCATION", "ADDRESS","REPORT", "TYPE", "CATEGORY", "DESCRIPTION", "CITY", "STATE", "LATITUDE","LONGITUDE")
    for c in df.columns:
        temp_c = c.upper()

        #If the string has any of the words in col, keep it
        if any(x in temp_c for x in col):
            continue
        else:
            df = df.drop(c, axis=1)
    
    #Find the column with TYPE within the name, and find all unique values
    types = list()
    colname = ""
    for c in df.columns:
        typeslist = ["BURGLARY", "THEFT", "STOLEN", "SUSPICIOUS", "ASSAULT", "MISCHIEF", "BATTERY", "THREATS", "STAB", "FIGHT", "ROBBERY", "DISTURBANCE", "HATE", "KIDNAPPING"]
        #Iterate through the first 100 rows to see if they have anything from typeslist
        for i in range(0, 100):
            try:
                temp_t = df[c].iloc[i].upper()
                if any(x in temp_t for x in typeslist):
                    types = list(df[c].unique())
                    colname = c
                    break
            except:
                continue
    
        #Drop all rows that have a type that is not in the list
    final_types = list()
    for t in types:
        #If t is not similar to typeslist, drop it
        print("TEMP", temp_t)
        temp_t = t.upper()
        if any(x in temp_t for x in typeslist):
            #remove from types
            final_types.append(t)
    
    print("Remaining Types")
    print(final_types)
    #Go through the dataframe and drop all rows that have a type that is not in the list
    for index, row in df.iterrows():
        if row[colname] not in final_types:
            df = df.drop(index)
            
    #Drop all rows that have a NaN value
    df = df.dropna()

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

    # record the start time
    start_time = time.time()

    distributed_df = []

    original_df = pd.read_csv("san_francisco.csv")
    print(original_df.shape)
    time.sleep(3)
    #Split the length of the dataframe into 100 parts
    for i in range(0, 100):
        distributed_df.append(original_df.iloc[i*len(original_df)//100:(i+1)*len(original_df)//100])


    new_distributed_df = []
    for results in retain_columns.map(distributed_df):
        new_distributed_df.append(results)


    #Add all the dataframes together from distributed_df list
    df = pd.concat(new_distributed_df, ignore_index=False)
    print(df.shape)
    df.to_csv("updated_sanfrancisco_policecalls2023.csv", index=True)


    # record the end time
    end_time = time.time()

    # calculate the elapsed time
    elapsed_time = end_time - start_time

    # print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
    return

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


