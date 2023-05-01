# Urban_Safety_ML

## Introduction

## Installation

``gdal`` is required to run the code.

To get the Python libraries required to run the code, run the following command:

``pip install -r requirements.txt``

## Files

fixed_preprocessing assumes that you have x and y co-ordinates.
modal_preprocess is a cloud(modal) version of preprocess.py file and they both do the same thing, which is get x and y co-ordinates using google maps API.

We trained LSTM initially but we weren't getting promising results(max 56% accuracy), so we moved on to Feed Forward Neural Network.

We will show an html page which takes lat, long and time of day as input and predicts the most probable type of crime that can occur based on these and historic data.
