# Preprocessing
Creating the preprocessing pipeline for the data.

### Column Name
- [ ] Partition the incoming csv files into parquet files
    - [ ] Create a folder for each csv file
    - [ ] OPTIONAL: Store the data within a data lake
- [ ] Clean up the CALL_TYPES column 
    - [ ] Remove unnecessary types in the CALL_TYPES column
    - [ ] Remove unnecessary rows 
- [ ] Clean up the Addresses column
    - [ ] Add lat and long to utilize in machine learning

- [ ] Implement a modal serverless architecture that handles preprocessing
    - [ ] Create a lambda function that handles the preprocessing
    - [ ] Create a lambda function that handles the data lake


### Completed Column ✓
- [x] Completed task title  