## Forecasting of slow moving SKUs

Random Samplers
- Ashish Dhiman
- Anshit Verma
- Yibei Hu


### Folder structure:
```
bestbuy
│   README.md
│   data.dvc   
│
└───notebooks
│   │
│   └───src/ Modules for implementations
│        │   ad_hmm.py #Module for HMM implementation
│        │   ad_stl_prophet.py #Module for Prophet/STL/MSTL implementation
│        │   utils.py #Utility function
│        │   ...
│
└───notebooks
│   │
│   └───ashish/ Test notebooks by Ashish | HMM/STL/Prophet
│   │    │   ...
│   │
│   └───ashish_validation_train/ Notebooks for training final models
│   │    │   file111.txt
│   │    │   file112.txt
│   │    │   ...
│   │    
│   └───yibei/ Test notebooks by Ashish | HMM/STL/Prophet
│        │   file111.txt
│        │   ...
│    
│
└───plots/ Folder for plots   
│   
└───Results/ Folder for RMSE and other results
```

### Folder structure: