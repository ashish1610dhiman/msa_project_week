## Forecasting of slow moving SKUs

Random Samplers
- Ashish Dhiman
- Anshit Verma
- Yibei Hu


### Folder structure:
```
bestbuy
│   README.md
│   data.dvc #DVC file for data  
│   bestbuy_env.yml #Conda env file  
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
│   │    │   b.run_hmm_final1.ipynb #Train and forecast from HMMM
│   │    │   c.run_stl_prophet_new.ipynb #Train and forecast from Prophet/ STL/MSTL
│   │    │   ...
│   │    
│   └───yibei/ Test notebooks by Yibei | HMM/STL/Prophet
│        │   HW_final.ipynb #Notebook to train Holt Winters Exp smoothing and Null model
│        │   ...
│    
│
└───plots/ Folder for plots   
│   
└───Results/ Folder for RMSE and other results
```

### Code Transition documents:
As listed in the folder structure above, these are the main codes and their description:
- notebooks/ashish/validation_train/b.run_hmm_final1.ipynb: The code is used to implement and train HMM models. It calls upon the ad_hmm.py module in the src folder.
- notebooks/ashish/validation_train/c.run_stl_prophet_new.ipynb: The code is used to implement and train Prophet/ STL/MSTL models. It calls upon the run_stl_prophet_new.py module in the src folder.
- notebooks/yibei/c.HW_final.ipynb: The code is used to implement and train Holt Winters Exp smoothing and the Null model.

### Random nuances:
- Statmodels beta realease has been used
- Joblib is used
- some codes are run on kaggle, to prevent personal laptop use