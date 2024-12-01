## Structure of the random forest branch:     
### Models:    
Contains 4 version of random forest models:    
base_model.py    
- Random forest without oversampling or feature extraction   
feature_select_only.py   
- Feature selection without oversampling    
featureselect+oversampling.py   
- Feature selection and oversampling    
oversampling_only.py
- oversampling only without feature selection

### Model_results     
Contains results from the 4 models     
base_model_results.txt    
feature_select_only_results.txt     
featureselect+oversampling_results.txt      
oversampling_only_results.txt     

### Misclassfication_report
Contains the analysis of misclassified cases and has 3 files:    
misclassification_report.py     
- Prints the most misclassified case as well as the LIME plot for said case   
Lime_actual3_predicted4.png
- The LIME plot saved as a png
misclassified_instances.csv
- All misclassified instances saved as a csv

### tuning_results
Contains all the tuning results as csv files from the 4 models

## Reproducing Results

### Step 1: Train and Evaluate the Model
Run `base_model.py` or any of the files in the Models folder to train the random forest model and evaluate its performance:
```bash
python base_model.py
```

### Step 2: Generate LIME Explanations
Run `misclassification_report.py` in the Misclassification_report folder to generate explanations for correctly classified and misclassified instances:
```bash
python misclassification_report.py
```

### Step 3: Visualize Feature Importances
The script `misclassification_report.py` generates a bar chart (`Lime_actual3_predicted4.png`) comparing the feature importances for correctly classified and misclassified instances. Open the file to view the visualization.

