### Experimental Pipeline

#### Notes:
This section documents the code used during the experiment. Please note the following points when reproducing the experiment:

1. To save storage space, many memory-intensive and reproducible duplicate contents have been removed, including the Data section in steps 2, 3, 4 and 5 (only the folder structure is retained in step 2). If you wish to regenerate the experimental data, refer to the 'Data Preparation' section.
2. The model code for comparative experiments can be downloaded from the corresponding websites listed in the references. Due to differences in datasets, minor adjustments were made to some comparative models during testing, including: correcting code errors (e.g., indentation styles), adapting to outdated library functions (e.g., RNATracker uses obsolete TensorFlow 1.x functions), etc. The comparative results are for reference only.

#### Data Preparation:
1. Navigate to the *Dataset Splitting* folder and run the `dataSplite` script to split the dataset, generating three dataset folders: `train`, `weight`, and `test`.
2. Place the generated dataset folders into the corresponding directories under *Feature Extraction\data*.
3. Perform feature extraction using the methods in *Feature Extraction\methods*.
4. Configure the target for feature fusion in *Feature Extraction\data\fusion.py* and execute the script to generate the `fusion` folder.
5. Move the contents of the `data` folder (including the `fusion` folder) to the `data` folders in steps 3, 4, and 5.

#### Python Library Installation:
Install the following Python libraries to configure the experimental environment:
1. matplotlib
2. numba
3. numpy
4. pandas
5. tqdm
6. scikit-learn (sklearn)
7. torch (PyTorch)
8. torchvision
9. imbalanced-learn (imblearn)
10. bayesian-optimization
11. shap
