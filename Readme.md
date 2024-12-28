![Gizmo mascot](gizmo_mascot.png)  

# Gizmo Project  

![Python](https://img.shields.io/badge/Python-3.11-blue)  
![Conda](https://img.shields.io/badge/Conda-Environment-green)  
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20MacOS-orange)  
![License](https://img.shields.io/badge/License-MIT-brightgreen)  

The **Gizmo Project** automates machine learning workflows from start to finish. It cleans your data, trains models, compares them, and generates professional reports—all with just a few commands.  

---

## Table of Contents  
1. [What Gizmo Does](#what-gizmo-does)  
2. [Features](#features)  
3. [What You Get](#what-you-get)  
4. [Setup](#setup)  
5. [How to Use](#how-to-use)  
6. [Supported Models](#supported-models)  
7. [System Requirements](#system-requirements)  
8. [Contributing](#contributing)  
9. [License](#license)  

---

## What Gizmo Does  

1. **Data Preparation**: Cleans missing values, removes outliers, and generates useful features.  
2. **Model Training**: Builds multiple machine learning models automatically.  
3. **Result Comparison**: Calculates metrics like accuracy, precision, recall, and more.  
4. **Reporting**: Produces detailed PDF and Word documents with results and visuals.  

---

## Features  

- **End-to-End Automation**: Handles the entire machine learning pipeline.  
- **Interactive GUI**: Easy-to-use interface for data preparation, training, and evaluation.  
- **Multi-Model Training**: Supports models like XGBoost, Random Forest, Logistic Regression, and more.  
- **Resource Optimization**: Automatically adjusts system settings for best performance.  
- **Comprehensive Reporting**: Includes graphs, metrics, and summaries.  

---

## What You Get  

When you use Gizmo, it produces:  
1. **Reports**:  
   - Professional Word and PDF summaries.  
   - Includes visualizations like ROC curves and confusion matrices.  
2. **Predictions**:  
   - CSV files with model predictions.  
3. **Model Comparisons**:  
   - Metrics for each model, highlighting the best one.  

---

## Setup  

You can use Gizmo on both Windows and Linux/Mac.  

### Step 1: Clone the Repository  

```bash  
git clone https://github.com/sikamikanikoBG/gizmo.git  
cd gizmo  
```  

### Step 2: Create the Environment  

#### For Windows  
Run the `setup.bat` file:  
```cmd  
setup.bat  
```  

#### For Linux/Mac  
Run the `setup.sh` file:  
```bash  
bash setup.sh  
```  

### What These Scripts Do:  
- Create a **Conda environment** called `gizmo_env`.  
- Install all required Python libraries listed in `requirements.txt`.  

### Step 3: Activate the Environment  

#### On Windows:  
```cmd  
conda activate gizmo_env  
```  

#### On Linux/Mac:  
```bash  
conda activate gizmo_env  
```  

You’re now ready to use Gizmo!  

---

## How to Use  

### Example 1: Data Preparation  

Prepare your data using the `data_prep` module:  
```bash  
python main.py --project my_project --data_prep_module standard  
```  

- **What Happens**:  
  - Cleans your data and handles missing values.  
  - Creates new features like ratios and binned values.  
  - Saves the prepared dataset for training.  

---

### Example 2: Train Models  

Train models on your prepared data using the `train` module:  
```bash  
python main.py --project my_project --train_module standard
```  

- **What Happens**:  
  - Trains a Logistic Regression model.  
  - Saves the model, metrics, and results.  


---

### Compare Results  

After training, Gizmo will automatically compare all models and highlight the best one in the reports.  

---

### Using the GUI  

For an interactive experience:  
1. Launch the GUI:  
   ```bash  
   python gizmo_ui.py  
   ```  
2. Use the interface to prepare data, train models, and review results.  

---

## Supported Models  

Gizmo supports the following machine learning models:  
- **XGBoost**  
- **Random Forest**  
- **Decision Trees**  

These models are trained using optimized hyperparameters and evaluated across multiple metrics.  

---

## System Requirements  

### Minimum Requirements  
- **RAM**: 8 GB  
- **CPU**: Dual-core  
- **Disk Space**: 20 GB free  

### Recommended  
- **RAM**: 16 GB or more  
- **GPU**: NVIDIA 3090 or equivalent (for faster processing).  

---

## Contributing  

We welcome contributions! Here's how you can help:  
1. Fork this repository.  
2. Create a new branch for your changes:  
   ```bash  
   git checkout -b feature/your-feature-name  
   ```  
3. Submit a pull request with your improvements.  

---

## License  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  
