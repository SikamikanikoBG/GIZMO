.. Scoring GIZMO documentation master file, created by
sphinx-quickstart on Thu Mar 11 10:25:48 2021.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

Getting ready
=========================================

Sessions
____________________________________________________________________________________________

The following sessions are available in Gizmo:
 1. **create** session - This session creates all needed folders in order Gizmo to operate
 2. **load** session - This session is loading the source file, cleaning the data based on the parameters that are in params file, feature engineering and selection procedures and finally stores the result file in output folder.
 3. **train** session - This session splits the data into train, test and 3 temporal validation samples in order to be used for model evaluation. After runs XGBoost, Random Forest and Decision tree methods to fit Training sample and applies the model on test, t1, t2 and t3 samples to validate. The 5 datasets are saved in session folder including additional information for the features.
 4. **eval** session - Scores evaluation. All scores built during the train session are being described in Word document file - each method in a separate file. The aim is to support the Data Scientist in their decision how to tweak the laod and/or training session in order to improve the results as well to automate the process of score documentation

Parametrization
____________________________________________________________________________________________

In *params* folder Gizmo will look for a json file for each project.
All parameters should be described in order to improve the scoring results.

Content of the param file:

 1. **Dependent column** - "criterion_column": "Criterion_6M" - we specify the column to be used for model classification "Criterion_6M"
 2. How Gizmo should **treat missing data**
      * column_mean - calculates column mean and fills all missing values with it
      * median - calculates column median and fills all missing values with it
      * delete - simply deletes all rows with missing values
      * other value - will be used to replace the missing value with it. For example 0 - this will fill the missing values with 0 number
 3. **Observation date** - observation_date_column": "MonthEntry" will specify column MonthEntry as the observation date
 4. **Temporal validation periods** - "t1df": 201907, "t2df": 201908, "t3df": 201909,
 5. **Periods to exclude** from the analysis - "periods_to_exclude": [201910] will exclude October 2019. If you dont want to exclude anything just keep the []
 6. **Exclude columns** - "columns_to_exclude": ["DIFF_MM_LD_SEDCREAT"] will exclude DIFF_MM_LD_SEDCREAT column. To add more, just separate them with "," and include them in '"'.
 7. **LR features** - place list of features if you want to run Logistic regression with them instead of brute force.
 8. **lr features to include** - place list of features if you want to run Logistic regression with them. Gizmo will use them + will add to them more random columns in order to try brute force modelling. In order to work - lr_features must not be specified in the param file.
 9. **trees features to exclude**  - Specify some final features if you want to exclude them from the tree models (XGB, RF, DT). Useful when you review the stability graphs in the documentation and some features look unstable.
 10. **Cut offs**  - write the bands that you want to have in the documentation based on the probability. example [0,0.5,1] will create 2 bands from 0 to 0.5 and 0.5 to 1
 11. **Under sampling** - if you want to perform undersampling after loading the data - specify the strategy between 0.1 and 1. This is the ratio between positive and negative cases. Example: Strategy 1 means 50/50 dataset as a result. If you don't want undersampling - place empty quotes: ""

Running the package
____________________________________________________________________________________________

to run a Gizmo session you have to run main.py:

*Example:*
``python3 main.py --arguments``

Arguments
----------
 * run
    expects 4 values to run the session:
        * create - for create session
        * load - for load session
        * train - for train session
        * eval - for evaluation session

    *Example:*
    ``--run create``

 * project
    specifies which project to run. the name that you provide here should match the name in the folders - input_data, output_data as well a param file containing the name of the project in the params folder.
    *Example:*
    ``--project hu_amicable``

 * session - specify the name of the train session that you want to evaluate.

   *Example:*
   ``--session Train_blah_blaj``

* tag - specify a key name if you want to add it to the session folder to be more convenient after to compare results

   *Example:*
   ``--tag additional_periods_excluded_to_see_improved_results``

Running
-------
 1. Create session requires arguments: run, project

    *Example:* ``python3 main.py --run create --project hu_amicable``

 2. Load session requires arguments: run, project, tag(optional)

    *Example:* ``python3 main.py --run load --project hu_amicable``

 3. Train session requires arguments: run, project, tag(optional)

    *Example:* ``python3 main.py --run train --project hu_amicable``

 4. Eval session requires arguments: run, project, session, tag(optional)

    *Example:* ``python3 main.py --run eval --project hu_amicable --session train_blah_blah``

