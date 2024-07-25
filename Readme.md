WIP :)

For building sphinx look in ``./conda_envs``

For building sphinx doc look in ``./docs/auto_make_html.sh``

### TODO 
- Add documentation on how to use main.py with params. Quick example:

  - ``python main.py --project bg_stage2 --data_prep_module standard``
  - ``python main.py --project bg_stage2 --train_module standard``

- Basic commands example:
  - ``python main.py --project bg_stage2 --data_prep_module standard``
  - ``python main.py --project bg_stage2 --train_module standard``
  - ``python main.py --project bg_stage2 --eval_module standard --session "TRAIN_<PROJECT_NAME_AND_DATE>_no_tag"``
    - Eval example: 
      - ``python main.py --project bg_stage2 --eval_module standard --session "TRAIN_bg_stage2_2024-06-18 13:12:00.802211_no_tag"``

- `src.classes.BaseModeller.BaseModeller.model_fit` throws an error if we have multilabel classification task
and using AUC evaluation. XGBoost documentation says:
  - `auc`: Receiver Operating Characteristic Area under the Curve. Available for classification and learning-to-rank tasks.
    - When used with binary classification, the objective should be binary:logistic or similar functions that work on probability.
    - **When used with multi-class classification**, objective should be `multi:softprob` instead of `multi:softmax`,
    as the latter doesn’t output probability. Also the AUC is calculated by 1-vs-rest with reference
    class weighted by class prevalence.
     
- ML Flow uploads model artifacts, but other metadata files are being blocked
 
- Docx output has about 30% correctly printed graphs. Possible causes:
  - Since AUC is still not implemented correctly, some graphs could be for AUC and thus not plotted
  - In the code for plotting graphs there are if statements checking if we are plotting a model with multiclass
    classification accompanied by comments about wondering if multiclass plotting works
  > Code for plotting is in: `src/functions/evaluation.py`

- More debugging is needed to make sure that multiclass classification actually works.
In many places in the package there are constant checks if we are doing mult. cls.
Those checks lead to running code designed for mult. cls., which is not extensively
tested.

- Vera mentioned that when choosing to *not* use undersampling on the dataset after data cleaning and feature filtering 
  we are left in an empty dataset. Possible causes:
  - Correlation score is too low
  - P-value is too high

- Debug prints have to be removed for the final version

- All erros can be found in ./logs

### Progress
- Eval now works, but the displayed graphs are missing
- Eval had some issues with getting the right session name. Now that's fixed. 
  > Reminder: related classes and functions are SessionManager; word_merge, etc


### Multiclass predictions with Gizmo

Now it is possible to build a multiclass models with Scoring Gizmo.
A multiclass model is one that targets more than two classes in its criterion (label / target) variable.

There's nothing special / different in using multiclass modeling: you specify the criterion variable in the project's params file, and the Scoring Gizmo package takes care of processing the data.
In the evaluation session it will create ROC graphs that capture all classes' performance.


There are three main differences between the binary classification case and the multiclass case:
- Only XGBoost is used right now. It is possible to use other algorithms, by means of creating one-vs-rest datasets, but that would require significant re-engineering effort for the whole package.
- Feature selection: In the binary classification case, feature selection is mainly done by calculating correlations between the predictors and the target. In the case of multiclass criterion variable we use **predictive power score**  to do feature selection. This methodology results in **far less features selected**, but it is possible to capture non-linear and/or asymetrical relationships between the predictor variables. In the case of correlation, it will always capture linear and symetrical relationships only.
- Probabilities and bands. As we have multiple classes, defining cut-offs for those classes is hard, and it is non-trivial to implement in a generic way.