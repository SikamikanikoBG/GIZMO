.. Scoring GIZMO documentation master file, created by
sphinx-quickstart on Thu Mar 11 10:25:48 2021.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

Functions in the modelling part
=========================================

XGBoost modelling
____________________________________________________________________________________________

def xgb(*df, criterion, test_X, test_y, predict_only_flag, model_to_predict, final_features*)
 * df - this is the data frame on which the model to be trained and tested - train df
 * criterion - this is column/array that we are predicting for train df
 * test_X - test df
 * test_y - criterion for test df
 * predict_only_flag - if "yes" then the function will not fit any models, but will expect an object with the model to use if to predict
 * model_to_predict - used model to run predictions when predict_only_flag is "yes"
 * final_features - list with features to be fit/used for predictions

Returns:
 * returns back the data frame that was passed, but with added new columns
 * model (if new model was fit)
 * accuracy score
 * precision score
 * number of features in the model
 * dataframe with feature importance if it was a train call (predict_only_flag!="yes")


*Example:*

``X_test, _, x_test_ac, x_test_auc, x_test_prec, x_test_nb_features, _ = functions.xgb(df=X_test, criterion=y_test,
model_to_predict=model_train, predict_only_flag='yes', test_X=None, test_y=None,final_features=final_features)``