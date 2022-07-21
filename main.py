import argparse
import sys
import warnings
import logging
from colorama import Fore
from colorama import Style
from importlib import import_module

from src.functions.printing_and_logging_functions import print_and_log

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # -- session start --

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    # Required positional argument
    #parser.add_argument('--run', type=str, help='After run write which command to execute: load, train, eval')
    parser.add_argument('--tag', type=str, help='Tag the training_flows session. Optional')
    parser.add_argument('--project', type=str,
                        help='name of the project. Should  the same as the input folder and the param file.')
    parser.add_argument('--session', type=str, help='Train session folder to generate evaluation docx')
    parser.add_argument('--model', type=str, help='Model to evaluate. xgb, rf, dt, lr')

    parser.add_argument('--data_prep_module', type=str, help='Data prep module to run')
    parser.add_argument('--train_module', type=str, help='Training module to run')
    parser.add_argument('--h', type=str, help='You need help...')
    args = parser.parse_args()

    help_need = args.h
    if help_need:
        print_and_log(parser.parse_known_args(), "RED")
        sys.exit()

    if args.data_prep_module:
        data_prep_module = args.data_prep_module.lower()
        module_lib = import_module(f'src.data_prep_flows.{data_prep_module}')
        data_prep = module_lib.ModuleClass(args=args)
        data_prep.run()
        data_prep.run_time_calc()
        print_and_log(data_prep.run_time, "")
    elif args.train_module:
        train_module_arg = args.train_module.lower()
        module_lib = import_module(f'src.training_flows.{train_module_arg}')
        train_module = module_lib.ModuleClass(args=args)
        train_module.run()
        train_module.run_time_calc()
        print_and_log(train_module.run_time, "")


"""
    def modeller(input_data_project_folder):
        
        
        models = pd.DataFrame(
            columns=['Method', 'AccuracyScore', 'AUC', 'PrecisionScore', 'Recall', 'F1', 'DataSet', 'NbFeatures'])

        

        # XGB
        print(Fore.YELLOW + '\n *** Starting XGB modeller *** \n' + Style.RESET_ALL)
        logging.info('\n *** Starting XGB modeller ***')
        X_train, model_train, x_train_ac, x_train_auc, x_train_prec, x_train_nb_features, feature_importance, cut_points_train, x_train_recall, x_train_f1 = src.xgb(
            df=X_train, criterion=y_train,
            df_us=X_train_us, criterion_us=y_train_us,
            test_X_us=X_test_us, test_y_us=y_test_us,
            model_to_predict=None,
            predict_only_flag='no', test_X=X_test, test_y=y_test, final_features=final_features, cut_points_train=None,
            cut_offs=cut_offs, params=params)
        X_test, _, x_test_ac, x_test_auc, x_test_prec, x_test_nb_features, _, _, x_test_recall, x_test_f1 = src.xgb(
            df=X_test, criterion=y_test,
            df_us=X_train_us,
            criterion_us=y_train_us,
            test_X_us=X_test_us,
            test_y_us=y_test_us,
            model_to_predict=model_train,
            predict_only_flag='yes',
            test_X=None, test_y=None,
            final_features=final_features,
            cut_points_train=cut_points_train,
            cut_offs=cut_offs,
            params=params)
        t1df, _, t1df_ac, t1df_auc, t1df_prec, t1df_nb_features, _, _, t1df_recall, t1df_f1 = src.xgb(df=t1df,
                                                                                                      criterion=t1df[
                                                                                                                criterion_column],
                                                                                                      df_us=X_train_us,
                                                                                                      criterion_us=y_train_us,
                                                                                                      test_X_us=X_test_us,
                                                                                                      test_y_us=y_test_us,
                                                                                                      model_to_predict=model_train,
                                                                                                      predict_only_flag='yes',
                                                                                                      test_X=None,
                                                                                                      test_y=None,
                                                                                                      final_features=final_features,
                                                                                                      cut_points_train=cut_points_train,
                                                                                                      cut_offs=cut_offs,
                                                                                                      params=params)
        t2df, _, t2df_ac, t2df_auc, t2df_prec, t2df_nb_features, _, _, t2df_recall, t2df_f1 = src.xgb(df=t2df,
                                                                                                      criterion=t2df[
                                                                                                                criterion_column],
                                                                                                      df_us=X_train_us,
                                                                                                      criterion_us=y_train_us,
                                                                                                      test_X_us=X_test_us,
                                                                                                      test_y_us=y_test_us,
                                                                                                      model_to_predict=model_train,
                                                                                                      predict_only_flag='yes',
                                                                                                      test_X=None,
                                                                                                      test_y=None,
                                                                                                      final_features=final_features,
                                                                                                      cut_points_train=cut_points_train,
                                                                                                      cut_offs=cut_offs,
                                                                                                      params=params)
        t3df, _, t3df_ac, t3df_auc, t3df_prec, t3df_nb_features, _, _, t3df_recall, t3df_f1 = src.xgb(df=t3df,
                                                                                                      criterion=t3df[
                                                                                                                criterion_column],
                                                                                                      df_us=X_train_us,
                                                                                                      criterion_us=y_train_us,
                                                                                                      test_X_us=X_test_us,
                                                                                                      test_y_us=y_test_us,
                                                                                                      model_to_predict=model_train,
                                                                                                      predict_only_flag='yes',
                                                                                                      test_X=None,
                                                                                                      test_y=None,
                                                                                                      final_features=final_features,
                                                                                                      cut_points_train=cut_points_train,
                                                                                                      cut_offs=cut_offs,
                                                                                                      params=params)

        models = models.append(
            {'Method': 'xgb', 'AccuracyScore': x_train_ac, 'AUC': x_train_auc, 'PrecisionScore': x_train_prec,
             'Recall': x_train_recall, 'F1': x_train_f1,
             'DataSet': 'X_train', 'NbFeatures': x_train_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'xgb', 'AccuracyScore': x_test_ac, 'AUC': x_test_auc, 'PrecisionScore': x_test_prec,
             'Recall': x_test_recall, 'F1': x_test_f1,
             'DataSet': 'X_test', 'NbFeatures': x_test_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'xgb', 'AccuracyScore': t1df_ac, 'AUC': t1df_auc, 'PrecisionScore': t1df_prec, 'Recall': t1df_recall,
             'F1': t1df_f1,
             'DataSet': 't1df', 'NbFeatures': t1df_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'xgb', 'AccuracyScore': t2df_ac, 'AUC': t2df_auc, 'PrecisionScore': t2df_prec, 'Recall': t2df_recall,
             'F1': t2df_f1,
             'DataSet': 't2df', 'NbFeatures': t2df_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'xgb', 'AccuracyScore': t3df_ac, 'AUC': t3df_auc, 'PrecisionScore': t3df_prec, 'Recall': t3df_recall,
             'F1': t3df_f1,
             'DataSet': 't3df', 'NbFeatures': t3df_nb_features},
            ignore_index=True)
        pickle.dump(model_train, open(session_id_folder + '/xgb/model_train.pkl', 'wb'))
        feature_importance.to_csv(session_id_folder + '/xgb/feat_importance.csv', index=False)
        print(f"\t Feature importance: {feature_importance}")
        logging.info(f"Feature importance: {feature_importance}")
        src.correlation_matrix(X=X_train[feature_importance['columns'].unique().tolist()], y=None, flag_matrix='all',
                               input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='xgb',
                               flag_raw='')
        src.correlation_matrix(
            X=X_train[src.raw_features_to_list(feature_importance['columns'].unique().tolist())], y=None,
            flag_matrix='all',
            input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='xgb', flag_raw='yes')

        # Random Forest
        print(Fore.GREEN + '\n *** Starting Random Forest *** \n' + Style.RESET_ALL)
        logging.info('*** Starting Random Forest ***')
        X_train, model_train, x_train_ac, x_train_auc, x_train_prec, x_train_nb_features, feature_importance, x_train_recall, x_train_f1 = src.rand_forest(
            df=X_train, criterion=y_train, df_us=X_train_us,
            criterion_us=y_train_us,
            test_X_us=X_test_us,
            test_y_us=y_test_us,
            model_to_predict=None,
            predict_only_flag='no', test_X=X_test, test_y=y_test, final_features=final_features, cut_offs=cut_offs,
            params=params)
        X_test, _, x_test_ac, x_test_auc, x_test_prec, x_test_nb_features, _, x_test_recall, x_test_f1 = src.rand_forest(
            df=X_test,
            criterion=y_test,
            df_us=X_train_us,
            criterion_us=y_train_us,
            test_X_us=X_test_us,
            test_y_us=y_test_us,
            model_to_predict=model_train,
            predict_only_flag='yes',
            test_X=None,
            test_y=None,
            final_features=final_features,
            cut_offs=cut_offs,
            params=params)
        t1df, _, t1df_ac, t1df_auc, t1df_prec, t1df_nb_features, _, t1df_recall, t1df_f1 = src.rand_forest(df=t1df,
                                                                                                           criterion=
                                                                                                                 t1df[
                                                                                                                     criterion_column],
                                                                                                           df_us=X_train_us,
                                                                                                           criterion_us=y_train_us,
                                                                                                           test_X_us=X_test_us,
                                                                                                           test_y_us=y_test_us,
                                                                                                           model_to_predict=model_train,
                                                                                                           predict_only_flag='yes',
                                                                                                           test_X=None,
                                                                                                           test_y=None,
                                                                                                           final_features=final_features,
                                                                                                           cut_offs=cut_offs,
                                                                                                           params=params)
        t2df, _, t2df_ac, t2df_auc, t2df_prec, t2df_nb_features, _, t2df_recall, t2df_f1 = src.rand_forest(df=t2df,
                                                                                                           criterion=
                                                                                                                 t2df[
                                                                                                                     criterion_column],
                                                                                                           df_us=X_train_us,
                                                                                                           criterion_us=y_train_us,
                                                                                                           test_X_us=X_test_us,
                                                                                                           test_y_us=y_test_us,
                                                                                                           model_to_predict=model_train,
                                                                                                           predict_only_flag='yes',
                                                                                                           test_X=None,
                                                                                                           test_y=None,
                                                                                                           final_features=final_features,
                                                                                                           cut_offs=cut_offs,
                                                                                                           params=params)
        t3df, _, t3df_ac, t3df_auc, t3df_prec, t3df_nb_features, _, t3df_recall, t3df_f1 = src.rand_forest(df=t3df,
                                                                                                           criterion=
                                                                                                                 t3df[
                                                                                                                     criterion_column],
                                                                                                           df_us=X_train_us,
                                                                                                           criterion_us=y_train_us,
                                                                                                           test_X_us=X_test_us,
                                                                                                           test_y_us=y_test_us,
                                                                                                           model_to_predict=model_train,
                                                                                                           predict_only_flag='yes',
                                                                                                           test_X=None,
                                                                                                           test_y=None,
                                                                                                           final_features=final_features,
                                                                                                           cut_offs=cut_offs,
                                                                                                           params=params)

        models = models.append(
            {'Method': 'rf', 'AccuracyScore': x_train_ac, 'AUC': x_train_auc, 'PrecisionScore': x_train_prec,
             'Recall': x_train_recall, 'F1': x_train_f1,
             'DataSet': 'X_train', 'NbFeatures': x_train_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'rf', 'AccuracyScore': x_test_ac, 'AUC': x_test_auc, 'PrecisionScore': x_test_prec,
             'Recall': x_test_recall, 'F1': x_test_f1,
             'DataSet': 'X_test', 'NbFeatures': x_test_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'rf', 'AccuracyScore': t1df_ac, 'AUC': t1df_auc, 'PrecisionScore': t1df_prec, 'Recall': t1df_recall,
             'F1': t1df_f1,
             'DataSet': 't1df', 'NbFeatures': t1df_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'rf', 'AccuracyScore': t2df_ac, 'AUC': t2df_auc, 'PrecisionScore': t2df_prec, 'Recall': t2df_recall,
             'F1': t2df_f1,
             'DataSet': 't2df', 'NbFeatures': t2df_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'rf', 'AccuracyScore': t3df_ac, 'AUC': t3df_auc, 'PrecisionScore': t3df_prec, 'Recall': t3df_recall,
             'F1': t3df_f1,
             'DataSet': 't3df', 'NbFeatures': t3df_nb_features},
            ignore_index=True)
        pickle.dump(model_train, open(session_id_folder + '/rf/model_train.pkl', 'wb'))
        feature_importance.to_csv(session_id_folder + '/rf/feat_importance.csv', index=False)
        logging.info(f"Feature importance: {feature_importance}")
        print(f"\t Feature importance: {feature_importance}")
        src.correlation_matrix(X=X_train[feature_importance['columns'].unique().tolist()], y=None, flag_matrix='all',
                               input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='rf',
                               flag_raw='')
        src.correlation_matrix(
            X=X_train[src.raw_features_to_list(feature_importance['columns'].unique().tolist())], y=None,
            flag_matrix='all',
            input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='rf', flag_raw='yes')

        print(Fore.GREEN + '\n *** Starting Logistic regression *** \n' + Style.RESET_ALL)
        logging.info('*** Starting Logistic regression ***')
        try:
            X_train, model_train, x_train_ac, x_train_auc, x_train_prec, x_train_nb_features, feature_importance, lr_table, x_train_recall, x_train_f1 = src.lr_run(
                df=X_train, criterion=y_train,
                model_to_predict=None,
                predict_only_flag='no', test_X=X_test, test_y=y_test, final_features=final_features,
                lr_features=params, cut_offs=cut_offs, params=params)
            X_test, _, x_test_ac, x_test_auc, x_test_prec, x_test_nb_features, _, _, x_test_recall, x_test_f1 = src.lr_run(
                df=X_test,
                criterion=y_test,
                model_to_predict=model_train,
                predict_only_flag='yes',
                test_X=None,
                test_y=None,
                final_features=final_features,
                lr_features=None,
                cut_offs=cut_offs,
                params=params)
            t1df, _, t1df_ac, t1df_auc, t1df_prec, t1df_nb_features, _, _, t1df_recall, t1df_f1 = src.lr_run(df=t1df,
                                                                                                             criterion=
                                                                                                                   t1df[
                                                                                                                       criterion_column],
                                                                                                             model_to_predict=model_train,
                                                                                                             predict_only_flag='yes',
                                                                                                             test_X=None,
                                                                                                             test_y=None,
                                                                                                             final_features=final_features,
                                                                                                             lr_features=None,
                                                                                                             cut_offs=cut_offs,
                                                                                                             params=params)
            t2df, _, t2df_ac, t2df_auc, t2df_prec, t2df_nb_features, _, _, t2df_recall, t2df_f1 = src.lr_run(df=t2df,
                                                                                                             criterion=
                                                                                                                   t2df[
                                                                                                                       criterion_column],
                                                                                                             model_to_predict=model_train,
                                                                                                             predict_only_flag='yes',
                                                                                                             test_X=None,
                                                                                                             test_y=None,
                                                                                                             final_features=final_features,
                                                                                                             lr_features=None,
                                                                                                             cut_offs=cut_offs,
                                                                                                             params=params)
            t3df, _, t3df_ac, t3df_auc, t3df_prec, t3df_nb_features, _, _, t3df_recall, t3df_f1 = src.lr_run(df=t3df,
                                                                                                             criterion=
                                                                                                                   t3df[
                                                                                                                       criterion_column],
                                                                                                             model_to_predict=model_train,
                                                                                                             predict_only_flag='yes',
                                                                                                             test_X=None,
                                                                                                             test_y=None,
                                                                                                             final_features=final_features,
                                                                                                             lr_features=None,
                                                                                                             cut_offs=cut_offs,
                                                                                                             params=params)

            models = models.append(
                {'Method': 'lr', 'AccuracyScore': x_train_ac, 'AUC': x_train_auc, 'PrecisionScore': x_train_prec,
                 'Recall': x_train_recall, 'F1': x_train_f1,
                 'DataSet': 'X_train', 'NbFeatures': x_train_nb_features},
                ignore_index=True)
            models = models.append(
                {'Method': 'lr', 'AccuracyScore': x_test_ac, 'AUC': x_test_auc, 'PrecisionScore': x_test_prec,
                 'Recall': x_test_recall, 'F1': x_test_f1,
                 'DataSet': 'X_test', 'NbFeatures': x_test_nb_features},
                ignore_index=True)
            models = models.append(
                {'Method': 'lr', 'AccuracyScore': t1df_ac, 'AUC': t1df_auc, 'PrecisionScore': t1df_prec,
                 'Recall': t1df_recall, 'F1': t1df_f1,
                 'DataSet': 't1df', 'NbFeatures': t1df_nb_features},
                ignore_index=True)
            models = models.append(
                {'Method': 'lr', 'AccuracyScore': t2df_ac, 'AUC': t2df_auc, 'PrecisionScore': t2df_prec,
                 'Recall': t2df_recall, 'F1': t2df_f1,
                 'DataSet': 't2df', 'NbFeatures': t2df_nb_features},
                ignore_index=True)
            models = models.append(
                {'Method': 'lr', 'AccuracyScore': t3df_ac, 'AUC': t3df_auc, 'PrecisionScore': t3df_prec,
                 'Recall': t3df_recall, 'F1': t3df_f1,
                 'DataSet': 't3df', 'NbFeatures': t3df_nb_features},
                ignore_index=True)
            pickle.dump(model_train, open(session_id_folder + '/lr/model_train.pkl', 'wb'))
            feature_importance.to_csv(session_id_folder + '/lr/feat_importance.csv', index=False)
            logging.info(f"Feature importance: {feature_importance}")
            print(f"\t Feature importance: {feature_importance}")
            lr_table.to_csv(session_id_folder + '/lr/lr_table.csv', index=False)
            src.correlation_matrix(X=X_train[feature_importance['columns'].unique().tolist()], y=None,
                                   flag_matrix='all',
                                   input_data_project_folder=None, session_id_folder=session_id_folder,
                                   model_corr='lr',
                                   flag_raw='')
            src.correlation_matrix(
                X=X_train[src.raw_features_to_list(feature_importance['columns'].unique().tolist())], y=None,
                flag_matrix='all',
                input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='lr', flag_raw='yes')
        except Exception as e:
            logging.error('LOGIT: %s', e)
            print('LOGIT error', e)
            pass

        print(Fore.GREEN + '\n *** Starting Decision Tree *** \n' + Style.RESET_ALL)
        logging.info('*** Starting Decision Tree ***')
        X_train, model_train, x_train_ac, x_train_auc, x_train_prec, x_train_nb_features, feature_importance, x_train_recall, x_train_f1 = src.decision_tree(
            df=X_train, criterion=y_train,
            df_us=X_train_us,
            criterion_us=y_train_us,
            test_X_us=X_test_us,
            test_y_us=y_test_us,
            model_to_predict=None,
            predict_only_flag='no', test_X=X_test, test_y=y_test, final_features=final_features, cut_offs=cut_offs,
            params=params)
        X_test, _, x_test_ac, x_test_auc, x_test_prec, x_test_nb_features, _, x_test_recall, x_test_f1 = src.decision_tree(
            df=X_test,
            criterion=y_test,
            df_us=X_train_us,
            criterion_us=y_train_us,
            test_X_us=X_test_us,
            test_y_us=y_test_us,
            model_to_predict=model_train,
            predict_only_flag='yes',
            test_X=None,
            test_y=None,
            final_features=final_features,
            cut_offs=cut_offs,
            params=params)
        t1df, _, t1df_ac, t1df_auc, t1df_prec, t1df_nb_features, _, t1df_recall, t1df_f1 = src.decision_tree(df=t1df,
                                                                                                             criterion=
                                                                                                                   t1df[
                                                                                                                       criterion_column],
                                                                                                             df_us=X_train_us,
                                                                                                             criterion_us=y_train_us,
                                                                                                             test_X_us=X_test_us,
                                                                                                             test_y_us=y_test_us,
                                                                                                             model_to_predict=model_train,
                                                                                                             predict_only_flag='yes',
                                                                                                             test_X=None,
                                                                                                             test_y=None,
                                                                                                             final_features=final_features,
                                                                                                             cut_offs=cut_offs,
                                                                                                             params=params)
        t2df, _, t2df_ac, t2df_auc, t2df_prec, t2df_nb_features, _, t2df_recall, t2df_f1 = src.decision_tree(df=t2df,
                                                                                                             criterion=
                                                                                                                   t2df[
                                                                                                                       criterion_column],
                                                                                                             df_us=X_train_us,
                                                                                                             criterion_us=y_train_us,
                                                                                                             test_X_us=X_test_us,
                                                                                                             test_y_us=y_test_us,
                                                                                                             model_to_predict=model_train,
                                                                                                             predict_only_flag='yes',
                                                                                                             test_X=None,
                                                                                                             test_y=None,
                                                                                                             final_features=final_features,
                                                                                                             cut_offs=cut_offs,
                                                                                                             params=params)
        t3df, _, t3df_ac, t3df_auc, t3df_prec, t3df_nb_features, _, t3df_recall, t3df_f1 = src.decision_tree(df=t3df,
                                                                                                             criterion=
                                                                                                                   t3df[
                                                                                                                       criterion_column],
                                                                                                             df_us=X_train_us,
                                                                                                             criterion_us=y_train_us,
                                                                                                             test_X_us=X_test_us,
                                                                                                             test_y_us=y_test_us,
                                                                                                             model_to_predict=model_train,
                                                                                                             predict_only_flag='yes',
                                                                                                             test_X=None,
                                                                                                             test_y=None,
                                                                                                             final_features=final_features,
                                                                                                             cut_offs=cut_offs,
                                                                                                             params=params)

        models = models.append(
            {'Method': 'dt', 'AccuracyScore': x_train_ac, 'AUC': x_train_auc, 'PrecisionScore': x_train_prec,
             'Recall': x_train_recall, 'F1': x_train_f1,
             'DataSet': 'X_train', 'NbFeatures': x_train_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'dt', 'AccuracyScore': x_test_ac, 'AUC': x_test_auc, 'PrecisionScore': x_test_prec,
             'Recall': x_test_recall, 'F1': x_test_f1,
             'DataSet': 'X_test', 'NbFeatures': x_test_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'dt', 'AccuracyScore': t1df_ac, 'AUC': t1df_auc, 'PrecisionScore': t1df_prec, 'Recall': t1df_recall,
             'F1': t1df_f1,
             'DataSet': 't1df', 'NbFeatures': t1df_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'dt', 'AccuracyScore': t2df_ac, 'AUC': t2df_auc, 'PrecisionScore': t2df_prec, 'Recall': t2df_recall,
             'F1': t2df_f1,
             'DataSet': 't2df', 'NbFeatures': t2df_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'dt', 'AccuracyScore': t3df_ac, 'AUC': t3df_auc, 'PrecisionScore': t3df_prec, 'Recall': t3df_recall,
             'F1': t3df_f1,
             'DataSet': 't3df', 'NbFeatures': t3df_nb_features},
            ignore_index=True)
        pickle.dump(model_train, open(session_id_folder + '/dt/model_train.pkl', 'wb'))
        feature_importance.to_csv(session_id_folder + '/dt/feat_importance.csv', index=False)
        logging.info(f"Feature importance: {feature_importance}")
        print(f"\t Feature importance: {feature_importance}")
        src.correlation_matrix(X=X_train[feature_importance['columns'].unique().tolist()], y=None, flag_matrix='all',
                               input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='dt',
                               flag_raw='')
        src.correlation_matrix(
            X=X_train[src.raw_features_to_list(feature_importance['columns'].unique().tolist())], y=None,
            flag_matrix='all',
            input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='dt', flag_raw='yes')

        # Create session folder and save results
        print(Fore.GREEN + '\n Saving Train, Test, T1,2,3 datasets (full datasets)' + Style.RESET_ALL)
        logging.info('\n Saving Train, Test, T1,2,3 datasets (full datasets)')

        X_train.to_parquet(session_id_folder + '/df_x_train.parquet')
        X_test.to_parquet(session_id_folder + '/df_x_test.parquet')
        t1df.to_parquet(session_id_folder + '/df_t1df.parquet')
        t2df.to_parquet(session_id_folder + '/df_t2df.parquet')
        t3df.to_parquet(session_id_folder + '/df_t3df.parquet')
        models.to_csv(session_id_folder + '/models.csv', index=False)
        print(Fore.YELLOW + '\n Session folder: ' + session_id_folder + Style.RESET_ALL)
        logging.info(f'Create session folder: {session_id_folder}')


    if run_info == 'create':
        print('Creating the folder structure')
        logging.info('Creating the folder structure')
        if not os.path.isdir(log_folder_name):
            os.mkdir(log_folder_name)
        if not os.path.isdir(session_folder_name):
            os.mkdir(session_folder_name)
        if not os.path.isdir(input_data_folder_name):
            os.mkdir(input_data_folder_name)
        if not os.path.isdir(output_data_folder_name):
            os.mkdir(output_data_folder_name)
        if not os.path.isdir(functions_folder_name):
            os.mkdir(functions_folder_name)
        if not os.path.isdir(params_folder_name):
            os.mkdir(params_folder_name)
        sys.exit()
    elif run_info == 'load':
        src.print_load()
        print(Fore.GREEN + 'Starting the session for: ' + input_data_project_folder + Style.RESET_ALL)
        logging.info(f'Starting the session for: {input_data_project_folder}')

        # create loader object
        loader = src.BaseLoader()
        loader.input_df, loader.input_df_full = src.data_load_prep(session)
        loader.input_df, loader.input_df_full, loader.final_features = src.data_cleaning(input_df=loader.input_df,
                                                                                         input_df_full=loader.input_df_full,
                                                                                         session=session)
        print('\n Saving processed data \n')
        loader.input_df.to_parquet(
            session.output_data_folder_name + session.input_data_project_folder + '/' + 'output_data_file.parquet')
        if params['under_sampling']:
            loader.input_df_full.to_parquet(
                session.output_data_folder_name + session.input_data_project_folder + '/' + 'output_data_file_full.parquet')
        with open(session.output_data_folder_name + session.input_data_project_folder + '/' + 'final_features.pkl',
                  'wb') as f:
            pickle.dump(loader.final_features, f)
        src.print_end()
    elif run_info == 'train':
        src.print_train()
        print(Fore.GREEN + 'Starting the session for: ' + input_data_project_folder + Style.RESET_ALL)
        logging.info(f'Starting the session for: {input_data_project_folder}')
        modeller(input_data_project_folder)
        src.print_end()
    elif run_info == 'eval':
        src.print_eval()
        print(Fore.GREEN + 'Starting the session for: ' + input_data_project_folder + Style.RESET_ALL)
        logging.info(f'Starting the session for: {input_data_project_folder}')

        # Create new session folder:
        print(Fore.GREEN + 'Createing session folder. Starting' + Style.RESET_ALL)
        logging.info('Createing session folder. Starting')
        session_id = 'EVAL_' + input_data_project_folder + '_' + str(start) + '_' + tag
        session_id_folder = session_folder_name + session_id
        os.mkdir(session_id_folder)
        print('Createing session folder. Done')
        logging.info('Createing session folder. Done')

        try:
            print(Fore.GREEN + '\n Starting Eval for LR\n' + Style.RESET_ALL)
            src.merge_word(input_data_folder_name, input_data_project_folder, session_to_eval, session_folder_name,
                           session_id_folder, criterion_column,
                           observation_date_column,
                           columns_to_exclude,
                           periods_to_exclude,
                           t1df_period,
                           t2df_period,
                           t3df_period,
                           model_arg='lr',
                           missing_treatment=missing_treatment, params=params)
        except Exception as e:
            print(f'ERROR with LR: {e}')
            pass
        try:
            print(Fore.GREEN + '\n Starting Eval for DT\n' + Style.RESET_ALL)
            src.merge_word(input_data_folder_name, input_data_project_folder, session_to_eval, session_folder_name,
                           session_id_folder, criterion_column,
                           observation_date_column,
                           columns_to_exclude,
                           periods_to_exclude,
                           t1df_period,
                           t2df_period,
                           t3df_period,
                           model_arg='dt',
                           missing_treatment=missing_treatment, params=params)
        except Exception as e:
            print(f'ERROR with DT: {e}')
            pass
        try:
            print(Fore.GREEN + '\n Starting Eval for XGB \n' + Style.RESET_ALL)
            src.merge_word(input_data_folder_name, input_data_project_folder, session_to_eval, session_folder_name,
                           session_id_folder, criterion_column,
                           observation_date_column,
                           columns_to_exclude,
                           periods_to_exclude,
                           t1df_period,
                           t2df_period,
                           t3df_period,
                           model_arg='xgb',
                           missing_treatment=missing_treatment, params=params)
        except Exception as e:
            print(f'ERROR with XGB: {e}')
            pass
        try:
            print(Fore.GREEN + '\n Starting Eval for RF\n' + Style.RESET_ALL)
            src.merge_word(input_data_folder_name, input_data_project_folder, session_to_eval, session_folder_name,
                           session_id_folder, criterion_column,
                           observation_date_column,
                           columns_to_exclude,
                           periods_to_exclude,
                           t1df_period,
                           t2df_period,
                           t3df_period,
                           model_arg='rf',
                           missing_treatment=missing_treatment, params=params)
        except Exception as e:
            print(f'ERROR with RF: {e}')
            pass

        src.print_end()
    else:
        print('No arguments')
        src.print_end()

    sys.exit()
"""
