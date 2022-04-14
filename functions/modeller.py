import logging
import xgboost
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
import statsmodels.api as sm
import random

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def xgb(df, criterion, test_X, test_y, df_us, criterion_us, test_X_us, test_y_us, predict_only_flag, model_to_predict,
        final_features, cut_points_train, cut_offs, params):
    """
    Generates the XGBoost model or applies it on a dataframe
    Returns df, xgb_def, ac, auc, prec, len(final_features), results, cut_points_train
        df - the df parameter, enriched with additional columns
        xgb_def - XGBoost model object
        ac - Accuracy score
        AUC - Area under the curve
        prec - Precision score
        len final features - the nb of the features in the model
        results - dataframe with feature names and importance
        cut points train - obsolete

    Parameters:
        df, criterion, test_X, test_y, predict_only_flag, model_to_predict, final_features, cut_points_train, cut_offs
        df - the dataframe on which to train or predict the model
        criterion - series, the column that the model is training/predicting
        test_y, test_X - test group used for modelling
        predict only flag - indicates if the function should train or predict
        final features - the features to be used for training/predicting
        cut_points_train - the cut points for probability to generate bands (experimental)
        cut_offs - defined cutoffs in the param file to generate score bands based on the probability
    """
    results = pd.DataFrame()
    if predict_only_flag != 'yes':
        if params['trees_features_to_include']:
            final_features = params['trees_features_to_include']

        xgb_def = xgboost.XGBClassifier()

        if params['under_sampling']:
            print('\n\t *** UNDERSAMPLING MODEL ***')
            xgb_def.fit(df_us[final_features], criterion_us, eval_set=[(test_X_us[final_features], test_y_us)],
                        early_stopping_rounds=15)
        else:
            xgb_def.fit(df[final_features], criterion, eval_set=[(test_X[final_features], test_y)],
                        early_stopping_rounds=15)

        results['columns'] = df[final_features].columns
        results['importances'] = xgb_def.feature_importances_
        results.sort_values(by='importances', ascending=False, inplace=True)
        results = results[results['importances'] > 0]
        results = results[results['importances'] < 0.95]
        results = results[:30]
        final_features = results['columns'].unique().tolist()

        if params['trees_features_to_exclude']:
            trees_features_to_exclude = params['trees_features_to_exclude']
            print(f'\t Removing features specified in the params file: {trees_features_to_exclude}')
            logging.info(f'Removing features specified in the params file: {trees_features_to_exclude}')
            for el in trees_features_to_exclude:
                try:
                    final_features.remove(el)
                    results = results[~results['columns'].str.contains(el)]
                except:
                    pass

        logging.info(f'FINAL FEATURES: {final_features}')
        if params['under_sampling']:
            xgb_def.fit(df_us[final_features], criterion_us, eval_set=[(test_X_us[final_features], test_y_us)],
                        early_stopping_rounds=15)
            ac, auc, prec, recall, f1 = get_metrics(y_pred=xgb_def.predict(df_us[final_features]), y_true=criterion_us,
                                        y_pred_prob=xgb_def.predict_proba(df_us[final_features])[:, 1])
            print(f'\t UNDERSAMPLING MODEL efficiency: AC: {ac}, AUC: {auc}, Prec: {prec}')
            logging.info(f'\t UNDERSAMPLING MODEL efficiency: AC: {ac}, AUC: {auc}, Prec: {prec}')
        else:
            xgb_def.fit(df[final_features], criterion, eval_set=[(test_X[final_features], test_y)],
                        early_stopping_rounds=15)
    else:
        xgb_def = model_to_predict
        final_features = xgb_def.get_booster().feature_names

    df['xgb_y_pred'] = xgb_def.predict(df[final_features])
    df['xgb_deciles_predict'] = pd.qcut(df['xgb_y_pred'], 10, duplicates='drop', labels=False)
    df['xgb_y_pred_prob'] = xgb_def.predict_proba(df[final_features])[:, 1]
    df['xgb_deciles_pred_prob'] = pd.qcut(df['xgb_y_pred_prob'], 10, duplicates='drop', labels=False)

    ac, auc, prec, recall, f1 = get_metrics(y_pred=df['xgb_y_pred'], y_true=criterion, y_pred_prob=df['xgb_y_pred_prob'])

    if cut_offs["xgb"]:
        df['xgb_bands_predict'] = pd.cut(df['xgb_y_pred'], bins=cut_offs["xgb"], include_lowest=True).astype('str')
        df['xgb_bands_predict_proba'] = pd.cut(df['xgb_y_pred_prob'], bins=cut_offs["xgb"], include_lowest=True).astype(
            'str')
    else:
        df['xgb_bands_predict'], _ = cut_into_bands(X=df[['xgb_y_pred']], y=criterion, depth=3)
        df['xgb_bands_predict_proba'], _ = cut_into_bands(X=df[['xgb_y_pred_prob']], y=criterion, depth=3)

    logging.info('XGB: Model found')
    return df, xgb_def, ac, auc, prec, len(final_features), results, cut_points_train, recall, f1


def rand_forest(df, criterion, df_us, criterion_us, test_X_us, test_y_us, test_X, test_y, predict_only_flag,
                model_to_predict, final_features, cut_offs, params):
    results = pd.DataFrame()
    if predict_only_flag != 'yes':
        if params['trees_features_to_include']:
            final_features = params['trees_features_to_include']

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, n_jobs=3)
        rf_model.fit(df[final_features], criterion)
        if params['under_sampling']:
            print('\n\t *** UNDERSAMPLING MODEL ***')
            rf_model.fit(df_us[final_features], criterion_us)
        else:
            rf_model.fit(df[final_features], criterion)

        results['columns'] = df[final_features].columns
        results['importances'] = rf_model.feature_importances_
        results.sort_values(by='importances', ascending=False, inplace=True)
        results = results[results['importances'] > 0]
        results = results[results['importances'] < 0.95]
        results = results[:30]
        final_features = results['columns'].unique().tolist()

        if params['trees_features_to_exclude']:
            trees_features_to_exclude = params['trees_features_to_exclude']
            print(f'\t Removing features specified in the params file: {trees_features_to_exclude}')
            logging.info(f'Removing features specified in the params file: {trees_features_to_exclude}')
            for el in trees_features_to_exclude:
                try:
                    final_features.remove(el)
                    results = results[~results['columns'].str.contains(el)]
                except:
                    pass

        logging.info(f'FINAL FEATURES: {final_features}')

        if params['under_sampling']:
            rf_model.fit(df_us[final_features], criterion_us)
            ac, auc, prec, recall, f1 = get_metrics(y_pred=rf_model.predict(df_us[final_features]), y_true=criterion_us,
                                        y_pred_prob=rf_model.predict_proba(df_us[final_features])[:, 1])
            print(f'\t UNDERSAMPLING MODEL efficiency: AC: {ac}, AUC: {auc}, Prec: {prec}')
            logging.info(f'\t UNDERSAMPLING MODEL efficiency: AC: {ac}, AUC: {auc}, Prec: {prec}')
        else:
            rf_model.fit(df[final_features], criterion)

        rf_model.feature_names = final_features
    else:
        rf_model = model_to_predict
        final_features = rf_model.feature_names

    df['rf_y_pred'] = rf_model.predict(df[final_features])
    df['rf_y_pred_prob'] = rf_model.predict_proba(df[final_features])[:, 1]
    y_pred = rf_model.predict(df[final_features])

    ac, auc, prec, recall, f1 = get_metrics(y_pred=y_pred, y_true=criterion, y_pred_prob=df['rf_y_pred_prob'])

    if cut_offs["rf"]:
        df['rf_bands_predict'] = pd.cut(df['rf_y_pred'], bins=cut_offs["rf"], include_lowest=True).astype('str')
        df['rf_bands_predict_proba'] = pd.cut(df['rf_y_pred_prob'], bins=cut_offs["rf"], include_lowest=True).astype(
            'str')
    else:
        df['rf_bands_predict'], _ = cut_into_bands(X=df[['rf_y_pred']], y=criterion, depth=3)
        df['rf_bands_predict_proba'], _ = cut_into_bands(X=df[['rf_y_pred_prob']], y=criterion, depth=3)
    logging.info('RF: Model found')

    # pickle.dump(xgb_def, open("./sessions/xgb.pkl", 'wb'))

    return df, rf_model, ac, auc, prec, len(final_features), results, recall, f1


def decision_tree(df, criterion, df_us, criterion_us, test_X_us, test_y_us, test_X, test_y, predict_only_flag,
                  model_to_predict, final_features, cut_offs, params):
    results = pd.DataFrame()
    if predict_only_flag != 'yes':
        if params['trees_features_to_include']:
            final_features = params['trees_features_to_include']

        if params['under_sampling']:
            print('\n\t *** UNDERSAMPLING MODEL ***')
            _, dt_model = cut_into_bands(df_us[final_features], criterion_us, depth=5)
        else:
            _, dt_model = cut_into_bands(df[final_features], criterion, depth=5)

        r = export_text(dt_model, feature_names=final_features)
        # print(r)
        results = pd.DataFrame()
        results['columns'] = df[final_features].columns
        results['importances'] = dt_model.feature_importances_
        results.sort_values(by='importances', ascending=False, inplace=True)
        results = results[results['importances'] > 0]
        results = results[results['importances'] < 0.95]
        final_features = results['columns'].unique().tolist()
        if params['trees_features_to_exclude']:
            trees_features_to_exclude = params['trees_features_to_exclude']
            print(f'\t Removing features specified in the params file: {trees_features_to_exclude}')
            logging.info(f'Removing features specified in the params file: {trees_features_to_exclude}')
            for el in trees_features_to_exclude:
                try:
                    final_features.remove(el)
                    results = results[~results['columns'].str.contains(el)]
                except:
                    pass

        if params['under_sampling']:
            dt_model.fit(df_us[final_features], criterion_us)
            ac, auc, prec, recall, f1 = get_metrics(y_pred=dt_model.predict(df_us[final_features]), y_true=criterion_us,
                                        y_pred_prob=dt_model.predict_proba(df_us[final_features])[:, 1])
            print(f'\t UNDERSAMPLING MODEL efficiency: AC: {ac}, AUC: {auc}, Prec: {prec}')
            logging.info(f'\t UNDERSAMPLING MODEL efficiency: AC: {ac}, AUC: {auc}, Prec: {prec}')
        else:
            dt_model.fit(df[final_features], criterion)

        dt_model.feature_names = final_features
        r = export_text(dt_model, feature_names=final_features)
        print(r)
        logging.info(f'FINAL FEATURES: {len(final_features)} {final_features}')
        raw_features = raw_features_to_list(final_features)
        dt_model.raw_features = raw_features
        logging.info(f'FINAL FEATURES: {len(raw_features)} {raw_features}')

    else:
        dt_model = model_to_predict
        final_features = dt_model.feature_names

    df['dt_y_pred'] = dt_model.predict(df[final_features])

    df['dt_y_pred_prob'] = dt_model.predict_proba(df[final_features])[:, 1]

    y_pred = dt_model.predict(df[final_features])

    ac, auc, prec, recall, f1 = get_metrics(y_pred=y_pred, y_true=criterion, y_pred_prob=df['dt_y_pred_prob'])

    if cut_offs["dt"]:
        df['dt_bands_predict'] = pd.cut(df['dt_y_pred'], bins=cut_offs["dt"], include_lowest=True).astype('str')
        df['dt_bands_predict_proba'] = pd.cut(df['dt_y_pred_prob'], bins=cut_offs["dt"], include_lowest=True).astype(
            'str')
    else:
        df['dt_bands_predict'], _ = cut_into_bands(X=df[['dt_y_pred']], y=criterion, depth=3)
        df['dt_bands_predict_proba'], _ = cut_into_bands(X=df[['dt_y_pred_prob']], y=criterion, depth=3)

    logging.info('RF: Model found')

    return df, dt_model, ac, auc, prec, len(final_features), results, recall, f1


def lr(X, y):
    X = sm.add_constant(X)  # add constant
    logit_mod = sm.Logit(y, X)  # add model
    logit_res = logit_mod.fit(disp=False)  # fit model
    logit_roc_auc = roc_auc_score(y, logit_res.predict(X))
    table = logit_res.summary()
    return logit_res, table, logit_roc_auc


def lr_run(df, criterion, test_X, test_y, predict_only_flag, model_to_predict, final_features, lr_features, cut_offs,
           params):
    lr_table = pd.DataFrame()
    results_lr = pd.DataFrame()
    if predict_only_flag != 'yes':
        df_sample = df.copy()

        if lr_features["lr_features"]:
            print('\t LR features: ', lr_features["lr_features"])
            logging.info(f'LR features: {lr_features["lr_features"]}')
            final_features = lr_features["lr_features"]
            lr_model, table, train_auc = lr(df_sample[final_features], criterion)
            print('\t LR modelAUC: ', train_auc)
            logging.info(f'LR modelAUC: {train_auc}')
        else:
            print('\t LR features not presented in params file, starting random models')
            if params["lr_features_to_include"]:
                lr_features_to_include = params["lr_features_to_include"]
                print(
                    '\t LR mandatory features presented in the params. Starting random models including the features: ',
                    lr_features_to_include)
            tries = 0
            auc_max = 0
            while tries < 100:
                tries += 1

                random_cols = random.sample(final_features, 15)
                if params["lr_features_to_include"]:
                    for el in params["lr_features_to_include"]:
                        if el in random_cols:
                            pass
                        else:
                            random_cols.append(el)

                try:
                    lr_model, table, train_auc = lr(df_sample[random_cols], criterion)

                    Coefficients = pd.read_html(table.tables[1].as_html(), header=0, index_col=0)[0]
                    Coefficients = Coefficients.reset_index()
                    Coefficients = Coefficients[['index', 'coef', 'P>|z|']]
                    Coefficients = Coefficients.rename(columns={'P>|z|': 'error'})
                    Coefficients = Coefficients[Coefficients.error < 0.01]
                    Coefficients = Coefficients['index'].tolist()
                    Coefficients.remove('const')

                    _, table, test_auc = lr(test_X[Coefficients], test_y)
                    Coefficients = pd.read_html(table.tables[1].as_html(), header=0, index_col=0)[0]
                    Coefficients = Coefficients.reset_index()
                    Coefficients = Coefficients[['index', 'coef', 'P>|z|']]
                    Coefficients = Coefficients.rename(columns={'P>|z|': 'error'})
                    Coefficients = Coefficients[Coefficients.error < 0.01]
                    Coefficients['coef_abs'] = abs(Coefficients['coef'])
                    Coefficients = Coefficients[Coefficients.coef_abs > 0.00005]
                    Coefficients = Coefficients['index'].tolist()
                    Coefficients.remove('const')

                    lr_model, table, train_auc = lr(df_sample[Coefficients], criterion)
                    Coefficients = pd.read_html(table.tables[1].as_html(), header=0, index_col=0)[0]
                    Coefficients = Coefficients.reset_index()
                    Coefficients = Coefficients[['index', 'coef', 'P>|z|']]
                    Coefficients = Coefficients.rename(columns={'P>|z|': 'error'})
                    Coefficients = Coefficients['index'].tolist()
                    Coefficients.remove('const')
                    if len(Coefficients) > 8:
                        df_abs = abs(df[Coefficients].corr())
                        df_abs = df_abs.where(df_abs < 1)
                        df_abs = df_abs.where(df_abs > 0.5)
                        df_abs = df_abs.isnull().values.all()
                        if df_abs:
                            if train_auc > auc_max:
                                tries = 0
                                auc_max = train_auc
                            print(
                                '\t After {} attempts: LR Model found with AUC: {} and {} nb of features. AUC max so far {}'.format(
                                    tries, round(train_auc, 2), len(Coefficients), round(auc_max, 2)))
                            results_lr = results_lr.append(
                                {'auc': train_auc, 'features': Coefficients},
                                ignore_index=True)
                except:
                    pass

            final_features = results_lr[results_lr['auc'] == results_lr['auc'].max()]
            final_features = final_features['features'].iloc[0]
            print(len(final_features), final_features)
            logging.info(f'The number of features is {len(final_features)} for feature {final_features}')

            lr_model, table, train_auc = lr(df[final_features], criterion)
            print('\t WINNER AUC: {}'.format(train_auc))
            logging.info(f'WINNER AUC: {train_auc}')

        lr_table = pd.read_html(table.tables[1].as_html(), header=0, index_col=0)[0]
        lr_table = lr_table.reset_index()
        lr_table = lr_table[['index', 'coef', 'P>|z|']]
        lr_table = lr_table.rename(columns={'P>|z|': 'error'})
        lr_model.feature_names = final_features

    else:
        lr_model = model_to_predict
        final_features = lr_model.feature_names

    df = sm.add_constant(df, has_constant='add')  # add constant
    final_features.insert(0, 'const')
    df['lr_y_pred'] = lr_model.predict(df[final_features])
    df['lr_y_pred_prob'] = lr_model.predict(df[final_features])
    y_pred = lr_model.predict(df[final_features])

    ac, auc, prec, recall, f1 = get_metrics(y_pred=y_pred, y_true=criterion, y_pred_prob=df['lr_y_pred_prob'])

    if cut_offs["lr"]:
        df['lr_bands_predict'] = pd.cut(df['lr_y_pred'], bins=cut_offs["lr"], include_lowest=True).astype('str')
        df['lr_bands_predict_proba'] = pd.cut(df['lr_y_pred_prob'], bins=cut_offs["lr"], include_lowest=True).astype(
            'str')
    else:
        df['lr_bands_predict'], _ = cut_into_bands(X=df[['lr_y_pred']], y=criterion, depth=3)
        df['lr_bands_predict_proba'], _ = cut_into_bands(X=df[['lr_y_pred_prob']], y=criterion, depth=3)
    logging.info('LR: Model found')

    final_features.remove('const')

    results = pd.DataFrame({'a': range(100)})
    results['columns'] = pd.Series(final_features, index=results.index[:len(final_features)])
    results['importances'] = 1
    results = results.dropna()
    del results['a']

    return df, lr_model, ac, auc, prec, len(final_features), results, lr_table, recall, f1


def cut_into_bands(X, y, depth):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X, y)
    return clf.predict(X), clf


def get_metrics(y_true, y_pred, y_pred_prob):
    try:
        roc_auc_score_val = round(roc_auc_score(y_true, y_pred_prob), 2)
        y_pred = [round(value) for value in y_pred[:]]
        accuracy_score_val = round(accuracy_score(y_true, y_pred), 2)
        precision_score_val = round(precision_score(y_true, y_pred), 2)
        recall_score_val = round(recall_score(y_true, y_pred), 2)
        f1_score_val = round(f1_score(y_true, y_pred), 2)
        print(
            f'\t AS:, {accuracy_score_val}, AUC: {roc_auc_score_val}, Precision: {precision_score_val}, Recall: {recall_score_val}, F1: {f1_score_val}, df shape: {len(y_true)}')
        logging.info(
            f'Metrics: Model found: AS:, {accuracy_score_val}, AUC: {roc_auc_score_val}, Precision: {precision_score_val}, Recall: {recall_score_val}, F1: {f1_score_val}, df shape: {len(y_true)}')
    except Exception as e:
        print(
            f"Metrics error: {e}. All metrics' values will be set to 0.5. May be the issue is that you have included "
            f"an observation period that has no full performance period and therefore no real cases to be predicted?")
        accuracy_score_val = 0.5
        roc_auc_score_val = 0.5
        precision_score_val = 0.5
        recall_score_val = 0.5
        f1_score_val = 0.5
    return accuracy_score_val, roc_auc_score_val, precision_score_val, recall_score_val, f1_score_val


def raw_features_to_list(final_features):
    raw_features = []
    for feat in final_features[:]:
        if 'binned' in feat:
            prefix, _, _ = str(feat).partition('_binned')
            if '_ratio_' in prefix:
                if '_div_' in prefix:
                    a, b, c = str(prefix).partition('_div_ratio_')
                    raw_features.append(a)
                    raw_features.append(c)
                elif '_add_' in prefix:
                    a, b, c = str(prefix).partition('_add_ratio_')
                    raw_features.append(a)
                    raw_features.append(c)
                elif '_subs_' in prefix:
                    a, b, c = str(prefix).partition('_subs_ratio_')
                    raw_features.append(a)
                    raw_features.append(c)
                elif '_mult_' in prefix:
                    a, b, c = str(prefix).partition('_mult_ratio_')
                    raw_features.append(a)
                    raw_features.append(c)
            else:
                raw_features.append(prefix)
        elif '_ratio_' in feat:
            if '_div_' in feat:
                a, b, c = str(feat).partition('_div_ratio_')
                raw_features.append(a)
                raw_features.append(c)
            elif '_add_' in feat:
                a, b, c = str(feat).partition('_add_ratio_')
                raw_features.append(a)
                raw_features.append(c)
            elif '_subs_' in feat:
                a, b, c = str(feat).partition('_subs_ratio_')
                raw_features.append(a)
                raw_features.append(c)
            elif '_mult_' in feat:
                a, b, c = str(feat).partition('_mult_ratio_')
                raw_features.append(a)
                raw_features.append(c)
        else:
            raw_features.append(feat)
    raw_features = list(dict.fromkeys(raw_features))
    return raw_features
