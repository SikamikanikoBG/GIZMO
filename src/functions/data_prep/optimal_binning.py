import pandas as pd
from optbinning import OptimalBinning
from sklearn.model_selection import train_test_split

from src import print_and_log


def optimal_binning(df, df_full, columns, criterion_column, final_features, observation_date_column, params):
    # creating binned dummie features from all numeric ones
    print_and_log('\n Starting numerical features Optimal binning (max 4 bins) based on train df \n', '')

    for col in columns[:]:
        temp_df = df.copy()
        temp_df2 = df.copy()
        temp_df_full = df_full.copy()

        # Removing all periods before splitting train and test
        temp_df2 = temp_df2[temp_df2[observation_date_column] != params['t1df']]
        temp_df2 = temp_df2[temp_df2[observation_date_column] != params['t2df']]
        temp_df2 = temp_df2[temp_df2[observation_date_column] != params['t3df']]

        x_train, _, y_train, _ = train_test_split(
            temp_df2, temp_df2[criterion_column], test_size=0.33, random_state=42)
        x_train = x_train.dropna(subset=[col])

        x = x_train[col].values
        y = x_train[criterion_column].values
        optb = OptimalBinning(name=col, dtype='numerical', solver='cp', max_n_bins=3, min_bin_size=0.1)
        optb.fit(x, y)

        temp_df = temp_df.dropna(subset=[col])
        binned_col_name = col + '_binned'
        temp_df[binned_col_name] = optb.transform(temp_df[col], metric='bins')

        dummies = pd.get_dummies(temp_df[binned_col_name], prefix=binned_col_name + '_dummie')
        print_and_log('{} is with the following splits: {} and dummie columns: {}'.format(col, optb.splits,
                                                                                          list(dummies.columns)), '')
        temp_df[dummies.columns] = dummies

        columns.remove(col)
        for el in list(dummies.columns):
            columns.append(el)

        df = pd.concat([df, temp_df[dummies.columns]], axis=1)
        df[dummies.columns] = df[dummies.columns].fillna(0)

        if params['under_sampling']:
            temp_df_full = temp_df_full.dropna(subset=[col])
            binned_col_name = col + '_binned'
            temp_df_full[binned_col_name] = optb.transform(temp_df_full[col], metric='bins')

            dummies = pd.get_dummies(temp_df_full[binned_col_name], prefix=binned_col_name + '_dummie')
            temp_df_full[dummies.columns] = dummies

            df_full = pd.concat([df_full, temp_df_full[dummies.columns]], axis=1)
            df_full[dummies.columns] = df_full[dummies.columns].fillna(0)

    # Recoding strings
    for string in columns[:]:
        new_string = string.replace("<", "less_than")
        new_string = new_string.replace(">", "more_than")
        new_string = new_string.replace(",", "_to_")
        new_string = new_string.replace("[", "from_incl_")
        new_string = new_string.replace("]", "_incl_")
        new_string = new_string.replace("(", "from_excl_")
        new_string = new_string.replace(")", "_excl_")
        new_string = new_string.replace(" ", "")
        columns.remove(string)
        columns.append(new_string)

    for string in final_features[:]:
        new_string = string.replace("<", "less_than")
        new_string = new_string.replace(">", "more_than")
        new_string = new_string.replace(",", "_to_")
        new_string = new_string.replace("[", "from_incl_")
        new_string = new_string.replace("]", "_incl_")
        new_string = new_string.replace("(", "from_excl_")
        new_string = new_string.replace(")", "_excl_")
        new_string = new_string.replace(" ", "")
        final_features.remove(string)
        final_features.append(new_string)

    for col in df:
        new_string = col.replace("<", "less_than")
        new_string = new_string.replace(">", "more_than")
        new_string = new_string.replace(",", "_to_")
        new_string = new_string.replace("[", "from_incl_")
        new_string = new_string.replace("]", "_incl_")
        new_string = new_string.replace("(", "from_excl_")
        new_string = new_string.replace(")", "_excl_")
        new_string = new_string.replace(" ", "")
        df.rename(columns={col: new_string}, inplace=True)

    for col in df_full:
        new_string = col.replace("<", "less_than")
        new_string = new_string.replace(">", "more_than")
        new_string = new_string.replace(",", "_to_")
        new_string = new_string.replace("[", "from_incl_")
        new_string = new_string.replace("]", "_incl_")
        new_string = new_string.replace("(", "from_excl_")
        new_string = new_string.replace(")", "_excl_")
        new_string = new_string.replace(" ", "")
        df_full.rename(columns={col: new_string}, inplace=True)
    return df, df_full, columns