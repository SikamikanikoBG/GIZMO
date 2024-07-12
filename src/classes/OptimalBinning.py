from multiprocessing import Pool

import pandas as pd
from optbinning import OptimalBinning, MulticlassOptimalBinning
from sklearn.model_selection import train_test_split

from src import print_and_log
import definitions


class OptimaBinning:
    def __init__(self, df, df_full, columns, criterion_column, final_features, observation_date_column, params, project_name):
        """
        Initialize the OptimaBinning object with the provided parameters.

        Args:
            df (pandas.DataFrame): Input DataFrame.
            df_full (pandas.DataFrame): Full DataFrame.
            columns (list): List of columns.
            criterion_column (str): Criterion column.
            final_features (list): List of final features.
            observation_date_column (str): Observation date column.
            params (dict): Dictionary of parameters.

        Attributes:
            df (pandas.DataFrame): Input DataFrame.
            df_full (pandas.DataFrame): Full DataFrame.
            columns (list): List of columns.
            criterion_column (str): Criterion column.
            final_features (list): List of final features.
            observation_date_column (str): Observation date column.
            params (dict): Dictionary of parameters.
            is_multiclass (bool): Flag indicating if the problem is multiclass.
        """
        self.df = df
        self.df_full = df_full
        self.columns = columns
        self.criterion_column = criterion_column
        self.final_features = final_features
        self.observation_date_column = observation_date_column
        self.project_name = project_name
        self.param_file_name = definitions.EXTERNAL_DIR + '/params/params_' + self.project_name + '.json'
        self.params = params
        self.is_multiclass = True if self.df_full[criterion_column].nunique() > 2 else False


    def optimal_binning_procedure(self, col):
        """
        Perform the optimal binning procedure for a specific column.

        Args:
            col (str): Column to perform optimal binning on.

        Returns:
            pandas.DataFrame: Processed DataFrame for the column.
            pandas.DataFrame: Processed DataFrame with under-sampling for the column.
            list: List of dummies columns.
        """
        try:
            print_and_log(f'[ OPTIMAL BINNING ] Starting for {col}', '')
            # creating binned dummie features from all numeric ones
            temp_df = self.df.copy()
            temp_df2 = self.df.copy()
            temp_df_full = self.df_full.copy()

            # Removing all periods before splitting train and test            
            temp_df2 = temp_df2[~temp_df2[self.observation_date_column].isin(
                        [self.params['t1df'],
                         self.params['t2df'], 
                         self.params['t3df']]
                    ) 
                ]           

            x_train, _, y_train, _ = train_test_split(
                temp_df2, temp_df2[self.criterion_column], test_size=0.33, random_state=42)
            x_train = x_train.dropna(subset=[col])

            x = x_train[col].values
            y = x_train[self.criterion_column].values
            
            if not self.is_multiclass:
                # Call model
                optb = OptimalBinning(name=col, dtype='numerical', solver='cp', max_n_bins=4, min_bin_size=0.1)
                # Fit model to data
                optb.fit(x, y)
                # Drop nan
                temp_df = temp_df.dropna(subset=[col])
                # Set the name of the new binned column
                binned_col_name = col + '_binned'
                # Tranform the column at hand, save it to the input df with the new name from binned_col_name
                temp_df[binned_col_name] = optb.transform(temp_df[col], metric='bins')
                # ohe the binned column into 4 categories as per the number of bins
                dummies = pd.get_dummies(temp_df[binned_col_name], prefix=binned_col_name + '_dummie')
                print_and_log(f'[ OPTIMAL BINNING ] {col} is with the following splits: {optb.splits} and '
                              f'dummie columns: {list(dummies.columns)}', 'GREEN')
                # Save the dummies into the input df
                temp_df[dummies.columns] = dummies
                # Same protocol if we have undersamplng, if not, full input df will have only ones (?)
                if self.params['under_sampling']:
                    temp_df_full = temp_df_full.dropna(subset=[col])
                    binned_col_name = col + '_binned'
                    temp_df_full[binned_col_name] = optb.transform(temp_df_full[col], metric='bins')

                    dummies = pd.get_dummies(temp_df_full[binned_col_name], prefix=binned_col_name + '_dummie')
                    temp_df_full[dummies.columns] = dummies
                else:
                    for col in list(dummies.columns):
                        temp_df_full[col] = 1

                # This here is really weird...
                dummies_list = list(dummies.columns)
                return temp_df[list(dummies.columns)], temp_df_full[list(dummies.columns)], dummies_list
            else:
                from sklearn.preprocessing import KBinsDiscretizer
                import numpy as np

                kbd = KBinsDiscretizer(n_bins=4, encode='onehot', strategy='uniform')
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                kbd.fit(x, y)

                temp_df, dummies_temp_df = multiclass_binning_transform_and_get_dummies(temp_df, col, kbd)

                if self.params['under_sampling']:
                    temp_df_full, dummies_temp_df_full = multiclass_binning_transform_and_get_dummies(temp_df_full, col, kbd)
                else:
                    # for col in list(dummies.columns):
                    #     temp_df_full[col] = 1
                    assert self.params['under_sampling'], f'[ OPTIMAL BINNING ERROR ] "under_sampling" in {self.param_file_name} is not set!'

                dummies_list = list(dummies_temp_df.columns)
                return temp_df, temp_df_full, dummies_list


        except Exception as e:
            print_and_log(f'[ OPTIMAL BINNING ] ERROR {e}. Skipping this column {col}', 'RED')
            return pd.DataFrame(), pd.DataFrame(), []

    def rename_strings_cols_opt_bin(self):
        """
       Rename string columns based on specific rules.
       """
        # Recoding strings
        for string in self.columns[:]:
            new_string = string.replace("<", "less_than")
            new_string = new_string.replace(">", "more_than")
            new_string = new_string.replace(",", "_to_")
            new_string = new_string.replace("[", "from_incl_")
            new_string = new_string.replace("]", "_incl_")
            new_string = new_string.replace("(", "from_excl_")
            new_string = new_string.replace(")", "_excl_")
            new_string = new_string.replace(" ", "")
            self.columns.remove(string)
            self.columns.append(new_string)
        for string in self.final_features[:]:
            new_string = string.replace("<", "less_than")
            new_string = new_string.replace(">", "more_than")
            new_string = new_string.replace(",", "_to_")
            new_string = new_string.replace("[", "from_incl_")
            new_string = new_string.replace("]", "_incl_")
            new_string = new_string.replace("(", "from_excl_")
            new_string = new_string.replace(")", "_excl_")
            new_string = new_string.replace(" ", "")
            self.final_features.remove(string)
            self.final_features.append(new_string)
        for col in self.df:
            new_string = col.replace("<", "less_than")
            new_string = new_string.replace(">", "more_than")
            new_string = new_string.replace(",", "_to_")
            new_string = new_string.replace("[", "from_incl_")
            new_string = new_string.replace("]", "_incl_")
            new_string = new_string.replace("(", "from_excl_")
            new_string = new_string.replace(")", "_excl_")
            new_string = new_string.replace(" ", "")
            self.df.rename(columns={col: new_string}, inplace=True)
        for col in self.df_full:
            new_string = col.replace("<", "less_than")
            new_string = new_string.replace(">", "more_than")
            new_string = new_string.replace(",", "_to_")
            new_string = new_string.replace("[", "from_incl_")
            new_string = new_string.replace("]", "_incl_")
            new_string = new_string.replace("(", "from_excl_")
            new_string = new_string.replace(")", "_excl_")
            new_string = new_string.replace(" ", "")
            self.df_full.rename(columns={col: new_string}, inplace=True)

    def run_optimal_binning_multiprocess(self):
        """
        Run the optimal binning procedure using multiprocessing.

        Returns:
            pandas.DataFrame: Processed DataFrame.
            pandas.DataFrame: Processed DataFrame with under-sampling.
            list: List of columns.
        """
        # TODO: Unit test
        df_all, df_full_all, columns_all = pd.DataFrame(), pd.DataFrame(), []
        pool = Pool(30)
        # install_mp_handler()

        arguments = self.columns
        progress_total = len(arguments)
        progress_counter = 1

        for temp_df, temp_df_full, dummies_columns in pool.map(self.optimal_binning_procedure, arguments):
            if dummies_columns:
                self.df = pd.concat([self.df, temp_df], axis=1)
                if self.params["under_sampling"]:
                    self.df_full = pd.concat([self.df_full, temp_df_full], axis=1)

                self.df = self.df.loc[:, ~self.df.columns.duplicated()].copy()
                if self.params["under_sampling"]:
                    self.df_full = self.df_full.loc[:,~self.df_full.columns.duplicated()].copy()

                # self.df[dummies_columns] = self.df[dummies_columns].fillna(self.df[dummies_columns].mean())
                if self.params["under_sampling"]: 
                    self.df_full[dummies_columns] = self.df_full[dummies_columns].fillna(
                        self.df_full[dummies_columns].mean())

                progress_current = round(progress_counter / progress_total, 2)
                print_and_log(f"[ OPTIMAL BINNING ] Progress:{progress_current} Done added data for: {dummies_columns}", "YELLOW")
                progress_counter+=1

                for col in dummies_columns:
                    columns_all.append(col)

        self.columns = columns_all

        if not self.is_multiclass:
            self.rename_strings_cols_opt_bin()

        return self.df, self.df_full, self.columns


def multiclass_binning_transform_and_get_dummies(df: pd.DataFrame, col: str, kbd):
    dummies = pd.DataFrame()
    df = df.dropna(subset=[col])
    binned_col_name = col + '_binned'
    x_from_col = df[col].to_numpy().reshape(-1, 1)
    x_from_col = kbd.transform(x_from_col).toarray()  # 0 1 2 3

    for ohe_feature_idx in range(1, 4):
        # e.g. 'SECPTT_div_ratio_SEMREGORIG' + "_dummie_" + "0"
        dummies[binned_col_name + '_dummie_' + str(ohe_feature_idx)] = x_from_col[:, ohe_feature_idx]

    df[dummies.columns] = dummies
    print_and_log(
        f'[ OPTIMAL BINNING ] {col} is with the following dummie columns: {list(dummies.columns)}',
        'GREEN')

    return df, dummies
