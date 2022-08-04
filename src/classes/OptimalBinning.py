from multiprocessing import Pool

import pandas as pd
from optbinning import OptimalBinning
from sklearn.model_selection import train_test_split

from src import print_and_log


# from multiprocessing_logging import install_mp_handler


class OptimaBinning:
    def __init__(self, df, df_full, columns, criterion_column, final_features, observation_date_column, params):
        self.df = df
        self.df_full = df_full
        self.columns = columns
        self.criterion_column = criterion_column
        self.final_features = final_features
        self.observation_date_column = observation_date_column
        self.params = params

    def optimal_binning_procedure(self, col):
        # creating binned dummie features from all numeric ones
        temp_df = self.df.copy()
        temp_df2 = self.df.copy()
        temp_df_full = self.df_full.copy()

        # Removing all periods before splitting train and test
        temp_df2 = temp_df2[temp_df2[self.observation_date_column] != self.params['t1df']]
        temp_df2 = temp_df2[temp_df2[self.observation_date_column] != self.params['t2df']]
        temp_df2 = temp_df2[temp_df2[self.observation_date_column] != self.params['t3df']]

        x_train, _, y_train, _ = train_test_split(
            temp_df2, temp_df2[self.criterion_column], test_size=0.33, random_state=42)
        x_train = x_train.dropna(subset=[col])

        x = x_train[col].values
        y = x_train[self.criterion_column].values
        optb = OptimalBinning(name=col, dtype='numerical', solver='cp', max_n_bins=3, min_bin_size=0.1)
        optb.fit(x, y)

        temp_df = temp_df.dropna(subset=[col])
        binned_col_name = col + '_binned'
        temp_df[binned_col_name] = optb.transform(temp_df[col], metric='bins')

        dummies = pd.get_dummies(temp_df[binned_col_name], prefix=binned_col_name + '_dummie')
        #print_and_log('{} is with the following splits: {} and dummie columns: {}'.format(col, optb.splits,
        #                                                                                 list(dummies.columns)),
        #              '')
        temp_df[dummies.columns] = dummies

        if self.params['under_sampling']:
            temp_df_full = temp_df_full.dropna(subset=[col])
            binned_col_name = col + '_binned'
            temp_df_full[binned_col_name] = optb.transform(temp_df_full[col], metric='bins')

            dummies = pd.get_dummies(temp_df_full[binned_col_name], prefix=binned_col_name + '_dummie')
            temp_df_full[dummies.columns] = dummies

        dummies_list = list(dummies.columns)
        return temp_df[list(dummies.columns)], temp_df_full[list(dummies.columns)], dummies_list

    def rename_strings_cols_opt_bin(self):
        # Recoding strings
        print(">>>>>>>>>>>>>>>>>>>>", type(self.columns), len(self.columns), self.columns)
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
        df_all, df_full_all, columns_all = pd.DataFrame(), pd.DataFrame(), []
        pool = Pool(10)
        # install_mp_handler()

        arguments = self.columns

        for col in arguments:
            temp_df, temp_df_full, dummies_columns = self.optimal_binning_procedure(col)
            #pool.close()
            #pool.join()
        #   for temp_df, temp_df_full, dummies_columns in pool.map(self.optimal_binning_procedure, arguments):

            if dummies_columns:
                self.df = pd.concat([self.df, temp_df], axis=1)
                self.df_full = pd.concat([self.df_full, temp_df_full], axis=1)

                self.df = self.df.loc[:, ~self.df.columns.duplicated()].copy()
                self.df_full = self.df_full.loc[:, ~self.df_full.columns.duplicated()].copy()

                self.df[dummies_columns] = self.df[dummies_columns].fillna(0)
                self.df_full[dummies_columns] = self.df_full[dummies_columns].fillna(0)

                for col in dummies_columns:
                    columns_all.append(col)

        self.columns = columns_all

        self.rename_strings_cols_opt_bin()

        return self.df, self.df_full, self.columns
