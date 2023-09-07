from imblearn.under_sampling import RandomUnderSampler

from src.functions.printing_and_logging import print_and_log


def under_sampling_df_based_on_params(input_df, params):
    """
    Under-sampling procedure for dataframe based on params
    Args:
        input_df:
        params:

    Returns: 2 dataframes - one with the under-sampled df_to_aggregate and one with the original (full)

    """
    is_multiclass = input_df[params['criterion_column']].nunique() > 2
    criterion_rate = input_df[params['criterion_column']].value_counts(dropna=False, normalize=True)
        
    input_df_full = input_df.copy()
    print_and_log(f'[ UNDERSAMPLING ] Starting under-sampling with strategy: {params["under_sampling"] if not is_multiclass else "auto"}. '
                  f'The initial dataframe length is {input_df.shape} and criterion rate:\n{criterion_rate}', 'GREEN')

    # define strategy for under-sampling
    under = RandomUnderSampler(sampling_strategy=params["under_sampling"] if not is_multiclass else 'auto')
    # create new df_to_aggregate (under_X)  under-sampled based on above strategy
    under_x, under_y = under.fit_resample(input_df, input_df[params['criterion_column']])
    input_df = under_x

    criterion_rate = input_df[params['criterion_column']].value_counts(dropna=False, normalize=True)
    print_and_log(f'[ UNDERSAMPLING ] Under-sampling done. The new dataframe length is {input_df.shape} and '
                  f'criterion rate:\n{criterion_rate}', 'GREEN')
    return input_df, input_df_full