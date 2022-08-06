from imblearn.under_sampling import RandomUnderSampler

from src.functions.printing_and_logging import print_and_log


def under_sampling_df_based_on_params(input_df, params):
    """
    Under-sampling procedure for dataframe based on params
    Args:
        input_df:
        params:

    Returns: 2 dataframes - one with the under-sampled df and one with the original (full)

    """
    criterion_rate = round(
        input_df[params['criterion_column']].sum() / input_df[params['criterion_column']].count(),
        2)
    input_df_full = input_df.copy()
    print_and_log(f'\n[ UNDERSAMPLING ] Starting under-sampling with strategy: {params["under_sampling"]}. '
                  f'The initial dataframe length is {input_df.shape} and criterion rate: {criterion_rate}', 'GREEN')

    # define strategy for under-sampling
    under = RandomUnderSampler(sampling_strategy=params["under_sampling"])
    # create new df (under_X)  under-sampled based on above strategy
    under_x, under_y = under.fit_resample(input_df, input_df[params['criterion_column']])
    input_df = under_x

    criterion_rate = round(
        input_df[params['criterion_column']].sum() / input_df[params['criterion_column']].count(),
        2)
    print_and_log(f'[ UNDERSAMPLING ] Under-sampling done. The new dataframe length is {input_df.shape} and '
                  'criterion rate: {criterion_rate}', 'GREEN')
    return input_df, input_df_full