from imblearn.under_sampling import RandomUnderSampler

from src.functions.printing_and_logging import print_and_log


def under_sampling_df_based_on_params(input_df, params):
    """
    Perform under-sampling on a DataFrame based on specified parameters.

    Steps:
    1. Check if the dataset is multi-class based on the criterion column.
    2. Calculate the criterion rate in the dataset.
    3. Create a copy of the input DataFrame as the full DataFrame.
    4. Log the start of the under-sampling process with strategy and initial dataset information.
    5. Define the under-sampling strategy based on the multi-class status.
    6. Perform under-sampling to create a new under-sampled DataFrame.
    7. Update the input DataFrame with the under-sampled data.
    8. Calculate the criterion rate in the updated DataFrame.
    9. Log the completion of under-sampling with the new dataset information.

    Parameters:
    - input_df: DataFrame, input DataFrame to under-sample
    - params: dict, parameters for under-sampling

    Returns:
    - input_df: DataFrame, under-sampled DataFrame
    - input_df_full: DataFrame, original full DataFrame
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