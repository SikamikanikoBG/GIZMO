def run(df):
    """
    Process the input DataFrame by creating a subset based on a condition and printing the number of rows.

    Args:
    df (pd.DataFrame): Input DataFrame containing financial data.

    Returns:
    pd.DataFrame: DataFrame with subset based on the condition.
    """
    # df = df[df["Direct_Active"] > 0].copy()
    print(f"[ CUSTOM CALCULATIONS ] Creating subset. New nb of rows in the table {len(df)}")
    return df


def calculate_criterion(df, predict_module):
    """
   Calculate the criterion based on a condition for the input DataFrame.

   Args:
   df (pd.DataFrame): Input DataFrame containing financial data.
   predict_module: Not used in the function.

   Returns:
   pd.DataFrame: DataFrame with added 'criterion_NbDirectApp_2M' column.
   """
    # df['criterion_NbDirectApp_2M'] = np.where((df['NbDirectApp_2M'] > 0), 1, 0)
    return df
