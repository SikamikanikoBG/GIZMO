def raw_features_to_list(final_features):
    """
    Extract raw features from a list of final features.

    Steps:
    1. Initialize an empty list to store raw features.
    2. Iterate over each feature in the final features list.
    3. If the feature contains 'binned':
        - Extract the prefix before '_binned'.
        - If the prefix contains '_ratio_':
            - Extract the components based on different ratio types ('_div_', '_add_', '_subs_', '_mult_').
        - Otherwise, add the prefix to the raw features list.
    4. If the feature contains '_ratio_':
        - Extract the components based on different ratio types ('_div_', '_add_', '_subs_', '_mult_').
    5. Otherwise, add the feature to the raw features list.
    6. Remove duplicates from the raw features list.
    7. Return the list of raw features.

    Parameters:
        final_features: list of str, final list of features

    Returns:
        raw_features: list of str, extracted raw features
    """

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