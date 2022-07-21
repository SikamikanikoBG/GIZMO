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