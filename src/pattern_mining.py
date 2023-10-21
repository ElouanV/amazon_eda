from skmine.itemsets import LCM


def create_and_fit_lcm(df, supp_ratio=0.5):
    """
    Create and fit an LCM model.
    :param df:  pandas.DataFrame: the dataframe to fit the model on
    :param supp_ratio: float: the minimum support ratio
    :return: skmine.itemsets.LCM: the fitted model
    """
    min_supp = int(supp_ratio * len(df))
    lcm = LCM(min_supp=min_supp)
    lcm.fit(df)
    return lcm