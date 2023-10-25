import gzip
import json
import pandas as pd
from tqdm import tqdm
import os
def parse_json(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path, save=True):
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Rename columns and add the "name" column
    df.rename(columns={'reviewerName': 'name'}, inplace=True)

    save_path = path.replace('.json', '.csv')
    if save:
        df.to_csv(save_path, index=False)
    return df

def build_id_dict(df):
    i =0
    id_dict = {}

    for id_ in df['asin'].unique():
        id_dict[id_] = i
        i += 1
    return id_dict

def replace_by_id(df):
    id_dict = build_id_dict(df)
    df['asin'] = df['asin'].apply(lambda x: id_dict[x])
    return df


def get_meta_data(path):
    print('Loading meta data...')
    df = pd.DataFrame(columns=['asin', 'title', 'price', 'salesRank'])
    with open(path, 'r') as file:
        for i, line in tqdm(enumerate(file)):
            data = json.loads(line)
            df.loc[i] = [data['asin'], data['title'], data['price'], data['rank']]
    return df

def check_dir(path):
    """
    Check if a directory exists, if not create it
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

def df_to_latex(df, path):
    print(df.to_latex(index=False))
    # Save the latex table
    check_dir(os.path.dirname(path))
    with open(path, 'w') as file:
        file.write(df.to_latex(index=False))