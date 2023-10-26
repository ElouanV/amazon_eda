import gzip
import json
import pandas as pd

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
        df.to_csv(save_path, index=False, escapechar='\\')
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