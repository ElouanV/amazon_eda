import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from tqdm import tqdm

def compress_data(database, association_rules, threshold=0.7):
    """
    Compress the database using association rules.
    :param database: pandas.DataFrame: the database to compress
    :param association_rules: pandas.DataFrame: the association rules
    :param threshold: float: the minimum confidence threshold
    :return: pandas.DataFrame: the compressed database
    """
    compressed_data = []
    # Keep only rules with confidence above threshold
    association_rules = association_rules[association_rules['confidence'] > threshold]
    anteriors = association_rules['antecedents'].apply(lambda x: list(x)).tolist()
    consequents = association_rules['consequents'].apply(lambda x: list(x)).tolist()
    print(f'Number of anteriors:  {len(anteriors)}')

    nb_items_compressed = 0
    # Iterate over transactions without header
    for i in tqdm(range(0, len(database))):
        transaction = database.iloc[i, :].dropna().tolist()
        user = transaction[0]
        items = transaction[1]
        compressed_transaction = []

        # Check if an itemset is in the transaction
        for (anterior, consequent) in zip(anteriors, consequents):
            pattern = anterior + consequent
            if all(item in items for item in pattern):
                compressed_transaction.extend(anterior)
                items = [item for item in items if item not in pattern]
                nb_items_compressed += len(pattern)
        compressed_transaction.extend(items)

        compressed_data.append([user, compressed_transaction])
    print(f'Number of items compressed: {nb_items_compressed}')
    print(f'Compression ratio: {nb_items_compressed / len(database)}')
    print(f'Original database information: {database.info()}')
    compressed_data = pd.DataFrame(compressed_data, columns=['reviewerID', 'items'])
    print(f'Compressed database information: {compressed_data.info()}')

    return compressed_data

def decompress_data(compressed_data, association_rules):
    """
    Decompress the database using association rules.
    :param compressed_data: pandas.DataFrame: the compressed database
    :param association_rules: pandas.DataFrame: the association rules
    :return: pandas.DataFrame: the decompressed database
    """
    decompressed_data = []
    # Keep only rules with confidence above threshold
    anteriors = association_rules['antecedents'].apply(lambda x: list(x)).tolist()
    consequents = association_rules['consequents'].apply(lambda x: list(x)).tolist()
    print(f'Number of anteriors:  {len(anteriors)}')

    nb_items_decompressed = 0
    # Iterate over transactions without header
    for i in tqdm(range(0, len(compressed_data))):
        transaction = compressed_data.iloc[i, :].dropna().tolist()
        user = transaction[0]
        items = transaction[1]
        decompressed_transaction = []

        # Check if an itemset is in the transaction
        for (anterior, consequent) in zip(anteriors, consequents):
            pattern = anterior + consequent
            if all(item in items for item in anterior):
                decompressed_transaction.extend(pattern)
                items = [item for item in items if item not in anterior]
                nb_items_decompressed += len(pattern)
        decompressed_transaction.extend(items)

        decompressed_data.append([user, decompressed_transaction])
    print(f'Number of items decompressed: {nb_items_decompressed}')
    print(f'Decompression ratio: {nb_items_decompressed / len(compressed_data)}')
    print(f'Original database information: {compressed_data.info()}')
    decompressed_data = pd.DataFrame(decompressed_data, columns=['reviewerID', 'items'])
    print(f'Decompressed database information: {decompressed_data.info()}')

    return decompressed_data


DATA_DIR = '../data'

df = pd.read_csv(f'{DATA_DIR}/Sports_and_Outdoors_5_2016_2018.csv')
df = df[['reviewerID', 'asin']]
df = df.drop_duplicates()
df = df.head(200_000)
df = df.groupby('reviewerID')['asin'].apply(list).reset_index(name='items')
df['items'] = df['items'].apply(lambda x: list(set(x)))

te = TransactionEncoder()
te_ary = te.fit(df['items']).transform(df['items'])
df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
import time

start = time.time()
frequent_itemsets = apriori(df_transactions, min_support=0.002, use_colnames=True)
end = time.time()
print(f'Apriori took {end - start} seconds')

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules.sort_values(by='confidence', ascending=False)

compressed_data = compress_data(df, rules, threshold=0.7)
# Save the compressed data
df.to_csv(f'{DATA_DIR}/Sports_and_Outdoors_5_2016_2018_not_compressed.csv', index=False)
compressed_data.to_csv(f'{DATA_DIR}/Sports_and_Outdoors_5_2016_2018_compressed.csv', index=False)
decompressed_data = decompress_data(df, rules)


