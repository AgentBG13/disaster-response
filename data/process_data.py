import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
        messages_filepath - path to messages.csv
        categories_filepath - path to categories.csv
    OUTPUT:
        df - prepared df
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df

def clean_data(df):
    '''
    INPUT:
        df - preprocessed df
    OUTPUT:
        df - cleaned df
    '''
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)
    for i in range(len(categories.columns)):
        df.drop(df[(df.iloc[:, i + 4] != 0) & (df.iloc[:, i + 4] != 1)].index, inplace=True)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filepath):
    '''
    INPUT:
        df - cleaned df
        database_filepath - path to store db at
    OUTPUT:
        None - save the table to db file
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()