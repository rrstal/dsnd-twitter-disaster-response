import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads data
    
    Args:
        messages_filepath (str): Path to the file with messages 
        categories_filepath (str): Path to the file with categories

    Returns:
        df (pandas.DataFrame): Merged data
    """
    msgs = pd.read_csv(messages_filepath)
    cats = pd.read_csv(categories_filepath)
    merged_df = pd.merge(msgs, cats, left_on='id', right_on='id')
    
    return merged_df



def clean_data(df):
    """Cleans the data
    
    Args:
        df (pandas.DataFrame): all input data
        
    Returns:
        df (pandas.DataFrame): clean input data
    """
    # 1. create a dataframe containing the 36 individual category columns
    # 2. select the first row of the `categories` df and use it to extract a list of new column names for the categories.
    # 3. rename the columns of `categories` df

    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    
    # 1. set each value to be the last char of the string
    # 2. convert each column from string to numeric format
    # 3. for each category, update the value in the column to 1

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
        categories.loc[categories[column]>1,column] = 1
        
    # 1. drop the original categories column
    # 2. concatenate the original dataframe with the new `categories` dataframe
    # 3. drop duplicates
    df.drop(columns=['categories'], inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """Saves data
    
    Args:
        df (pandas.DataFrame): data to be saved
        database_filename (str): output database name
    """
   
    engine = create_engine('sqlite:///' + database_filename)
    # save to database and overwrite the existing one
    df.to_sql('twitter-data', engine, if_exists='replace', index=False)


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