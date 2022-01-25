### Script for creating an ETL pipeline ###

# This script will create a SQLite database and a table
# within that database called LabeledMessages.

### Import libraries and load data ###

# import libraries
import pandas as pd
from sqlalchemy import create_engine
import argparse
import sys
from threading import Timer

def load_data(messages_filepath, categories_filepath):
    # Load the `messages` dataset
    messages = pd.read_csv(messages_filepath)

    # Load the `categories` dataset
    categories = pd.read_csv(categories_filepath)

    return messages, categories

def clean_data(messages, categories):
    # Drop duplicate rows from `messages`
    messages.drop_duplicates(inplace=True)

    # Drop duplicate rows from `categories`
    categories.drop_duplicates(inplace=True)

    # Join the dataframes on their common ids
    df = messages.merge(categories, how='inner', on='id')

    ### Split the category data into separate columns ###

    # Split the 'categories' column on semicolons
    df['categories'] = df['categories'].str.split(';')

    # Get the category names from the first row of the dataframe
    cat_names = df['categories'].iloc[0]

    # Strip the last two characters of each element to find the category names
    cat_names = [c[:-2] for c in cat_names]

    # Replace the 'categories' column entries with lists of the numbers alone (no text),
    # being sure to convert them from strings to integers
    df['categories'] = df['categories'].apply(
        lambda x: [int(s[-1]) for s in x])

    # Split the `categories` column lists into different columns and use
    # the `cat_names` to name them
    df_cat = pd.DataFrame(df['categories'].to_list(), index=df.index,
                columns=cat_names)

    # Drop the old 'categories' column from df
    df.drop(columns='categories', inplace=True)
    # Concatenate the new columns to df
    df = pd.concat([df, df_cat], axis=1)

    return df

def save_data(df, database_filename, do_replace):
    # Create SQLAlchemy engine and a SQLite database
    engine = create_engine(f'sqlite:///{database_filename}')
    # Write the dataframe to a table in the database and
    # name the table LabeledMessages.

    if do_replace:
        df.to_sql('LabeledMessages', engine, index=False, if_exists='replace')
    else:
        df.to_sql('LabeledMessages', engine, index=False, if_exists='append')

def main():
    if len(sys.argv) == 5:

        messages_filepath, categories_filepath, database_filepath, do_replace = sys.argv[1:]

        print('Loading data...')
        print(f'Messages: {messages_filepath}')
        print(f'Categories: {categories_filepath}')
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        if do_replace == 'r':
            do_replace = True
        else:
            do_replace = False
        print('Saving data...')
        print(f'Database: {database_filepath}')
        save_data(df, database_filepath, do_replace)
        
        print('Data has been saved to the database')
    
    else:
        print_str = 'Provide the filepaths of the messages and categories '
        print_str += 'datasets as the first and second arguments, respectively, '
        print_str += 'as well as the filepath of the database where you want '
        print_str += 'to save your cleaned data. For the last argument, type \'r\' '
        print_str += 'if you want to replace the database, and any other character, '
        print_str += 'such as \'a\' if you want to append to it.'
        print_str += '\n\nExample: python etl_pipeline.py messages.csv categories.csv '
        print_str += 'data.db r'
        print(print_str)

if __name__ == '__main__':
    main()