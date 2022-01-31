### Script for creating an ETL pipeline ###

# This script will create a SQLite database and a table
# within that database called LabeledMessages.

### Import libraries and load data ###

# import libraries``
import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """Loads message and category data from CSV files.

    Args:
        messages_filepath -- a string that gives the path of a CSV
            file from which to load message data.
        categories_filepath -- a string that gives the path of a CSV
            file from which to load category data corresponding to
            the message data.
    Returns:
        messages -- a Pandas DataFrame object that contains messages
            along with some other data for each message included
            in the file.
        categories -- a dataframe that gives categories for each
            message.
    """
    # Load the `messages` dataset
    messages = pd.read_csv(messages_filepath)

    # Load the `categories` dataset
    categories = pd.read_csv(categories_filepath)

    return messages, categories

def clean_data(messages, categories):
    """Cleans the messages and categories data by dropping duplicates,
        merging dataframes, and naming columns appropriately.
    
    Args:
        messages -- a dataframe containing messages and some other data,
            created using the `load_data()` function from a disaster
            response messages file.
        categories -- a dataframe containing the categories for each 
            message.
    Returns:
        df -- a dataframe with the combined message and categories data.
    """
    ### Drop duplicate rows###

    # Drop duplicate rows from `messages`
    messages.drop_duplicates(inplace=True)

    # Drop duplicate rows from `categories`
    categories.drop_duplicates(inplace=True)
    
    # In my Jupyter notebook ("ETL Pipeline Preparation.ipynb") I 
    # found that a few dozen messages have more than one corresponding
    # row in the `categories` dataframe - a data labeling mistake, so
    # I'll remove these messages.
    # Find the ids of messages to remove:
    ids_to_remove = categories[categories['id'].duplicated(keep=False)]['id']
    # Drop the appropriate rows from `categories` and `messages`
    categories.drop(categories[categories['id'].isin(ids_to_remove)].index, inplace=True)
    messages.drop(messages[messages['id'].isin(ids_to_remove)].index, inplace=True)

    ### Split the category data into separate columns ###

    # Split the 'categories' column on semicolons
    categories['categories'] = categories['categories'].str.split(';')

    # Get the category names from the first row of the dataframe
    cat_names = categories['categories'].iloc[0]
    # Strip the last two characters of each element to find the category names
    cat_names = [c[:-2] for c in cat_names]

    # Replace the 'categories' column entries with lists of the numbers alone (no text),
    # being sure to convert them from strings to integers
    categories['categories'] = categories['categories'].apply(
        lambda x: [int(s[-1]) for s in x])

    # Split the `categories` column lists into different columns and use
    # the `cat_names` to name them
    df_cat = pd.DataFrame(list(categories['categories']), index=categories.index,
                columns=cat_names)
    df_cat.head()

    # Drop the old 'categories' column from the `categories` dataframe
    categories.drop(columns='categories', inplace=True)
    # Concatenate the new columns to `categories`
    categories = pd.concat([categories, df_cat], axis=1)

    ### Join the dataframes ###

    # Join the dataframes on their common ids
    df = messages.merge(categories, how='inner', on='id')

    ### Prepare for binary classification ###

    # Figure out which category columns (which have column names in 
    # `cat_names`) have some label that's not in {0, 1}
    cols = []
    for c in cat_names:
        unique_vals = set(df[c].unique())
        if not unique_vals.issubset({0, 1}):
            cols.append(c)
            print(f'Values in \'{c}\' column: {unique_vals}')

    # Drop the rows that have values not in {0, 1} for any of the categories
    # Find the indices of rows to drop
    for c in cols:
        to_drop = df[~df[c].isin({0, 1})].index
    # Drop the rows
    df.drop(to_drop, inplace=True)

    return df

def save_data(df, database_filename):
    """Saves the data to a table in a SQLite database.
    
    Args:
        df -- the dataframe to save.
        database_filename -- the filename of the database to save
            to; should have the extension '.db'. The data will be
            saved to the table 'LabeledMessages' in this database.
        
    Returns:
        None.
    """
    # Create SQLAlchemy engine and a SQLite database
    engine = create_engine(f'sqlite:///{database_filename}')
    # Write the dataframe to a table in the database and
    # name the table LabeledMessages.
    df.to_sql('LabeledMessages', engine, index=False, if_exists='replace')

def main():
    """Executes the ETL pipeline. Loads data from two CSV files, cleans and 
    combines the data into one dataframe, and saves it to a SQLite database."""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...')
        print(f'Messages: {messages_filepath}')
        print(f'Categories: {categories_filepath}')
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...')
        print(f'Database: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Data has been saved to the database')
    
    else:
        print_str = 'Provide the filepaths of the messages and categories '
        print_str += 'datasets as the first and second arguments, respectively, '
        print_str += 'as well as the filepath of the database where you want '
        print_str += 'to save your cleaned data.'
        print_str += '\n\nExample: python data/process_data.py data/messages.csv '
        print_str += 'data/categories.csv data/data.db'
        print_str += '\n\nwhere this is run from the top-level directory (one above this script)'
        print(print_str)

if __name__ == '__main__':
    main()