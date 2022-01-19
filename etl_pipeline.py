### Script for creating an ETL pipeline ###

# 'messages.csv' and 'categories.csv' must be in the same directory
# as this script.

# This script will create a SQLite database named data.db and a table
# within that database called LabeledMessages.

### Import libraries and load data ###

# import libraries
import pandas as pd
from sqlalchemy import create_engine
import argparse

# I will include an optional command line flag, '-r'. If it's invoked,
# then the existing table is dropped and the data are used to populate
# a newly created table.
parser = argparse.ArgumentParser()
help_str = 'If -r is used, then the existing table is dropped and the '
help_str += 'data will populate a newly created table.'
parser.add_argument('-r', '--replace', help = help_str, action="store_true")
argument = parser.parse_args()

# Load the `messages` dataset
messages = pd.read_csv('messages.csv')

# Load the `categories` dataset
categories = pd.read_csv('categories.csv')


### Drop duplicate rows ###

# Drop duplicate rows from `messages`
messages.drop_duplicates(inplace=True)

# Drop duplicate rows from `categories`
categories.drop_duplicates(inplace=True)


### Merge dataframes ###

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

# Create SQLAlchemy engine and a SQLite database named data.db
engine = create_engine('sqlite:///data.db')
# Write the dataframe to a table in data.db and
# name the table LabeledMessages.
# If the command-line argument -r is used, then the table is replaced.
# Otherwise, the new data is appended to the existing table.
# The default behavior is to append, not replace.

# If the '-r' flag was not already invoked, then I will ask for user input
# as to whether to replace the existing table or just append to it.
# My reasoning here is that an experienced user will invoke the flag
# if they want the table replaced, but a newer user should be prompted
# to think about the behavior they want.
if not argument.replace:
    input_str = 'Do you want to replace the table? Type \'y\' (lowercase) if you do; \n'
    input_str += 'otherwise just press Enter. The default behavior is to append to the \n'
    input_str += 'table instead of replacing it.\n'
    do_replace = input(input_str)
else:
    do_replace = False

if do_replace == 'y':
    do_replace = True

if argument.replace or do_replace:
    df.to_sql('LabeledMessages', engine, index=False, if_exists='replace')
else:
    df.to_sql('LabeledMessages', engine, index=False, if_exists='append')