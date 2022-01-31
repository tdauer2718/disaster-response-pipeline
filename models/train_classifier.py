### Script for training, evaluating, and saving a classifier ###

# Imports
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import re
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download the 'punkt' package for sentence tokenization
nltk.download(['punkt', 'wordnet', 'stopwords'])

# Import a tokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def load_data(database_filepath):
    '''Loads data from a database and splits it into features and targets.

        Args:
            datbase_filepath -- a string that gives the filepath for the database
                to load. The class ratios will also be written the the table
                ClassRatios in this database.
        Returns:
            X -- a Pandas Series object containing messages to the disaster relief 
                team.
            Y -- a Pandas DataFrame object containing targets, i.e. categories
                    of disaster response message, each in its own column,
                    corresponding to the messages in X.
    '''
    # Load the data from the database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM LabeledMessages", engine)
    # Create the series of messages
    X = df['message']
    # Create the dataframe of targets (each column is a target/category)
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])

    # Some of the messages become the empty string after applying the 
    # tokenizer and removing stopwords; these will cause an error when using 
    # `CountVectorizer`. I will therefore check for such messages explicitly 
    # and remove the corresponding rows. I'll do this by making each message 
    # into a set of tokenized words, subtracting the set of stopwords 
    # (subtracting as sets is faster than using a list-based approach), 
    # and then seeing if the result is the empty set. Alternatively, I could use 
    # `try`-`except` to see on which rows the `CountVectorizer` with my 
    # tokenizer and stop words fails, but that approach is rather wasteful 
    # since I don't need to do vectorization on rows that don't return the 
    # empty set after tokenization and stopword removal.
    # Define some stopwords:
    my_stopwords = set([tokenize(w)[0] for w in stopwords.words('english')])
    # Compute the result after tokenization and stopword removal on the messages
    X_check = X.apply(lambda x: set(tokenize(x)) - my_stopwords)
    # Find the resulting rows
    failed_rows = X_check[X_check == set()].index
    # Drop the appropriate row(s)
    X.drop(failed_rows, inplace=True)
    Y.drop(failed_rows, inplace=True)
    

    # I want to save the class ratio, which here I define as the fraction of all
    # instances that are in Class 1, for each column for this cleaned data, for later. 
    # To do this, I'll create a dataframe with those fractions now and save it to 
    # another table in the database.
    cat_names = list(Y.columns) # category names
    class_ratios = [] # list to hold the class ratios
    for i, c in enumerate(Y.columns):
        #n_class_0 = len(Y[Y[c] == 0])
        n_class_1 = len(Y[Y[c] == 1])
        class_ratios.append(n_class_1/len(Y))
    ratios_df = pd.DataFrame({'category': cat_names, 'class_ratio': class_ratios})
    # Write the dataframe to a table in the database from above and
    # name the table 'ClassRatios'.
    ratios_df.to_sql('ClassRatios', engine, index=False, if_exists='replace')

    # Return the loaded, split, and cleaned data
    return X, Y


def tokenize(text):
    """Takes some text as input, then tokenizes it, lemmatizes it, strips it
        of characters that are not letters or numbers, and normalizes its case 
        (to lowercase only).
        
    Args:
        text -- a string of text.
    
    Returns:
        tokens_cleaned -- the final list of cleaned tokens.
    """
    text = text.lower()
    # Remove characters that aren't letters or numbers
    text = re.sub('[^a-z0-9]', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Instantiate a lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Lemmatize and strip the tokens
    tokens_cleaned = [lemmatizer.lemmatize(t).strip() for t in tokens]
    return tokens_cleaned


def build_model():
    """Builds a model as an sklearn Pipe object using a CountVectorizer,
        TF.IDF transformer, and a Random Forest Classifier for each target.

    Args:
        None
    Returns:
        pipe -- the model (not yet fit to data) as an sklearn Pipe object.
    """
    # Create stopwords that are consistent with my tokenizer
    my_stopwords = list(set([tokenize(w)[0] for w in stopwords.words('english')]))
    # Create a Pipeline that uses a CountVectorizer with my above tokenizer
    # function, then uses a TF.IDF transformer, then uses an independent
    # random forest classifier for each output (utilizing MultiOutputClassifier())
    pipe = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, stop_words=my_stopwords)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42))),
        ])
    best_params_found = {'clf__estimator__class_weight': 'balanced',
                        'clf__estimator__min_samples_split': 3,
                        'clf__estimator__n_estimators': 100,
                        'tfidf__norm': 'l2',
                        'tfidf__use_idf': True,
                        'vect__max_features': None}
    pipe.set_params(**best_params_found)
    return pipe


def evaluate_model(model, X_test, Y_test):
    """Evaluates the fitted model performance on the test set.
    
    Args:
        model -- a fitted model that should be an sklearn object with a
            `predict` method, such as a Pipe.
        X_test -- a test set of messages. Should be a Pandas Series object.
        Y_test -- the test set of labels in each category for the messages 
            in X_test. Should be a Pandas DataFrame with number of rows equal
            to the number of rows of X_test and number of columns equal to
            the number of targets.
    Returns: 
        None, but prints the F1-score, recall, and precision for predictions
            on each class, in addition to the ratio of instances in Class 1
            to Class 0 in the test set.
    """
    # Make predictions on the test set using the trained model
    Y_pred = model.predict(X_test)
    # Show the recall and precision for test set predictions for each target
    # Also show the class imbalance, i.e. the ratio of instances in Class 1 to Class 0
    # in the test set
    print('Here are the results for each category (for Class 1):')
    for i, col in enumerate(Y_test.columns):
        print(f'\033[1m{col}:\033[0m')
        report = classification_report(Y_test[col], Y_pred[:, i], 
                                    zero_division=0, output_dict=True)
        if '0' not in report.keys():
            # In this case there are no instances in Class 0
            n_class_0 = 0
        else:
            n_class_0 = report['0']['support']
        try:
            recall = report['1']['recall']
            precision = report['1']['precision']
            f1 = report['1']['f1-score']
            frac = report['1']['support']/(n_class_0 + report['1']['support'])
        except KeyError:
            # In this case there are no instances in Class 1
            recall, precision, f1, frac = 0, 0, 0, 0
        print_str = f'F1-score = {f1:.3g}; recall = {recall:.3g}; '
        print_str += f'precision = {precision:.3g}; class_1_fraction = {frac:.3g}'
        print(print_str)
        print('-'*50)


def save_model(model, model_filepath):
    """Pickles the model and saves it.

    Args:
        model -- a fitted model from sklearn.
        model_filepath -- the filepath where the model is to be saved.
    Returns:
        None
    """
    # Pickle the model and save to it to model_filepath
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """Executes the machine learning pipeline on the data in the database.
        Loads the data, builds the model, does a train-test split, trains 
        the model on the train data, evaluates the model on the test data, 
        fits the model on the full dataset, and finally pickles and saves 
        the fully trained model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data from database {database_filepath} and preparing it...')
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model (this could take a few minutes)...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Training model on full dataset (post testing - this could take a few minutes):')
        model.fit(X, Y)

        print(f'Saving model to {model_filepath}')
        save_model(model, model_filepath)
        print('Fully trained model saved')

    else:
        print_str = 'Provide the filepath of the prepared disaster messages database '
        print_str += 'as the first argument and the filepath where you want '
        print_str += 'to save the pickle file for the trained model as the '
        print_str += 'second argument.'
        print_str += '\n\nExample: python models/train_classifier.py data/data.db models/final_model.pkl'
        print_str += '\n\nwhere this is run from the top-level directory (one above this script)'
        print(print_str)

if __name__ == '__main__':
    main()