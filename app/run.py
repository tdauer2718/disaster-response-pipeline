# Imports
import sys
# Make functions in files in the '/models' folder available
# Make sure when you run this program, your current directory is
# the one above the one this script is in
sys.path.insert(1, './models')

import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
import pickle
from sqlalchemy import create_engine
import re

# Import a functions and objects needed for my tokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# The pickled model needs the tokenize function available or I'll
# get an error
from train_classifier import tokenize

# Instantiate a web app object
app = Flask(__name__)

# Load the data from my database
engine = create_engine('sqlite:///data/data.db')
df = pd.read_sql_table('LabeledMessages', engine)
ratios_df = pd.read_sql_table('ClassRatios', engine)

# Load the model trained on all the data
with open('./models/final_model.pkl' , 'rb') as f:
    model = pickle.load(f)

# Create the index webpage, which displays some visualizations and
# can take user input text and classify it according to the predefined
# categories
@app.route('/')
@app.route('/index')
def index():
    """Creates a template for the index webpage, including plots."""

    # Make a list with the number of messages in each genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Make a list of the fraction of instances in Class 1 for each category
    class_ratios = ratios_df['class_ratio']

    
    # Create visualizations
    # First, define a dictionary with information for each plot
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of message genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=class_ratios,
                    nbinsx=50
                )
            ],

            'layout': {
                'title': 'Fraction of messages in each category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Fraction"
                }
            }
        },
    ]
    
    # Next, encode the graphs in JSON format
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Finally, render the webpage with the Plotly graphs above
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Create a webpage that handles the user query and displays results from the model
@app.route('/go')
def go():
    """Creates a template for the '/go' webpage."""
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """Runs the web app when called."""
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()