# Import modules
import json
import plotly
import pandas as pd
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

from figures import return_figures

# Instansiate flask app
app = Flask(__name__)

def tokenize(text):
    """ tokenize text from user input

    Args:
        text (string): message from webapp user

    Returns:
        list (string): list of tokens
    """
    # create word tokens
    tokens = word_tokenize(text)
    # initialize lemmetizer
    lemmatizer = WordNetLemmatizer()
    # initialize tokens list
    clean_tokens = []
    for tok in tokens:
        # lemmetize, normalize and strip words for space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # add to tokens list
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data from sql database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('coded_responses', engine)

# load model
model = pickle.load(open("../models/classifier.pkl", 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """ run when user hits / or /index, serves html with data.

    Args:
        None

    Returns:
        master.html (string): name of teplate to use
        ids (list): list of figure ids
        graphJSON (json): plotly figures
    """
    figures = return_figures(df)
    # plot ids for the html id tag
    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]
    # plot figures
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=figuresJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """ function to be called when user enters text to test classifier

    Args:
        None

    Returns:
        go.html (string): name of html to render
        query (string): the message sent by user
        classification_result (dict): result from classification
    """
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
