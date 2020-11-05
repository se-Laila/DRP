import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

#from sklearn.externals import joblib
from sqlalchemy import create_engine
import joblib

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")

       
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Visuals
    # extract data needed for visuals
    # 1. display the genres of the messages with their count (given)
    genre_counts = (df.groupby('genre').count()['message']).sort_values(ascending = False) 
    genre_names = list(genre_counts.index)
    
    # 2. display the categories of the messages with their count 
    # extract the categories only by dropping all the other columns
    category_msg = (df.drop(columns=['id','message','original','genre']).sum()).sort_values(ascending = False) 
    category_count = list(category_msg.index)
    
    # 3. display the count of messages that have specific words (this can be enhanced further to allow the user to search for specific words)
    # a dictionary of critical words
    critical_word = {'emergency', 'urgant', 'danger', 'death', 'crisis'} 
    tokenized_text = [] 
    
    # tokanize the message
    for text in df['message'].values:
        tokenized_ = tokenize(text)
        tokenized_text.extend(tokenized_)

   # a dictionary of the critical words and their counts (the key us the word and the value is the count)
    words_dict={}

    for word in tokenized_text:
        if word in critical_word:
            words_dict[word] = words_dict.get(word,0) + 1
    

    critical_word = list(words_dict.keys())
    critical_word_count = list(words_dict.values())
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
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
                Bar(
                    x = category_count,
                    y = category_msg
                )
            ],

            'layout': {
                'title': 'Messages Distribution per Category',
                'yaxis': {
                    'title': "Messages Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x = critical_word,
                    y = critical_word_count
                )
            ],

            'layout': {
                'title': 'Messages Distribution per Category',
                'yaxis': {
                    'title': "Messages Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
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