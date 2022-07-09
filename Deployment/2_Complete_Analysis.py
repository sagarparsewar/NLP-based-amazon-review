import pandas as pd
import plotly.express as px

import dash
from dash import dcc
from dash import html
from dash import dash_table
import dash.dependencies as dd
from dash.dependencies import Input, Output

import re
import numpy as np
import spacy
from wordcloud import WordCloud
from nltk import tokenize
from nltk.probability import FreqDist
from io import BytesIO
import base64
from collections import Counter 
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex


df_reviews_ratings=pd.DataFrame()
df_reviews_ratings = pd.read_csv('Amazon_reviews.csv')
# Reading txt file of Positive words :
with open("C:\\Users\\Mani\\EXCELR\\Projects\\NLP_AmazonReview\\NLP_scr\\positive-words.txt","r") as pos:
      positive_words = pos.read().split("\n")

positive_words = positive_words[36:]
# Reading txt file of Negative words:
# Reading txt file
with open("C:\\Users\\Mani\\EXCELR\\Projects\\NLP_AmazonReview\\NLP_scr\\negative-words.txt","r") as pos:
      negative_words = pos.read().split("\n")

negative_words = negative_words[35:]

app = dash.Dash(__name__)

layout_colors = {
    'headings' : '#B8255F',
    'sub_headings' : '#AF38EB',
    'Main_heading' : '#008080'
}

app.layout = html.Div(
    children=[
       html.H1('Enter the product URL : ',
                    style = {  'textAlign' : 'center',
                    'color' : layout_colors['Main_heading']}),
        html.Br(),
        dcc.Input(
            id="user_entered_url",
            placeholder = 'Enter the URL ',
            type = 'url',
            style = {'width' : '50%' , 'margin-left': '350px'},
            value = ''
        ),
        html.Br(),
        html.Button(id='URL_Submit_button', n_clicks=0, children="Submit" ,
                       style = {'background-color': 'white',
                                    'color': 'black',
                                    'height': '50px',
                                    'width': '100px',
                                    'margin-top' : '10px',
                                    'margin-left': '700px'}),
        html.Br(),
        html.Br(),
        html.H2('The Extracted Data Sample is :' ,  
                    style = {  'textAlign' : 'center',
                                    'color' : layout_colors['headings']}
                ),
        dash_table.DataTable(id='Extracted_Data',
                        style_data={
                                    'whiteSpace': 'normal',
                                    },
                        css=[{
                                'selector': '.dash-spreadsheet td div',
                                'rule': '''
                                    line-height: 15px;
                                    max-height: 30px; min-height: 30px; height: 30px;
                                    display: block;
                                    overflow-y: hidden;
                                '''
                             }],
                             style_cell={'textAlign': 'left'} # left align text in columns for readability

        ),
        html.H2('Rating Overview :' ,
                    style = {
                    'textAlign' : 'center',
                    'color' : layout_colors['headings']}
                    ),
        html.Div(
            dcc.Graph(id='Rating_Overview',
                    figure={},
                    style={'display': 'inline-block'})
                 ),
        html.H2('Text Statistics : ' ,
                style = {
                    'textAlign' : 'center',
                    'color' : layout_colors['headings']}
                    ),
        html.Div(children =[
            dcc.Graph(id='Word_freq_analysis',
                    figure={},
                    style={'display': 'inline-block'}),

            dcc.Graph(id='Avg_words',
                         figure={},
                         style={'display': 'inline-block'})
        ]),
        html.H2('Word Clouds :',
                style = {
                    'textAlign' : 'center',
                    'color' : layout_colors['headings']}
                ),
        html.Div(children=[
            html.H3('Top 100 Most Repeated Words' ,style={'display': 'inline-block' ,'color' : layout_colors['sub_headings']} ),
            html.H3('Positive Word Cloud',style={'display': 'inline-block' , "margin-left": "400px" ,'color' : layout_colors['sub_headings']}),
            html.H3('Negative Word Cloud',style={'display': 'inline-block' , "margin-left": "400px" , 'color' : layout_colors['sub_headings'] })
                        ]
                 ),
        html.Div(children=[
            html.Img(id="Normal_word_cloud",style={'display': 'inline-block'}),
            html.Img(id="Positive_word_cloud",style={'display': 'inline-block' , "margin-left": "150px"}),
            html.Img(id="Negative_word_cloud",style={'display': 'inline-block',"margin-left": "150px"})
        ]
        ),
        html.Br(),
        html.Br(),
        html.H2('Named Entity Recognition : ' ,
                    style = {
                    'textAlign' : 'center',
                    'color' : layout_colors['headings']}
                    ),
        html.Div(
            children=[
                dcc.Graph(id='NER',
                    figure={}
            )
            ]
        ),
        html.Br(),
        html.Br(),
        html.H2('Parts of speech Tagging : ' ,
                    style = {
                    'textAlign' : 'center',
                    'color' : layout_colors['headings']}
                    ),
        html.Div(
            children=[
                dcc.Graph(id='POS',
                    figure={}
            )
            ]
        ),
        html.Br(),
        html.Br(),
        html.H2('Sentiment Analysis' , 
                style = {
                    'textAlign' : 'center',
                    'color' : layout_colors['headings']}
                    ),

        html.Div(children=[
            html.H3('Confusion Martix' ,style={'display': 'inline-block' ,'color' : layout_colors['sub_headings']} ),
            html.H3('Radar of Sentiments',style={'display': 'inline-block' , "margin-left": "800px" ,'color' : layout_colors['sub_headings']}),            ]
                 ),
                 
        html.Div(
            children =[
                dcc.Graph(id='Sentiment_Analysis_HM',
                    figure={},
                    style={'display': 'inline-block'}),

                dcc.Graph(id='Radar_SentimentAnalysis',
                         figure={},
                         style={'display': 'inline-block'})
            ]
        ),
        html.H2('Emotion Mining' , 
                style = {
                    'textAlign' : 'center',
                    'color' : layout_colors['headings']}
                    ),
        html.Div(
            
                dcc.Graph(id='emotion_mining',
                    figure={} )
        ),
     
    ]
)

################### Single Input, single Output
@app.callback(
            [Output(component_id='Extracted_Data', component_property='data'), 
             Output(component_id='Extracted_Data', component_property='columns')],
            [Input('URL_Submit_button', 'n_clicks_timestamp')],
            prevent_initial_call=True
            )

def update_table(entered_url):
    df = get_data(entered_url)
    columns = [{'name': col, 'id': col} for col in df.columns]
    data = df.to_dict(orient='records')
    return data, columns


def get_data(entered_url):
    df = pd.read_csv('Amazon_reviews.csv')
    print(type(df))
    return df[0:5]

############# Single input - multiple output : GRAPHS 

@app.callback(
    [Output(component_id='Rating_Overview', component_property='figure'),
    Output(component_id='Word_freq_analysis', component_property='figure'), 
    Output(component_id='Avg_words', component_property='figure')],
    [Input('URL_Submit_button', 'n_clicks_timestamp')],
    prevent_initial_call=True
)
def get_plots(df):
    fig1 , fig2 , fig3 = pre_processing_textStats(df_reviews_ratings)
    return fig1 , fig2 , fig3 


def pre_processing_textStats(df):
    # preprocessing: 
    df_reviews_ratings['Reviews']=df_reviews_ratings['Reviews'].astype(str)
    df_reviews_ratings['Reviews']=df_reviews_ratings['Reviews'].apply(lambda x: x.lower())
    df_reviews_ratings['Reviews']=df_reviews_ratings['Reviews'].apply(lambda x: re.sub(r'http\S+', '', x))
    df_reviews_ratings['Reviews']=df_reviews_ratings['Reviews'].apply(lambda x: re.sub('@[^\s]+','',x))
    df_reviews_ratings['Reviews'] = df_reviews_ratings['Reviews'].apply(lambda x: re.sub('[^a-z ]','', x))
    tot_rev = [Reviews.strip() for Reviews in df_reviews_ratings['Reviews']]
    #Text Statistics :
    # Finding no of characters in each review : 
    df_reviews_ratings["no_of_characters"]=df_reviews_ratings['Reviews'].str.len()
    # Finding the no of words in each review 
    a=df_reviews_ratings["no_of_characters"]
    df_reviews_ratings["no_of_words"] = df_reviews_ratings['Reviews'].str.split().map(lambda x: len(x))
    # finding avg length of words in each review:
    avg_Word_length=[]
    for i in df_reviews_ratings["Reviews"].str.split():
        word_len=[]
        for j in i:
            word_len.append(len(j))
        avg_Word_length.append(np.mean(word_len))
    df_reviews_ratings["avd_Word_Length"]=avg_Word_length
    
    # Count of reviews :
    df_count=pd.DataFrame()
    df_count['count']=df_reviews_ratings.groupby(by=["Ratings"]).size()

    rating_count = px.bar(data_frame=df_count , y= df_count['count'] ,template='seaborn')
    wfa = px.histogram(data_frame=df_reviews_ratings , x= df_reviews_ratings['no_of_words'] , title ='Number of Words in Reviews',template='seaborn')
    avgWords = px.histogram(data_frame=df_reviews_ratings , x=df_reviews_ratings['avd_Word_Length'] ,title='Average Word Length',template='seaborn')
    return rating_count , wfa , avgWords

############# Basic pre-processing :
# Word Tokenizarion 
df_reviews_ratings['Reviews'] = df_reviews_ratings['Reviews'].apply(lambda x: tokenize.word_tokenize(str(x)))
    
# Lemmatization s
df_reviews_ratings['Reviews'] = df_reviews_ratings['Reviews'].apply(lambda x: " ".join(x) )
nlp = spacy.load("en_core_web_sm")
df_reviews_ratings['Reviews'] = df_reviews_ratings['Reviews'].apply(lambda x: [token.lemma_ for token in nlp(x)] )
tot_reviwes=[]
# Creating a single list of all the lemmatized sentences which were lists
for i in range(0,len(df_reviews_ratings.Reviews)):
    tot_reviwes += df_reviews_ratings.Reviews[i]
    

####################### Word Clouds : 

# NORMAL WORD CLOUD
@app.callback(
            [dd.Output(component_id='Normal_word_cloud', component_property='src')],
             #dd.Output(component_id='Positive_word_cloud', component_property='src'),
             #dd.Output(component_id='Negative_word_cloud', component_property='src')],
            [dd.Input('URL_Submit_button', 'n_clicks_timestamp')],
            prevent_initial_call=True
            )

def get_img(a):
    img = BytesIO()
    get_NormalwordClouds().save(img, format='PNG')
    return ['data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())]

def get_NormalwordClouds():
    
   
    # Top 100 most repeated words:
    mostcommon = FreqDist(tot_reviwes).most_common(100)
    print("****************",len(mostcommon))
    Normal_wordcloud = WordCloud(width=400, height=290, background_color='black').generate(str(mostcommon))
    
    return Normal_wordcloud.to_image()

# POSITIVE WORD CLOUD
@app.callback(
            [dd.Output(component_id='Positive_word_cloud', component_property='src')],
            [dd.Input('URL_Submit_button', 'n_clicks_timestamp')],
            prevent_initial_call=True
            )

def get_img(a):
    img = BytesIO()
    get_positive_Wordcloud().save(img, format='PNG')
    return ['data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())]

def get_positive_Wordcloud():
    # Positive Word CLoud :
    reviews_positiveWords = [word  for word in tot_reviwes if word  in positive_words] 
    pstv_wordcloud = WordCloud(width=400, height=290, background_color='black').generate(str(reviews_positiveWords))
    return pstv_wordcloud.to_image()

# NEGATIVE WORD CLOUD 
@app.callback(
            [dd.Output(component_id='Negative_word_cloud', component_property='src')],
            [dd.Input('URL_Submit_button', 'n_clicks_timestamp')],
            prevent_initial_call=True
            )

def get_img(a):
    img = BytesIO()
    get_negative_Wordcloud().save(img, format='PNG')
    return ['data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())]

def get_negative_Wordcloud():
    # Positive Word CLoud :
    reviews_negativeWords = [word  for word in tot_reviwes if word  in negative_words] 
    neg_wordcloud = WordCloud(width=400, height=290, background_color='black').generate(str(reviews_negativeWords))
    return neg_wordcloud.to_image()

###### NAMED ENTITY RECOGNITION : 
@app.callback(
            dd.Output(component_id='NER', component_property='figure'),
            dd.Input('URL_Submit_button', 'n_clicks_timestamp'),
            prevent_initial_call=True
            )

def get_NER(a):
    fig = get_NER_list(a)
    return fig

def count_texts(text) : 
    counter=Counter(text)
    count=counter.most_common()
    rev_ner,counts=map(list,zip(*count))
    return rev_ner,counts

rev_comp=" ".join(tot_reviwes)
doc=nlp(rev_comp)
[(x.text,x.label_) for x in doc.ents]
tot_text = [x.label_ for x in doc.ents]

def get_NER_list(a):
    rev_ner,counts=count_texts(tot_text)
    ner = px.bar( x=rev_ner ,y=counts,template='seaborn')
    return ner

###### PARTS OF SPEECH TAGGING 
@app.callback(
            dd.Output(component_id='POS', component_property='figure'),
            dd.Input('URL_Submit_button', 'n_clicks_timestamp'),
            prevent_initial_call=True
            )

def get_POS(a):
    fig = get_POS_list(a)
    return fig

def get_POS_list(a):
    tot_text = [token.pos_ for token in doc]
    rev_pos,counts=count_texts(tot_text)
    pos = px.bar( x=rev_pos ,y=counts,template='seaborn')
    return pos
    

###### Sentiment Analysis 

# Creating the corpus of the reviews : 
corpus= []
for i in range(0,len(df_reviews_ratings.Reviews)):
    text = ''.join(df_reviews_ratings.Reviews[i])
    corpus.append(text)

# TF-IDF word embedding :
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()
y = df_reviews_ratings.iloc[:,1].values

# The classes are imbalanced ... so balancing them using SMOTE 
sm =SMOTE(random_state = 100)          
x_sm,y_sm = sm.fit_resample(X, y)

@app.callback(
            dd.Output(component_id='Sentiment_Analysis_HM', component_property='figure'),
            dd.Input('URL_Submit_button', 'n_clicks_timestamp'),
            prevent_initial_call=True
            )

def get_sentiment_analysis(a):
    fig = get_sentiAnalysis(a)
    return fig

def get_sentiAnalysis(a):
    model=RandomForestClassifier(n_estimators=100,random_state=8,max_features = 2)
    model.fit(x_sm,y_sm)
    y_pred_sm=model.predict(x_sm)
    #df_reviews_ratings['Pred_ratings'] = y_pred_sm
    clf_report = classification_report(y_sm,
                                   y_pred_sm ,output_dict=True)
    clf_df = pd.DataFrame.from_dict(clf_report)
    clf_HM = px.imshow(clf_df,text_auto=True, aspect="auto")
    return clf_HM

# VADER Sentiment Analysis : 
analyzer = SentimentIntensityAnalyzer() 
pos=neg=neu=compound=0
print("*************************************************************")
print("Pos at start " , pos,neg,neu)
sentiment_pred = []
#cmp = []
for sentence in df_reviews_ratings.Reviews:
    vs = analyzer.polarity_scores(sentence)
    pos += (vs["pos"])
    neg += (vs["neg"])
    neu += (vs["neu"])
    compound = (vs["compound"])
    #cmp.append(compound)
    # decide sentiment as positive, negative and neutral
    if vs["compound"] >= 0.05 :
        sentiment_pred.append(1)
 
    elif vs["compound"] <= - 0.05 :
        sentiment_pred.append(-1)
 
    else :
        sentiment_pred.append(0)

df_reviews_ratings['Vader_Classificaton'] = sentiment_pred

@app.callback(
            dd.Output(component_id='Radar_SentimentAnalysis', component_property='figure'),
            dd.Input('URL_Submit_button', 'n_clicks_timestamp'),
            prevent_initial_call=True
            )

def get_sentiment_analysis(a):
    df = pd.DataFrame(dict(
                    r=[pos/len(df_reviews_ratings.Reviews),neg/len(df_reviews_ratings.Reviews),neu/len(df_reviews_ratings.Reviews)],
                    theta=['Positive','negative','Neutral']))
    print(pos,neg,neg)
    fig = px.line_polar(df , r='r' , theta= 'theta' ,line_close = True)
    fig.update_traces(fill='toself')
    return fig

 ########## Emotion Mining :

@app.callback(
            dd.Output(component_id='emotion_mining', component_property='figure'),
            dd.Input('URL_Submit_button', 'n_clicks_timestamp'),
            prevent_initial_call=True
            )


def get_emotionMining(a):
    emotion = NRCLex(str(corpus))
    emotion_scores = pd.DataFrame(emotion.raw_emotion_scores , index=[0]).transpose()
    fig = px.bar( data_frame = emotion_scores ,template='seaborn')
    return fig

if __name__ == '__main__':
    app.run_server(port= 4050,debug=True)