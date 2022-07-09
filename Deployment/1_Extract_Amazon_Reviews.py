import pandas as pd
import plotly.express as px
from selenium import webdriver
from bs4 import BeautifulSoup
import re


from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.action_chains import ActionChains
import time

import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output

layout_colors = {
    'headings' : '#B8255F',
    'sub_headings' : '#AF38EB',
    'Main_heading' : '#008080'
}
df_reviews_ratings=pd.DataFrame()
app = dash.Dash(__name__)

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
        dash_table.DataTable(id='Extracted_Data',
                        style_data={
                                    'whiteSpace': 'normal'
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

        )
        #dcc.Graph(id='graph_output', figure={}),

       
    ]
)

# Single Input, single Output, State, prevent initial trigger of callback, PreventUpdate
@app.callback(
            [Output(component_id='Extracted_Data', component_property='data'), 
             Output(component_id='Extracted_Data', component_property='columns')],
            [Input('user_entered_url', 'value')],
            prevent_initial_call=True
            )

def update_table(entered_url):
    print("************ Entered url is :" , str(entered_url))
    df = get_data(str(entered_url))
    columns = [{'name': col, 'id': col} for col in df.columns]
    data = df.to_dict(orient='records')
    return data,columns


def get_data(entered_url):
    #initializing webdriver
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(entered_url)
   
    starRatings = []
    tot_rev=[]

    # for loop for reviews extraction : 
    for i in range(1,100):
        print("************** in the for loop")
        html_soup = BeautifulSoup(driver.page_source, 'html.parser')
        if driver.find_element_by_xpath('//li[@class="a-last"]') is not None :
            reviews = html_soup.findAll("span",attrs = {"class","a-size-base review-text review-text-content"})
            stars_div = html_soup.find("div",attrs = {"class","a-section a-spacing-none review-views celwidget"})
            
            stars = stars_div.findAll("span",attrs = {"class","a-icon-alt"})
            for iteam in range(len(reviews)):
                    tot_rev.append(reviews[iteam].text)
                    starRatings.append(stars[iteam].text)
                    
        time.sleep(2)
        element=driver.find_element_by_xpath('//li[@class="a-last"]')
        action = ActionChains(driver)
        action.move_to_element(element).click().perform()
        time.sleep(2)
    
    driver.close()
    print("************* closed the instance")
    tot_starRatings=[]
    for i in starRatings:
        tot_starRatings.append(i[0])

    # Creating data frame of the extracted data :
    df_reviews_ratings["Reviews"]=pd.DataFrame(tot_rev)
    df_reviews_ratings["Ratings"]=pd.DataFrame(tot_starRatings)
    return df_reviews_ratings[0:5]



if __name__ == '__main__':
    app.run_server(debug=True)