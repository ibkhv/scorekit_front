# (A) INIT
# (A1) LOAD MODULES
from flask import Flask, render_template
import pandas as pd

# (A2) FLASK SETTINGS + INIT
app = Flask(__name__)
# app.debug = True
 
# (B) DEMO - READ CSV & GENERATE HTML TABLE

@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/data',  methods=("POST", "GET"))
def showData():
    data = pd.read_csv('demo-data.csv')  
    rows = data.shape[0]
    cols = data.shape[1]-1
    columns = data.columns
    df = data.describe().round(2).T
    df['Missing'] = '0.0'
    df['Type'] = 'Integer'
    df['Quality'] = 'Medium'
    df['Role'] = 'Include'
    df['count'] = round(df['count'],0)
    df = df.reset_index()
    df.rename(columns={'index':'Flied Name','count':'Count','mean':'Mean','min':'Min','max':'Max'}, inplace=True)
    for i in range(len(df)):
        for c in ["Mean", "Min", "Max"]:
            if abs(float(df.loc[i, c])) > 1000000000000:
                df.loc[i, c] = str( round(df.loc[i, c] / 1000000000000 ,1) ) + "T"
            elif abs(float(df.loc[i, c])) > 1000000000:
                df.loc[i, c] = str( round(df.loc[i, c] / 1000000000 ,1) ) + "B"
            elif abs(float(df.loc[i, c])) > 1000000:
                df.loc[i, c] = str( round(df.loc[i, c] / 1000000 ,1) ) + "M"
            elif abs(float(df.loc[i, c])) > 1000:
                df.loc[i, c] = str( round(df.loc[i, c] / 1000 ,1) ) + "K"
        
        df.loc[i, 'Missing'] = round(data[df.loc[i, 'Flied Name']].isna().sum() / data[df.loc[i, 'Flied Name']].count(),2)

        if abs(float(df.loc[i, 'Missing'])) > 0.2:
            df.loc[i, 'Quality'] = 'Low'

        if df.loc[i, 'Flied Name'] == 'record_id':
            df.loc[i, 'Role'] = 'Exclude'

        if df.loc[i, 'Flied Name'] == 'default_12m':
            df.loc[i, 'Role'] = 'Target'

    df = df.drop(columns=['std','25%','50%','75%'])

    df_html = df.to_html()

    return render_template("data.html", rows=rows, cols=cols, columns=columns, df_html=df_html)

@app.route('/model',  methods=("POST", "GET"))
def showModel():

    return render_template("model.html")


# (C) START
if __name__ == "__main__":
    from waitress import serve
    serve(app)
  #app.run()