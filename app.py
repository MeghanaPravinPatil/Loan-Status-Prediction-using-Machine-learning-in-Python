# app.py
from flask import Flask, render_template, request
import joblib, pandas as pd, sqlite3, os, matplotlib.pyplot as plt

app = Flask(__name__)
MODEL_PATH = 'model/loan_model.pkl'
DB_PATH = 'predictions.db'
CHART_PATH = 'static/chart.png'

model = joblib.load(MODEL_PATH)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Gender TEXT, Married TEXT, Dependents TEXT, Education TEXT,
        Self_Employed TEXT, ApplicantIncome REAL, CoapplicantIncome REAL,
        LoanAmount REAL, Loan_Amount_Term REAL, Credit_History REAL,
        Property_Area TEXT, prediction INTEGER)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        data = {
            'Gender': request.form['Gender'],
            'Married': request.form['Married'],
            'Dependents': request.form['Dependents'],
            'Education': request.form['Education'],
            'Self_Employed': request.form['Self_Employed'],
            'ApplicantIncome': float(request.form['ApplicantIncome']),
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'LoanAmount': float(request.form['LoanAmount']),
            'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
            'Credit_History': float(request.form['Credit_History']),
            'Property_Area': request.form['Property_Area']
        }
        df = pd.DataFrame([data])
        pred = int(model.predict(df)[0])
        label = 'Approved' if pred==1 else 'Not Approved'

        conn = sqlite3.connect(DB_PATH)
        conn.execute('INSERT INTO predictions (Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,prediction) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                     (data['Gender'],data['Married'],data['Dependents'],data['Education'],data['Self_Employed'],data['ApplicantIncome'],data['CoapplicantIncome'],data['LoanAmount'],data['Loan_Amount_Term'],data['Credit_History'],data['Property_Area'],pred))
        conn.commit()
        conn.close()
        return render_template('index.html', prediction=label)
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM predictions', conn)
    conn.close()
    if df.empty:
        return 'No predictions yet.'
    counts = df['prediction'].value_counts()
    plt.bar(['Not Approved','Approved'], [counts.get(0,0), counts.get(1,0)])
    plt.title('Predictions Summary')
    plt.savefig(CHART_PATH)
    plt.close()
    return render_template('dashboard.html', chart_url=CHART_PATH, total=len(df))

if __name__ == '__main__':
    app.run(debug=True)
