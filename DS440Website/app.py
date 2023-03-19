import pickle
import pandas as pd
from flask import Flask,render_template, request
from flask_mysqldb import MySQL
 
app = Flask(__name__)
 
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flask'
 
mysql = MySQL(app)
 
@app.route('/form')
def form():
    return render_template('DS440_Website_V2.html')
 
@app.route('/login', methods = ['POST', 'GET'])
def login():
    if request.method == 'GET':
        return "Login via the login Form"
     
    if request.method == 'POST':
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        skinthickness = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        age = request.form['age']
        cursor = mysql.connection.cursor()
        cursor.execute(''' INSERT INTO info_table VALUES(%s,%s,%s,%s,%s,%s,%s)''',(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, age))
        mysql.connection.commit()

        #Grab data to predict on model
        cursor.execute('SELECT * FROM info_table')
        result = cursor.fetchall()
        
        df = pd.DataFrame(result, columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age'])
       
        #delete case from SQL Server
        cursor.execute('DELETE FROM info_table WHERE "preganancies" BETWEEN 0 AND 100')
        mysql.connection.commit()
    
        #import in previously trained model
        filename = 'FinalSVCModel.sav'
        PredictionModel = pickle.load(open(filename, 'rb'))

        #Determine liklihood patient has diabetes
        output = PredictionModel.predict_proba(df)
        probdiabetes = round(output[0][-1]*100, 2)
        cursor.close()

        #Return Liklihood of diabetes
        return f"The patient a {probdiabetes}% liklihood of Diabetes"
 
app.run(host='localhost', port=5000)



if __name__ == '__main__':
    app.run(debug=True)