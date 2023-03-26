import pickle
import pandas as pd
from flask import Flask,render_template, request
from flask_mysqldb import MySQL
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)


with app.app_context(): 
    db = SQLAlchemy()

app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///DataInputs.db' 

db.init_app(app)

class DataInput(db.Model):
   id = db.Column('case_id', db.Integer, primary_key = True)
   pregnancies = db.Column(db.Float)
   glucose = db.Column(db.Float)  
   bloodpressure = db.Column(db.Float)
   skinthickness = db.Column(db.Float)
   insulin = db.Column(db.Float)
   bmi = db.Column(db.Float)
   age = db.Column(db.Float)

   def __repr__(self):
        return '<DataInput %r>' % self.pregnancies

with app.app_context():
    db.create_all()

@app.route('/form')
def form():
    return render_template('DS440_Website_V2.html')
 
@app.route('/results', methods = ['POST', 'GET'])
def results():
    if request.method == 'GET':
        return "Go to /form"
     
    #Function that replaces empty string in html form input with a zero
    def zeros(forms):
        if forms == '':
            forms = 0
            return forms 
        else:
            return forms
    
    if request.method == 'POST':
        addition = DataInput(pregnancies = zeros(request.form['pregnancies']), glucose = zeros(request.form['glucose']),  bloodpressure = zeros(request.form['bloodpressure']), skinthickness = zeros(request.form['skinthickness']), insulin = zeros(request.form['insulin']), bmi = zeros(request.form['bmi']), age = zeros(request.form['age']))     
        db.session.add(addition)
        db.session.commit()
    
        #Grab data from database to predict on model
        data_query = db.session.query(DataInput).with_entities(DataInput.id, DataInput.pregnancies, DataInput.glucose, DataInput.bloodpressure, DataInput.skinthickness, DataInput.insulin, DataInput.bmi, DataInput.age)
        inputted_data = data_query.all()
        output_df = pd.DataFrame.from_records(inputted_data, index='id', columns=['id', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age'])
        


       
        #delete case from SQL Database
        db.session.delete(addition)
        db.session.commit()
    
        #import in previously trained model
        filename = 'FinalSVCModel.sav'
        PredictionModel = pickle.load(open(filename, 'rb'))

        #Determine liklihood patient has diabetes
        output = PredictionModel.predict_proba(output_df)
        probdiabetes = round(output[0][-1]*100, 2)
        

        #Return Liklihood of diabetes
        return f"The patient a {probdiabetes}% liklihood of Diabetes."
 
app.run(host='localhost', port=5000)



if __name__ == '__main__':
    app.run(debug=True)