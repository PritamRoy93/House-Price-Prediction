from flask import Flask,render_template,request
import numpy as np
import joblib

app = Flask(__name__)

model=joblib.load("house_price_prediction.pkl")

@app.route("/")
def home():   
   
    return render_template("index.html")

@app.route("/resu",methods=['POST'])
def resu():
    frm1=float(request.form['f1'])
    frm2=float(request.form['f2'])
    frm3=float(request.form['f3'])
    frm4=float(request.form['f4'])
    frm5=float(request.form['f5'])

    """"
    outp=frm30 ,fr1=frm1,fr2=frm2,fr3=frm3,fr4=frm4,fr5=frm5,fr6=frm6,fr7=frm7,fr8=frm8,fr9=frm9,fr10=frm10,fr11=frm11,fr12=frm12,fr13=frm13,fr14=frm14,fr15=frm15,fr16=frm16,fr17=frm17,fr18=frm18,fr19=frm19,fr20=frm20,fr21=frm21,fr22=frm22,fr23=frm23,fr24=frm24,fr25=frm25,fr26=frm26,fr27=frm27,fr28=frm28,fr29=frm29"""
    features=np.array([[frm1,frm2,frm3,frm4,frm5]])
    output=model.predict(features)
    if output[0]==1:
         return render_template("resu.html")
    else:
        return render_template("resu.html",outp=output)


app.run(debug=True)
# -*- coding: utf-8 -*-

