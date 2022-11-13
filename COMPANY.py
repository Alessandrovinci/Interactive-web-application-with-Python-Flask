
from flask import Flask, g, render_template,request,redirect
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
import json
from sqlalchemy import create_engine
import sqlite3
from sqlalchemy import or_,and_
import pandas as pd
from twitter__ import twitter_app
import re

app = Flask(__name__)
app.register_blueprint(twitter_app)


@app.route('/')
def index():
    title='COMPANY University Program'
    return render_template('index.html', title=title)

@app.route('/about.html')
def about():
    title='About the data'
    return render_template('about.html', title=title)


#db_name='test_import_csv.db'
db_name='WUR1.db'
#db_name2='Research.db'
db_name2='prova.db'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_name
app.config['SQLALCHEMY_BINDS'] = {'prova':'sqlite:///' + db_name2}


app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db=SQLAlchemy(app)

class THE(db.Model):
    #__tablename__='Word_University_Rank_2020'
    __tablename__='WUR'
    Rank_Char=db.Column(db.Integer)
    Score_Rank=db.Column(db.Integer)
    University=db.Column(db.String,primary_key=True)
    Country=db.Column(db.String)
    Number_students=db.Column(db.Integer)
    Numb_students_per_Staff=db.Column(db.Float)
    International_Students=db.Column(db.Float)
    Percentage_Female=db.Column(db.Float)
    Percentage_Male=db.Column(db.Float)
    Research=db.Column(db.Float)
    Teaching=db.Column(db.Float)
    Citations=db.Column(db.Float)
    Industry_Income=db.Column(db.Float)
    International_Outlook=db.Column(db.Float)
    Score_Result=db.Column(db.Float)
    new_inter=db.Column(db.Integer)
    #Overall_Ranking=db.Column(db.Float)


@app.route('/filter.html',methods=["POST", "GET"])
def filter():
    title='Filter'
    with open("ALL_uni.txt", "r") as fp:
            u = json.load(fp)

    with open("ALL_countries.txt", "r") as fp:
            c = json.load(fp)
    
    if request.method == "GET":
        
        return render_template('filter.html', title=title,UNIS=u,COUNTRIES=c)


@app.route('/form.html', methods=["POST"])
def form():
    title="Results"
    uni_name=request.form.get("uni_name")
    country=request.form.get('country')

    #FILTERS
    student_number=request.form.get('SN')
    student_per_staff=request.form.get('SR')
    new_inter=request.form.get('INT')
    rank=request.form.get('Rank')
    research=request.form.get('Research')
    teaching=request.form.get('Teaching')
    #QUERIES

    if uni_name=='' and country!='':
        try:
            QR = THE.query.filter_by(Country=country).order_by(desc(student_number),student_per_staff,desc(new_inter),rank,desc(research),desc(teaching)).all()
            emptiness=len(QR)
            if emptiness==0:
                return render_template("alert_input.html")
            else:
                return render_template("form.html",title=title,QR=QR)
        except:
            return 'Something went wrong'
    elif uni_name!='':
        try:
            QR = THE.query.filter_by(University=uni_name).all()
            emptiness=len(QR)
            if emptiness==0:
                return render_template("alert_input.html")
            else:
                return render_template("form.html",title=title,QR=QR)
        except:
            return 'Something went wrong'
    else:
        try:
            QR = THE.query.order_by(desc(student_number),student_per_staff,desc(new_inter),rank,desc(research),desc(teaching)).all()
            return render_template("form.html",title=title,QR=QR)
        except:
            return 'Something went wrong'




@app.route('/stats/<uni>/<country>')
def stats(uni,country):
    title='STATS'
    QR = THE.query.filter_by(University=uni).all()
    return render_template('stats.html', title=title, uni=uni, country=country, QR=QR)



# RUSSEL GROUP
class RES(db.Model):
    __bind_key__='prova'
    __tablename__='prova'
    University=db.Column(db.String,primary_key=True)
    Teaching=db.Column(db.String)
    Research=db.Column(db.String)
    #Specific=db.Column(db.String)
    sp1=db.Column(db.String)
    sp2=db.Column(db.String)
    sp3=db.Column(db.String)
    sp4=db.Column(db.String)
    sp5=db.Column(db.String)
    sp6=db.Column(db.String)
    sp7=db.Column(db.String)
    webs=db.Column(db.String)
    img=db.Column(db.String)



@app.route('/russel_group.html',methods=["POST", "GET"])
def russel_group():
    title='Russel Group'
    return render_template('russel_group.html', title=title)

def regex_filt(sentence):
    b=[]
    aaa=sentence.split(" ")
    for i,j in enumerate(aaa):
        if j in ['Arts','Engineering']:
            b.append(j+' '+aaa[i+1]+' '+aaa[i+2])
        elif j in ['Life','Social','Computer','Physical','Environmental']:
            b.append(j+' '+aaa[i+1])
        elif j in ['Law','Medicine','Business','Theology']:
            b.append(j)
    return b


@app.route('/russel_form.html', methods=["POST"])
def russel_form():
    title="Results!"
    #uni_name=request.form.get("uni_name")
    #country=request.form.get('country')

    #FILTERS
    #student_number=request.form.get('SN')
    law=request.form.get('Law')
    bus=request.form.get('Bus')
    eng=request.form.get('Engineering & Technology')
    medicine=request.form.get('Medicine')
    arts=request.form.get('Arts & Humanities')
    life=request.form.get('Life Science')
    physical=request.form.get('Physical Science')
    social=request.form.get('Social Science')
    computer=request.form.get('Computer Science')
    theology=request.form.get('Theology')


    #QUERIES
    select_uni_name=request.form.get("select_uni_name")
    details=request.form.get('Details')
    teach_filters=[law,bus,eng,medicine,arts,life,physical,social,computer,theology]
    teach_filters=[f for f in teach_filters if f!=None]

    alert_msg=0



    if select_uni_name!='ALL':
        QR = RES.query.filter_by(University=select_uni_name).all()
        flag="1"
        this_q_t=QR[0].Teaching
        the_teachings=regex_filt(this_q_t)
        this_q_r=str(QR[0].Research)
        the_research=regex_filt(this_q_r)
        return render_template('russel_form.html', teach_filters=teach_filters, title=title,alert_msg=alert_msg,QR=QR,
                                flag=flag,the_teachings=the_teachings,this=this_q_t,the_research=the_research,uni_name=select_uni_name)
    
    elif len(teach_filters)==1:
        flag="0"
        search1 = "%{}%".format(teach_filters[0])
        QR = RES.query.filter(RES.Teaching.like(search1)).all()
        return render_template('russel_form.html',teach_filters=teach_filters, title=title,alert_msg=alert_msg,QR=QR,flag=flag)
    elif len(teach_filters)==2:
        flag="0"
        search1 = "%{}%".format(teach_filters[0])
        search2 = "%{}%".format(teach_filters[1])
        QR = RES.query.filter(and_(RES.Teaching.like(search1),RES.Teaching.like(search2))).all()
        return render_template('russel_form.html', teach_filters=teach_filters,title=title,alert_msg=alert_msg,QR=QR,flag=flag)

    elif len(teach_filters)==3:
        flag="0"
        search1 = "%{}%".format(teach_filters[0])
        search2 = "%{}%".format(teach_filters[1])
        search3 = "%{}%".format(teach_filters[2])
        QR = RES.query.filter(and_(RES.Teaching.like(search1),RES.Teaching.like(search2),RES.Teaching.like(search3))).all()
        return render_template('russel_form.html', teach_filters=teach_filters,title=title,alert_msg=alert_msg,QR=QR,flag=flag)
    elif len(teach_filters)==4:
        flag="0"
        search1 = "%{}%".format(teach_filters[0])
        search2 = "%{}%".format(teach_filters[1])
        search3 = "%{}%".format(teach_filters[2])
        search4 = "%{}%".format(teach_filters[3])
        QR = RES.query.filter(and_(RES.Teaching.like(search1),RES.Teaching.like(search2),RES.Teaching.like(search3),RES.Teaching.like(search4))).all()
        return render_template('russel_form.html', teach_filters=teach_filters,title=title,alert_msg=alert_msg,QR=QR,flag=flag)

    elif len(teach_filters)>4:
        if len(teach_filters)>5:
            teach_filters=teach_filters[:5]
            alert_msg='1'
        flag="0"
        search1 = "%{}%".format(teach_filters[0])
        search2 = "%{}%".format(teach_filters[1])
        search3 = "%{}%".format(teach_filters[2])
        search4 = "%{}%".format(teach_filters[3])
        search5 = "%{}%".format(teach_filters[4])

        QR = RES.query.filter(and_(RES.Teaching.like(search1),RES.Teaching.like(search2),RES.Teaching.like(search3),
            RES.Teaching.like(search4),RES.Teaching.like(search5))).all()
        return render_template('russel_form.html', teach_filters=teach_filters,title=title,alert_msg=alert_msg,QR=QR,flag=flag)
    else:
        QR = RES.query.all()
        flag="0"
        return render_template('russel_form.html', teach_filters=teach_filters,title=title,QR=QR,alert_msg=alert_msg,flag=flag)

#RESEARCH

@app.route('/russel_form_res.html', methods=["POST"])
def russel_form_res():
    title="Results!"
    env=request.form.get('Environmental studies')
    bus=request.form.get('Bus2')
    eng=request.form.get('Engineering & Technology2')
    medicine=request.form.get('Medicine2')
    arts=request.form.get('Arts & Humanities2')
    life=request.form.get('Life Science2')
    physical=request.form.get('Physical Science2')
    social=request.form.get('Social Science2')
    computer=request.form.get('Computer Science2')
    law=request.form.get('Law2')
   

    #QUERIES
    teach_filters=[env,bus,eng,medicine,arts,life,physical,social,computer,law]
    teach_filters=[f for f in teach_filters if f!=None]

    alert_msg=0

    if len(teach_filters)==1:
        search1 = "%{}%".format(teach_filters[0])
        QR = RES.query.filter(RES.Research.like(search1)).all()
        return render_template('russel_form_res.html',teach_filters=teach_filters, title=title,alert_msg=alert_msg,QR=QR)
    elif len(teach_filters)==2:
        search1 = "%{}%".format(teach_filters[0])
        search2 = "%{}%".format(teach_filters[1])
        QR = RES.query.filter(and_(RES.Research.like(search1),RES.Research.like(search2))).all()
        return render_template('russel_form_res.html', teach_filters=teach_filters,title=title,alert_msg=alert_msg,QR=QR)

    elif len(teach_filters)==3:
        search1 = "%{}%".format(teach_filters[0])
        search2 = "%{}%".format(teach_filters[1])
        search3 = "%{}%".format(teach_filters[2])
        QR = RES.query.filter(and_(RES.Research.like(search1),RES.Research.like(search2),RES.Research.like(search3))).all()
        return render_template('russel_form_res.html', teach_filters=teach_filters,title=title,alert_msg=alert_msg,QR=QR)
    elif len(teach_filters)==4:
        search1 = "%{}%".format(teach_filters[0])
        search2 = "%{}%".format(teach_filters[1])
        search3 = "%{}%".format(teach_filters[2])
        search4 = "%{}%".format(teach_filters[3])
        QR = RES.query.filter(and_(RES.Research.like(search1),RES.Research.like(search2),RES.Research.like(search3),RES.Research.like(search4))).all()
        return render_template('russel_form_res.html', teach_filters=teach_filters,title=title,alert_msg=alert_msg,QR=QR)

    elif len(teach_filters)>4:
        if len(teach_filters)>5:
            teach_filters=teach_filters[:5]
            alert_msg='1'
        search1 = "%{}%".format(teach_filters[0])
        search2 = "%{}%".format(teach_filters[1])
        search3 = "%{}%".format(teach_filters[2])
        search4 = "%{}%".format(teach_filters[3])
        search5 = "%{}%".format(teach_filters[4])

        QR = RES.query.filter(and_(RES.Research.like(search1),RES.Research.like(search2),RES.Research.like(search3),
            RES.Research.like(search4),RES.Research.like(search5))).all()
        return render_template('russel_form_res.html', teach_filters=teach_filters,title=title,alert_msg=alert_msg,QR=QR)
    else:
        QR = RES.query.all()
        return render_template('russel_form_res.html', teach_filters=teach_filters,title=title,QR=QR,alert_msg=alert_msg)


@app.route('/specific/<uni>')
def specific(uni):
    title='Research'
    flag="1"
    QR = RES.query.filter_by(University=uni).all()
    #Check if none research
    QR2=RES.query.filter_by(University=uni).with_entities(RES.sp1).all()
    if QR2[0][0]==None:
        return render_template("alert_research.html",QR=QR)
    else:
        this_q_t=QR[0].Teaching
        the_teachings=regex_filt(this_q_t)
        this_q_r=str(QR[0].Research)
        the_research=regex_filt(this_q_r)
        return render_template('specific.html', title=title, uni_name=uni,QR=QR,QR2=QR2,
                                flag=flag,the_teachings=the_teachings,this=this_q_t,the_research=the_research)


