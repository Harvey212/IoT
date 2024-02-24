from flask import Flask,render_template,url_for,request,redirect, make_response, render_template
import json
from time import time
from random import random
import psycopg2

#########################
CONNECTION = "dbname =tsdb user=tsdbadmin password=123456789101112 host=jcfv5h341q.nouesk1fev.tsdb.cloud.timescale.com port=39656 sslmode=require"
conn = psycopg2.connect(CONNECTION)
cursor = conn.cursor()
app=Flask(__name__)
##########################################################
@app.route('/', methods=["GET", "POST"])
def main():
    return render_template('index.html')


@app.route('/data', methods=["GET", "POST"])
def data():

	#start=index*48000+1
	#end=(index+1)*48000

	#########################################################
	q="SELECT COUNT(*) FROM data"
	cursor.execute(q)
	row_num=[float(row[0]) for row in cursor.fetchall()]

	################################################################
	try:
		#row start from 1
		start=int(row_num[0])
		#print(start)
		end=start+1
		query="SELECT value FROM data WHERE (id>={} AND id<{});".format(start,end)
		cursor.execute(query)
		#####################################################################
		val=[float(row[0]) for row in cursor.fetchall()]
		#print(val)
		###################################
		data = [time()*1000,val[0]]
	except:
		data=[time()*1000,0]

	response = make_response(json.dumps(data))
	response.content_type = 'application/json'
    
	return response
	

#################################################
if __name__ == "__main__":
	app.run(debug=True)


