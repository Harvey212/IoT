import psycopg2

import requests
import serial
import msvcrt
from threading import Thread
import time
import numpy as np
from paho.mqtt import client as mqtt_client
###################################################
broker = 'broker.emqx.io'
port = 1883
topic = "python/mqtt"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{0}'
###################################################
CONNECTION = "dbname =tsdb user=tsdbadmin password=123456789101112 host=jcfv5h341q.nouesk1fev.tsdb.cloud.timescale.com port=39656 sslmode=require"
conn = psycopg2.connect(CONNECTION)
cursor = conn.cursor()

####################################
dr = "DROP TABLE data;"
cursor.execute(dr)
conn.commit()

#######################################
table = """CREATE TABLE data(id SERIAL PRIMARY KEY,value DECIMAL);"""
cursor.execute(table)
conn.commit()
#####################################################
COM_PORT = "COM3"
BAUD_RATES =9600
ser=serial.Serial(COM_PORT, BAUD_RATES)
##########################################
#240 numbers/sec
summ=0
idd=0
print('hi')
################################################
def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    #client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client
#####################################
client = connect_mqtt()
client.loop_start()

#########################################################
def read():
	global summ,ser,idd
	
	if ser.in_waiting:
		data_raw = ser.readline()
		try:	
			mydd = float(data_raw.decode())
			msg = "{}".format(mydd-35)#/np.iinfo(np.int16).max
			result = client.publish(topic, msg)
			print(result)

			if mydd>40:
				summ+=(mydd-35)
				idd+=1

			#print('read')
		except:
			pass  

def store():
	##database will slowdown the processing
	global conn,cursor,summ,idd

	if idd>0:
		temp=summ/idd
	else:
		temp=0

	#print(temp)
	summ=0
	idd=0
	query="INSERT INTO data(value) VALUES ({});".format(temp)	
	cursor.execute(query)
	conn.commit()
	#print('store')
	

def read_thread():
	global ser
	while True:
		# 在wihle迴圈中使用多線程確保每0.01秒會讀取一次感測值
		if msvcrt.kbhit() and msvcrt.getch()[0] == 27:
			break
		Thread(target=read).start()
		time.sleep(1/240)

	ser.close()

def store_thread():
	global cursor
	while True:
		if msvcrt.kbhit() and msvcrt.getch()[0] == 27:
			break
		time.sleep(1)
		#在wihle迴圈中使用多線程確保每秒會觸發回傳
		Thread(target=store).start()

	cursor.close()

thread_arr = [Thread(target=read_thread), Thread(target=store_thread)]
for thread in thread_arr:
	thread.start()