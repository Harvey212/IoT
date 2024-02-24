import paho.mqtt.client as mqtt
import msvcrt
from threading import Thread
from myModel import project

import time
import numpy as np
###################################################
#####################################################
broker = 'broker.emqx.io'
port = 1883
topic = "python/mqtt"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{0}'
# username = 'emqx'
# password = 'public'
temp=[]
pro=project()
last_time=time.time()
idd=0
#########################
tempid=0
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(topic)

def on_message(client, userdata, msg):
    global temp,idd,last_time,tempid
    
    try:
        idd+=1
        if idd==1:
            last_time=time.time()

        
        m=float(msg.payload.decode())
        #print(idd)
        if (m>0) and (m<1000):     
            temp.append(m/np.iinfo(np.int16).max)#/np.iinfo(np.int16).max     
        tempid+=1
    except:
        pass

client = mqtt.Client()
client.on_connect = on_connect

client.on_message = on_message
client.connect(broker,port)
client.loop_start()
#client.loop_forever()

def pre():
    global temp,pro,last_time,tempid
    temp2=temp
    if tempid>500:
        tempid=0
        temp=[]
        now_time=time.time()
        timetag=now_time-last_time
        ###################################
        pro.show(temp2,timetag)

def pre_thread():
    while True:
        if msvcrt.kbhit() and msvcrt.getch()[0] == 27:
            break
        time.sleep(1)
        #在wihle迴圈中使用多線程確保每秒會觸發回傳
        Thread(target=pre).start()

thread_arr = [Thread(target=pre_thread)]
for thread in thread_arr:
    thread.start()



