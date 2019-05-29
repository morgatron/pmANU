""" Script to constantly read from the picoscope, process 
the traces as magnetometry, and serve the magnetometry data on port number @port


The script here needs to know about the pulse-sequence used, in particular the timing and ordering.

TODO:
Feburary 2016: Need to update the processing part, to use the newer methods in run_exp3/4, and to add inputs for this program to change the parameters used for processing.

Should also add the ability to send the raw data, for display purposes. It's unlikely to be the major limiting factor in speed.
"""
import numpy as np
import zmq
from pr import pSerf
#from picoscope import ps5000a
import numpy as np
import time
import pmParams

glbP=pmParams.GlbParams()

socket=None
#ZMQ Stuff
def initSocket():
    global socket
    port = "5557"
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:%s" % port)


#Picoscope stuff
Ncaps=100
totSeqTime=16.66e-3;
tLast=0;
dSdB=None
ref=None
t0=time.time()
def loadCalData():
    global dSdB, ref
    D=np.load(r"c:/PulseSerfData/Dec4CalData_1.npz")
    dSdB=D['dSdB']
    ref=D['ref']

def init():
    acq.init(bLocal=True)
    acq.startAcquisition()


if 0:
    class ScopeCommunicate(object):
        def __init__(self):
            port = "5558"
            context = zmq.Context()
            self.socket = context.socket(zmq.PAIR)
            self.socket.bind("tcp://*:%s" % port)

        def checkForCommands():
            if self.socket.poll(1):
                st = socket.recv().split(b' ',1)
                D=pickle.loads(st) #Should be dictionary for 
                acq.setupAcquire(**D)
                #self.ps.quickSetup(**D)
                
                print("Updated scope parameters: {}".format(D))

def loopTest():
    time.sleep(3)
    #t=arange(300)*dt

    t=time.time()-t0+np.arange(Ncaps)*totSeqTime
    measVs=np.random.normal(size=(3, Ncaps))
    print(measVs[:,:5])
    print("sending mag:...")
    socket.send(b"mag "+np.vstack([t,measVs]).tobytes())

def loopRaw():
    t,rawData=acq.acquireRaw(bReacquire=True)
    Ncaps=rawData.shape[0]
    t=time.time()+np.arange(Ncaps)*seqTime
    print("sending mag:...")
    socket.send(b"mag "+np.vstack([t,rawData]).tobytes())
    #socket.send(b"raw " + data.tobytes())
    #socket.send(b"t:"+t.tobytes())
    #socket.send(b"Vx:"+Vx.tobytes())
    #socket.send(b"Vy:"+Vy.tobytes())
    #socket.send(b"Vz:"+Vz.tobytes())
    #socket.send("t:%s"%t.tobytes())
    #socket.send("Vx:%s"%Vx.tobytes())
    #socket.send("Vy:%s"%Vy.tobytes())
    #socket.send("Vz:%s"%Vz.tobytes())

bTest=True
if __name__=="__main__":
    initSocket()
    if bTest:
        while 1:
            loopTest()
    else:
        initScope()
        loadCalData()
        try:
            while 1:
                getAndProcessData()
        finally:
            ps.close();
    socket.close()
