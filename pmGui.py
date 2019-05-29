
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import zmq
import pmParams
from munch import Munch, munchify
import pmAcquire as acq
from scipy import signal

glbP=pmParams.GlbParams()

bTest=False
class TestC(object):
    def setupExperiment(self):
        print("Setup with {}".format(glbP.p))
    def setupWithChanged(self, p):
        print("Setup with changed, {}".format(glbP.p))
if bTest:
    c=TestC()
    
else:
    import pmController as c
    c.init()

app = QtGui.QApplication([])
win = QtGui.QMainWindow()



colors=[(255,0,0), (0,255,0), (0,0,255), (0,128,128), (128,128,0), (128,0,128), (200,200,200)]
#Time domain plot
tdDock = Dock("Time domain", size=(500,200))
tdPlotWidget = pg.PlotWidget(title="TD")
tdPlotWidget.plotItem.setDownsampling(mode='peak')
#tdPlotWidget.plotItem.setClipToView(True)
tdDock.addWidget(tdPlotWidget)
tdPlotWidget.addLegend(offset=[-30,-30])
#Bt_curves=[tdPlotWidget.plot([], pen=None, symbolPen=col) for col, label in zip(colors, ['Vx','Vy', 'Vz'])] 
Bt_curves=[tdPlotWidget.plot([],[], pen=col) for col, label in zip(colors, ['Vx','Vy', 'Vz', 'piX', 'mPiX'])] 

#freq domain plot
fdDock = Dock("Frequency Domain", size=(500,200))
fdPlotWidget = pg.PlotWidget(title="Spectrum")
fdPlotWidget.plotItem.setDownsampling(mode='peak')
#fdPlotWidget.plotItem.setClipToView(True)
fdDock.addWidget(fdPlotWidget)
fdPlotWidget.addLegend()
Bf_curves=[fdPlotWidget.plot([],[], pen=col,symbolBrush=col, symbolSize=1) for col, label in zip(colors, ['Vx','Vy', 'Vz', 'piX', 'mPiX'])] 

#parameter tree
paramsDock = Dock("Params", size=(300, 200))     ## give this dock the minimum possible size
paramTree = ParameterTree()
topPT = Parameter.create(name='contParams', type='group')
paramsDock.addWidget(paramTree)



## WINDOW DEFINITION----------------------------------------------------------
area = DockArea()
win.setCentralWidget(area)
win.resize(800,600)
win.setWindowTitle('Plot stuff')

## LAYOUT THE WINDOWS
area.addDock(paramsDock, 'left')      ## place paramsDock at left edge of dock area (it will fill the whole space since there are no other docks yet)
#area.addDock(consoleDock, 'bottom', paramsDock)
area.addDock(tdDock, 'right')  
area.addDock(fdDock, 'top', tdDock)   

from pr import pSerf
import numpy as np
import time

#from picoscope import ps5000a
#from pulse_mag_DAQ import OutputTask, AcquireTask
#outputTask=OutputTask()
#acquireTask=AcquireTask()

if 0:
    socket=None
    port = "5560"
    def retrieveVals(socket):
        st = socket.recv().split(b' ',1)
        if 1:#st[0]==b"mag":
            t,Vx,Vy,Vz=np.fromstring(st[1], dtype='f8').reshape(4,-1)
            print(t[:5])
        else:
            raise ValueError("Don't know what to do with '{}'!".format(st[0]))
        return t,Vx,Vy,Vz
magClient=None
def init():
    global magClient
    #acq.subscribe(b'mag')
    #_,(t,Vx,Vy,Vz)=acq.checkForPublished(b'mag')
    #t,Vx,Vy,Vz=retrieveVals(socket)
    from pmProcessMagServe import Client
    global magClient
    magClient=Client()

    Ncaps=100
    seqTime=1/60.
    for cv in Bf_curves:
        #cv.setData((Nave*cv.yData+newYData)/(Naves+1))
        cv.setData(np.fft.fftfreq(Ncaps, seqTime), np.zeros(Ncaps))
    #Vx,Vy,Vz=np.random.rand(3,Naves)


updateNum=0
def update():
    newVals=magClient.getNewVals()
    global updateNum
    if newVals:
        updateNum+=1
        t,VL=newVals#.retrieveVals(socket)
        #print("got t: {}".format(t))
        #Update magnetic field time-domain graph
        maxTimePoints=graphPT['MaxTimePoints']
        if updateNum>50:
            bFreqPlot=True
            updateNum=0
        else:
            bFreqPlot=False

        for cv_t, cv_f, V in zip(Bt_curves,Bf_curves, VL):
            #newXData=np.hstack([cv_t.xData, t])
            Nnew=V.shape[-1]
            if len(cv_t.xData)==0:
                lastX=0
            else:
                lastX=cv_t.xData[-1]
            #newXData=np.hstack([cv_t.xData, lastX+(1+np.arange(Nnew))*1./60.])
            newXData=np.hstack([cv_t.xData, t])
            newYData=np.hstack([cv_t.yData, V])
            cv_t.setData(newXData[-maxTimePoints:], newYData[-maxTimePoints:])

            if bFreqPlot:
                newXData_f,newYData_f=signal.welch(newYData, fs=graphPT['sampleRate'], nperseg=graphPT['nperseg'])
                cv_f.setData(newXData_f, np.sqrt(newYData_f) )


        if 0:
            Naves=graphPT['NFreqAves']
            for cv, V in zip(Bf_curves, VL):
                newYData=abs(np.fft.fft(V))
                #newXData=np.fft.fftfreq(len(newYData), 1/60.)
                newXData,newYData=signal.welch(V, fs=60., nperseg=300, noverlap=0)
                newYData=np.sqrt(newYData)
                oldX,oldY=cv.getData()
                if np.all(newXData==oldX):
                    newYData=(Naves*oldY+newYData)/(Naves+1)
                cv.setData(newXData, newYData) 


# Params for Parameter Tree
expParDef= [
        #{'name': 'start/stop', 'type': 'action'},
        {'name': 'Experiment', 'type': 'group', 'children': [
            {'name': 'pulseSeqDesc', 'type': 'str', 'value': '[("X", "Y")]'},
            {'name': 'pulseAmps', 'type': 'group', 'children': [
                {'name': 'piX', 'type': 'float', 'value': 1.4, 'step': 1e-2, 'siPrefix': True, 'suffix': 'V', 'limits': (-180,180), 'default': 90.0},
                {'name': 'piY', 'type': 'float', 'value': 1.4, 'step': 1e-2, 'siPrefix': True, 'suffix': 'V', 'limits': (-180,180), 'default': 90.0},
                {'name': 'mPiX', 'type': 'float', 'value': 1.4, 'step': 1e-2, 'siPrefix': True, 'suffix': 'V', 'limits': (-180,180), 'default': -90.0},
                {'name': 'mPiY', 'type': 'float', 'value': -1.4, 'step': 1e-2, 'siPrefix': True, 'suffix': 'V', 'limits': (-180,180), 'default': -90.0},

                {'name': 'piXOff', 'type': 'float', 'value': 0, 'step': 1e-2, 'siPrefix': True, 'suffix': 'V', 'limits': (-90,90), 'default': -0.0},
                {'name': 'mPiXOff', 'type': 'float', 'value': 0, 'step': 1e-2, 'siPrefix': True, 'suffix': 'V', 'limits': (-90,90), 'default': .0},
                {'name': 'piYOff', 'type': 'float', 'value': 0, 'step': 1e-2, 'siPrefix': True, 'suffix': 'V', 'limits': (-90,90), 'default': 0.0},
                {'name': 'mPiYOff', 'type': 'float', 'value': 0, 'step': 1e-2, 'siPrefix': True, 'suffix': 'V', 'limits': (-90,90), 'default': .0},
                {'name': 'pi2XOff', 'type': 'float', 'value': 0, 'step': 1e-2, 'siPrefix': True, 'suffix': 'V', 'limits': (-90,90), 'default': 0.0},
                {'name': 'pi2YOff', 'type': 'float', 'value': 0, 'step': 1e-2, 'siPrefix': True, 'suffix': 'V', 'limits': (-90,90), 'default': .0},
                {'name': 'reverseHeight', 'type': 'float', 'value': 1, 'step': 1e-2, 'siPrefix': True, 'limits': (-2,2), 'default': 1},
                ]},
            {'name': 'fields', 'type': 'group', 'children': [
                {'name': 'Vx', 'type': 'float', 'value': 0., 'step': 1e-3, 'siPrefix': True, 'suffix': 'V', 'limits': (-10,10), 'default': 0.},

                {'name': 'Vy', 'type': 'float', 'value': 0., 'step': 1e-3, 'siPrefix': True, 'suffix': 'V', 'limits': (-10,10), 'default': 0.},
                {'name': 'Vz', 'type': 'float', 'value': 0., 'step': 1e-3, 'siPrefix': True, 'suffix': 'V', 'limits': (-10,10), 'default': 0.},
                ]},
            {'name': 'pulseTiming', 'type': 'group', 'children': [
                {'name': 't0', 'type': 'float', 'value': 30e-6, 'step': 1e-6, 'siPrefix': True, 'suffix': 's', 'limits': (0,20e-3), 'default': 10e-6},
                {'name': 'width', 'type': 'float', 'value': 2.5e-6, 'step': 1e-6, 'siPrefix': True, 'suffix': 's', 'limits': (0,30e-6), 'default': 10e-6},
                {'name': 'pi2RelWidth', 'type': 'float', 'value': 1, 'step': .01, 'limits': (0,2), 'default': 1.},
                {'name': 'tau', 'type': 'float', 'value': 1000e-6, 'step': 1e-6, 'siPrefix': True, 'suffix': 's', 'limits': (0,20e-3), 'default': 1000e-6},
                {'name': 'tLong', 'type': 'float', 'value': 2000e-6, 'step': 1e-6, 'siPrefix': False, 'suffix': 's', 'limits': (0,20e-3), 'default': 2000e-6},
                {'name': 'pumpTime', 'type': 'float', 'value': 10e-6, 'step': 1e-6, 'siPrefix': True, 'suffix': 's', 'limits': (0,8e-3), 'default': 10e-6},
            ]},     
            {'name': 'totalTime', 'type': 'float', 'value': 16.5e-3, 'step': 10e-6, 'siPrefix': True, 'suffix': 's', 'limits': (0,100e-3), 'default': 10e-6},
        ]},
    ]
expControlDef = [
        {'name': 'Other', 'type': 'group', 'children': [
            {'name': 'Toggle Repeating', 'type': 'action'},
            {'name': 'Calculate Grad', 'type': 'action'},
            {'name': 'Naves', 'type': 'int', 'value': 1, 'limits': (1,1000)},
            {'name': 'Nreps', 'type': 'int', 'value': 1, 'limits': (1,20)},
            {'name': 'dV', 'type': 'float', 'value': 0.04},
            {'name': 'dVPulse', 'type': 'float', 'value': 0.1},
            {'name': 'gradList', 'type': 'str', 'value': "Vx, Vy, Vz, piX, piY "},
            ]
        },
        ]
graphParDef = [
        {'name': 'Graphing', 'type': 'group', 'children': [
            {'name': 'start/stop', 'type': 'action'},
            {'name': 'nperseg', 'type': 'int', 'value': 5, 'limits': (1,1000000)},
            {'name': 'MaxTimePoints', 'type': 'int', 'value': 10000},
            {'name': 'sampleRate', 'type': 'float', 'value': 60},
            ]
        },
        ]

processingParamDef = [
        {'name': 'Processing', 'type':'group', 'children': [
            {'name': 'tIgnore', 'type': 'float', 'value':200e-6},
            ]
        }    
    ]
scopeParamDef = [
        {'name': 'Scope', 'type':'group', 'children': [
            {'name': 'VRange', 'type': 'float', 'value':2},
            {'name': 'Ncaps', 'type': 'int', 'value' :100},
            {'name': 'SampleRate', 'type': 'float', 'value' :3e6},
            ]
        }    
    ]


timer = QtCore.QTimer()
timer.timeout.connect(update)

def setPiYAmp(peakYV=5.):
    lY = 0 if RY>0 else peakYV/RY
    if abs(peakYV-lY)>0.5:
        fgK.setLH(lY,peakYV);
    print("New Yrange: {}, {}".format(lY, peakYV))
def setPiZAmp( peakZV=5.):
    lZ = 0 if RZ>0 else peakZV/RZ
    if abs(peakZV-lZ)>0.5:
        fgA.setLH(lZ,peakZV);
    print("New Zrange: {}, {}".format(lZ, peakZV))

glbREPEATING=False;
def controlParamChanged(param, changes):
    global glbREPEATING
    for param, changeType, newVal in changes:
        if changeType=='activated':
            if param.name()=="Calculate Grad":
                gradList=[s.strip(' ,') for s in contPT['gradList'].split(',')]
                gradList=[s for s in gradList if s!='']
                #c.acq.setRepeating(False)
                c.acquireGradData(gradList, Naves=contPT['Naves'], Nreps=contPT['Nreps'], dV=contPT['dV'], dVPulse=contPT['dVPulse'])
                #c.acq.setRepeating(glbREPEATING)
            elif param.name()=="Toggle Repeating":
                if glbREPEATING:
                    c.acq.setRepeating(False)
                else:
                    c.acq.setRepeating(True)
                glbREPEATING=~glbREPEATING


def graphingParamChanged(param, changes):
    for param, changeType, newVal in changes:
        if changeType=='activated':
            if param.name()=="start/stop":
                if timer.isActive():
                    timer.stop()
                else:
                    timer.start(500)
def expParamChanged(param, changes):
    """ Currently stuck on the old version. But what it SHOULD do, is update the expPars bunch object, then re-run setupExperiment. Can modify setup experiment to avoid re-doing absolutely everything each time.
    """
    for param, changeType, newVal in changes:
        paramName = param.name()
        print('  parameter: %s'% paramName)
        print('  changeType:    %s'% changeType)
        print('  data:      %s'% str(newVal))
        print('  ----------')
        #newPars=paramObjToBunch(expPT)
        #glbP.p=newPars
        if paramName=="pulseSeqDesc":
            newVal=eval(newVal)
        c.setupWithChanged(bPermanent=True, **{paramName: newVal})#WithChanged(**newPars)

        if 0:
            if changeType=='activated':
                if paramName=="start/stop":
                    if timer.isActive():
                        timer.stop()
                    else:
                        timer.start()

            elif paramName=='Vx':
                print("write a new Vx")
                outputTask.write(Vx=newVal)
            elif paramName=='Vy':
                outputTask.write(Vy=newVal)
            elif paramName=='Vz':
                outputTask.write(Vz=newVal)
            elif paramName=='piY Amp':
                print("write a new piY")
                setPiYAmp(newVal);
            elif paramName=='piZ Amp':
                setPiZAmp(newVal);

topPT.addChildren(expParDef)
topPT.addChildren(graphParDef)
topPT.addChildren(expControlDef)
#topPT.sigTreeStateChanged.connect(paramChanged)
expPT=topPT.child('Experiment')
expPT.sigTreeStateChanged.connect(expParamChanged)
contPT=topPT.child('Other')
contPT.sigTreeStateChanged.connect(controlParamChanged)
graphPT=topPT.child('Graphing')
graphPT.sigTreeStateChanged.connect(graphingParamChanged)
paramTree.addParameters(topPT)

def paramObjToBunch(paramObj=topPT):
    if paramObj.value() is not None:
        return paramObj.value()
    else:
        return Munch(**{child.name(): paramObjToBunch(child) for child in paramObj})

def updateParamObjFromBunch(bunch, pObj=expPT):
    """ Convert the experiment parameter tree to a Bunch as used by the experiment"""
    #Check if bunch is not actually a bunch
    if not isinstance(bunch, dict): #it's not a bunch, it's just a value
        print("new value = {}".format(bunch))
        if isinstance(bunch, list) or isinstance(bunch, tuple):
            pObj.setValue(str(bunch))
        else:
            pObj.setValue(bunch)
    else:
        for key, val in bunch.items():
            print("update {}".format(key))
            if key in pObj.names:
                updateParamObjFromBunch(val, pObj.child(key))
            else:
                print("Couldn't find parameter {}".format(key))


def updateFromSaved():
    expPT.blockSignals(True)
    updateParamObjFromBunch(glbP.p, pObj=expPT)
    expPT.blockSignals(False)
    try:
        c.setupExperiment()
    except ValueError as e:
        print("Caught exception while updating experiment: {}".format(e.args[0]))

if __name__ == '__main__':
    win.show()
    updateFromSaved()
    init()
    if 0:
        pass;
        #win.show()
        #timer.start(500)
   

    # Data side
    # Pico-scope
    # Connect up to open/close buttons


    


