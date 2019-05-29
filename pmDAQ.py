# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:36:41 2015

@author: Lab User
"""

from PyDAQmx import Task
from numpy import zeros, array
import numpy as np
from time import sleep
import time


from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.DAQmxConstants import *


"""This example is a PyDAQmx version of the ContAcq_IntClk.c example
It illustrates the use of callback functions

This example demonstrates how to acquire a continuous amount of
data using the DAQ device's internal clock. It incrementally stores the data
in a Python list.
"""
import MT
import fractions
def lcm(a,b): return abs(a * b) / fractions.gcd(a,b) if a and b else 0


def createCOPulseTask2(chan, totTime, delay,pulseWidth, numPulses=4,edge=DAQmx_Val_Rising, triggerSource="PFI0"):
    th=Task()
    #th.CreateCOPulseChanFreq ( chan, "", DAQmx_Val_Hz, DAQmx_Val_Low, delay, freq,
    th.CreateCOPulseChanTime ( chan, "", DAQmx_Val_Seconds, DAQmx_Val_Low, delay, totTime-pulseWidth,
                                             pulseWidth);
    th.CfgDigEdgeStartTrig(triggerSource,edge);
    th.CfgImplicitTiming (DAQmx_Val_FiniteSamps, numPulses);
    th.SetTrigAttribute (DAQmx_StartTrig_Retriggerable, True);
    return th
def createCOPulseTask(chan, freq, delay,duty, numPulses=4,edge=DAQmx_Val_Rising, triggerSource="PFI0"):
    th=Task()
    th.CreateCOPulseChanFreq ( chan, "", DAQmx_Val_Hz, DAQmx_Val_Low, delay, freq,
                                             duty);
    th.CfgDigEdgeStartTrig(triggerSource,edge);
    th.CfgImplicitTiming (DAQmx_Val_FiniteSamps, numPulses);
    th.SetTrigAttribute (DAQmx_StartTrig_Retriggerable, True);
    return th

class PmTriggerTask(object):
    initTH=None;
    pmpTH=None
    def __init__(self):
        pass;
    def setParams(self, totalTime, pumpTime=10e-6, Npmp=4, pmpDelay=0, tBetweenPmp=None):
        repRate=1./totalTime
        #self.initTH=createCOPulseTask('Dev12/ctr1', freq= repRate, delay=0, duty= 20e-6*repRate, numPulses=1, edge=DAQmx_Val_Rising, triggerSource='PFI0')
        self.initTH=createCOPulseTask2('Dev12/ctr1', totTime=totalTime, delay=0, pulseWidth=100e-6, numPulses=1, edge=DAQmx_Val_Rising, triggerSource='PFI0')

        if tBetweenPmp is None:
            print("Default pump time??")
            tBetweenPmp= (totalTime-20e-6)/Npmp
        if Npmp==1:
            Npmp=2;
            tBetweenPmp=totalTime-pumpTime
            print("Warning: using single pump pulses is a bit dodgy")
        self.pmpTH=createCOPulseTask2('Dev12/ctr0', totTime=tBetweenPmp, delay=pmpDelay, numPulses=Npmp, edge=DAQmx_Val_Rising, pulseWidth=pumpTime, triggerSource='PFI1')

    def setParamsOld(self, totalTime, pumpTime=10e-6, Npmp=4, pmpDelay=0, tBetweenPmp=None):
        repRate=1./totalTime
        #self.initTH=createCOPulseTask('Dev12/ctr1', freq= repRate, delay=0, duty= 20e-6*repRate, numPulses=1, edge=DAQmx_Val_Rising, triggerSource='PFI0')
        self.initTH=createCOPulseTask('Dev12/ctr1', freq= repRate, delay=0, duty= 0.01, numPulses=1, edge=DAQmx_Val_Rising, triggerSource='PFI0')

        if tBetweenPmp is None:
            tBetweenPmp= (totalTime-20e-6)/Npmp
        self.pmpTH=createCOPulseTask('Dev12/ctr0', freq=1./tBetweenPmp, delay=pmpDelay, numPulses=Npmp, edge=DAQmx_Val_Rising, duty=pumpTime/tBetweenPmp, triggerSource='PFI1')

    def start(self):
        self.initTH.StartTask()
        self.pmpTH.StartTask()
    def stop(self):
        """Should do exception handling, e.g. for if it's already stopped"""
        self.initTH.StopTask()
        self.pmpTH.StopTask()

    def pumpOn(self, val):
        if val:
            self.pmpTH.StartTask()
        else:
            self.pmpTH.StopTask()

    def reset(self):
        try:
            self.stop()
            sleep(0.2)
        except DAQError:
            pass
        finally:
            self.start()

    def clear(self):
        self.initTH.ClearTask()
        self.pmpTH.ClearTask()

    def freeRun(self, bFreeRun=True):
        if bFreeRun==True:
            self.initTH.DisableStartTrig()
        else:
            pass
            #self.initTH.DisableStartTrig()
 







class TriggerOutputDig(object):
    t=None;
    wvfms=None; 
    Nsamps=None
    sampleRate=None
    pulseLength=None


    
    def __init__(self, port="Dev12/port0", startTrigChan="PFI0"):
        self.th=Task()#Task.__init__(self)
        self.th.CreateDOChan(port,"",DAQmx_Val_ChanForAllLines);
        #self.setTiming(sampleRate, pulseLength)
        self.th.CfgDigEdgeStartTrig (startTrigChan, DAQmx_Val_Rising);
        self.th.SetStartTrigRetriggerable(True )
        #DAQmxLoadTask("WvfmOutputTask", byref(self.taskHandle))

    def start(self):
        self.th.StartTask()
    def stop(self):
        try:
            self.th.StopTask()
        except DAQError as e:
            if e.error ==200010:
                self.th.StopTask()
            else:
                raise(e)


    def setTiming(self, sampleRate, waveformLength ):
        Nsamps=int(waveformLength*sampleRate)
        #self.th.CfgSampClkTiming("",sampleRate,DAQmx_Val_Rising,DAQmx_Val_ContSamps, 5*Nsamps);
        self.th.CfgSampClkTiming("",sampleRate,DAQmx_Val_Rising,DAQmx_Val_FiniteSamps, Nsamps);
        self.sampleRate=sampleRate
        self.Nsamps=Nsamps
        
    def reset(self):
        try:
            self.stop()
            sleep(0.2)
        except DAQError:
            pass
        finally:
            self.start()

    def setWaveforms(self, t, *wvfms):    
        Npts=t.size
        sampleRate=1./(t[1]-t[0])
        totalLength=Npts/sampleRate
        self.setTiming(sampleRate, totalLength)

        wvfmPort=np.zeros(t.size, dtype='u4')
        for k,wv in enumerate(wvfms):
            wv[:]=np.where(wv,1,0)
            wv=wv.astype('u4')
            wvfmPort+=(k+1)*wv

        written = int32()
        self.th.WriteDigitalU32(1*wvfmPort.size, 1,10, DAQmx_Val_GroupByChannel, np.hstack(1*[wvfmPort]).astype('u4'),byref(written), None)
        self.t=t
        self.wvfms=wvfms
        return self.t, wvfmPort

class TestTask(object):
    Vx=0;
    Vy=0;
    Vz=0;
    data=None;
    th=None; #Task handle
    
    def __init__(self):
        freq1=10
        freq2=20
        freq3=30
        sampRate=1000
        modAmp=5
        N=1000
        self.data=zeros(3, dtype='f8')
        th=Task()#Task.__init__(self)
        th.CreateAOVoltageChan("Dev12/ao0:2","", -10,10, DAQmx_Val_Volts, None)
        th.CfgSampClkTiming("",sampRate, DAQmx_Val_Rising, DAQmx_Val_ContSamps,N)
        self.th=th
class OutputTaskMod(Task): # <- Use this one (it's got modulation)
    modFreqL=np.ones(3)
    modAmpL=np.zeros(3)
    modAmpLCopy=None
    Vx=0;
    Vy=0;
    Vz=0;
    data=None;
    wvfm=None
    th=None; #Task handle
    
    def __init__(self, Nsamps=1000, sampRate=1000, modAmpL=[5e-3, 5e-3, 5e-3], modFreqL=[5,3, 1]):
        self.data=zeros(3, dtype='f8')
        self.Vx, self.Vy, self.Vz=self.data
        th=Task()#Task.__init__(self)
        th.CreateAOVoltageChan("Dev12/ao0:2","", -10,10, DAQmx_Val_Volts, None)
        th.CfgSampClkTiming("",sampRate, DAQmx_Val_Rising, DAQmx_Val_ContSamps,Nsamps)
        self.th=th
        self.Nsamps=Nsamps
        self.sampRate=sampRate
        self.modFreqL=modFreqL
        self.modAmpL=modAmpL
        self.wvfm=zeros((3, Nsamps), dtype='f8')
        self.t=np.linspace(0,Nsamps/sampRate,Nsamps)
         

    def modOff(self):
        self.modAmpLCopy=self.modAmpL.copy()
        self.modAmpL=np.zeros(3)
        self.write()
    def modOn(self):
        if self.modAmpLCopy is not None:
            self.modAmpL=self.modAmpLCopy.copy()
            self.write()
        else:
            print("Modulation hadn't been turned off, doing nothing")
    def setModFreq(self, Vx=None, Vy=None, Vz=None):
        if Vx is not None:
            self.modFreqL[0]=Vx;
        if Vy is not None:
            self.modFreqL[1]=Vy;
        if Vz is not None:
            self.modFreqL[2]=Vz;
        self.write()

    def setModAmp(self, Vx=None, Vy=None, Vz=None):
        if Vx is not None:
            self.modAmpL[0]=Vx;
        if Vy is not None:
            self.modAmpL[1]=Vy;
        if Vz is not None:
            self.modAmpL[2]=Vz;
        self.write()

    def write(self, Vx=None, Vy=None, Vz=None):
        if Vx is not None:
            self.Vx=Vx;
        if Vy is not None:
            self.Vy=Vy;
        if Vz is not None:
            self.Vz=Vz;
        self.data[:]=[self.Vx, self.Vy, self.Vz]
        self.writeStraight(self.data)
        return self.data

    def writeStraight(self, Vlist):
        print("writing (Vx, Vy, Vz): {}".format(Vlist))
        written = int32()
        NsampsMax=50000 
        #Calc new sample Nsamps
        periods=array([1./f for f in self.modFreqL])

        #Make sure the periods are even numbers of samples
        #periods=[int(p*self.sampRate)/self.sampRate if not p==np.inf else 0 for p in periods]
        periods=[p if not p==np.inf else 0 for p in periods]

        #newTotT=lcm(lcm(periods[0], periods[1]), periods[2])
        newTotT=periods[0]*periods[1]*periods[2]
        if newTotT/self.sampRate > NsampsMax:
            print("Warning: LCM of periods is too high!")
            newTotT=NsampsMax/self.sampRate
        Nsamps=int(newTotT*self.sampRate)
        k=2
        while Nsamps < 5000:
            Nsamps=int(k*newTotT*self.sampRate)
            k+=1
            #Nsamps=np.ceil(1000./Nsamps)*Nsamps
        print("num samps: {}".format(Nsamps))
        self.wvfm=zeros((3, Nsamps), dtype='f8')
        self.t=np.linspace(0,Nsamps/self.sampRate, Nsamps)

        for k in range(3):
            self.wvfm[k]=np.sin(2*np.pi*self.t*self.modFreqL[k])*self.modAmpL[k]+Vlist[k]

        self.th.StopTask()
        self.th.WriteAnalogF64(Nsamps, 1,10, DAQmx_Val_GroupByChannel, self.wvfm, byref(written), None)
        #self.th.StartTask()

    def writeDeviation(self, dVx=0., dVy=0., dVz=0.):
        currData=self.data.copy()
        currData+=np.array([dVx, dVy, dVz])
        self.writeStraight(currData)

    def writePrevious(self):
        self.writeStraight(self.data)
class OutputTask(Task):
    Vx=0;
    Vy=0;
    Vz=0;
    data=None;
    
    def __init__(self):
        self.data=zeros(3, dtype='f8')
        Task.__init__(self)
        DAQmxLoadTask("TestVoltageOutTask", byref(self.taskHandle))

    def write(self, Vx=None, Vy=None, Vz=None):
        if Vx is not None:
            self.Vx=Vx;
        if Vy is not None:
            self.Vy=Vy;
        if Vz is not None:
            self.Vz=Vz;
        self.data[:]=[self.Vx, self.Vy, self.Vz]
        self.writeStraight(self.data)
        if 0:
            written = int32()
            self.WriteAnalogF64(1, #numSampsPerChan
                               1, #Autostart
                               10,#timout
                               DAQmx_Val_GroupByScanNumber, #data layout
                               self.data, #writeArray
                               byref(written), #sampsPerChanWritten
                               None) #reserved
            #print(time.time()-self.time)
            #print(len(self.a))
        return self.data

    def writeStraight(self, data):
        print("writing (Vx, Vy, Vz): {}".format(data))
        written = int32()
        self.WriteAnalogF64(1, #numSampsPerChan
                        1, #Autostart
                        10,#timout
                        DAQmx_Val_GroupByScanNumber, #data layout
                        np.array(data,dtype='f8')[:3], #writeArray
                        byref(written), #sampsPerChanWritten
                        None) #reserved

    def writeDeviation(self, dVx=0., dVy=0., dVz=0.):
        currData=self.data.copy()
        currData+=np.array([dVx, dVy, dVz])
        self.writeStraight(currData)

    def writePrevious(self):
        self.writeStraight(self.data)
        

from PyDAQmx import Task
from numpy import zeros

import time

class AcquireToFileTask(Task):
    def __init__(self, targetFileName=None, saveRate=125, Nds=2, startOnTrig=None):
        Task.__init__(self)
        self.saveRate=saveRate
        self.Nds=Nds
        self.data = np.zeros(saveRate*Nds*4)
        self.sampRate=saveRate*Nds
        self.a = []
        self.CreateAIVoltageChan("Dev12/ai0:3","",DAQmx_Val_RSE,-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("",self.sampRate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,self.sampRate)
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer,self.sampRate,0)
        self.AutoRegisterDoneEvent(0)
        self.t0=None

        if startOnTrig is not None:
            self.CfgDigEdgeStartTrig(startOnTrig,DAQmx_Val_Rising);
        if targetFileName is not None:
            self.saveFile=open(targetFileName, 'wb')
        else:
            self.saveFile = None
    def EveryNCallback(self):
        if self.t0 is None:
            #print("init")
            self.t0=time.time()
        else:
            pass
            #print("rate: {}".format(float(len(self.a))/(time.time()-self.t0)), flush=True)
        read = int32()
        self.ReadAnalogF64(self.sampRate,10.0,DAQmx_Val_GroupByScanNumber,self.data,self.sampRate*4,byref(read),None)
        dnSamped=self.data.reshape(self.Nds,-1,4).mean(axis=0)
        self.a.append(dnSamped)

        if self.saveFile:
            dnSamped.T.tofile(self.saveFile)
        return 0 # The function should return an integer
    def DoneCallback(self, status):
        print("Status: {}".format(status.value))
        return 0 # The function should return an integer





if __name__=="__main__":
    from pylab import plot, figure
    import pylab as pl
    #task=AcquireTask(20)
    #outTask=OutputSingleValsTask()
    #arr=task.acquire()
    #t=task.getT()
    #task.StartTask()
    task=CallbackTask('temp2.bin')
    task.StartTask()
    input('Acquiring samples continuously. Press Enter to interrupt\n')
    task.StopTask()
    task.ClearTask()
    
    #sleep(10)
    #input('Acquiring samples continuously. Press Enter to interrupt\n')
    
    #task.StopTask()
    #task.ClearTask()
    #outTask.ClearTask()
