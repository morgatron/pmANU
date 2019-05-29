""" Module to perform setup of the instruments in the pulsed-magnetometer experiment.
Functions to set them all up given experiment parameters, as well as stop() and clear() to stop and clear the experiments.
"""
import pdb
from munch import Munch
import dill as pickle
import MT
import numpy as np
import pylab as pl
import pmDAQ as daq
from pmDAQ import  PmTriggerTask, OutputTaskMod, DAQError, AcquireToFileTask
import tdsscope
from pmPulseSeq import generatePulseMagPattern, generateTriggerWaveforms
from pmFG import PmFgController
import pmParams
import pmAcquire as acq
from time import sleep
import rigolfg
import visa
import os
from copy import deepcopy
import fgAgilent
import pmProcessRaw as proc

rgl=rigolfg.RigolFG()

ldCont=visa.ResourceManager().get_instrument("GPIB::2::INSTR")
#pmpCont=visa.ResourceManager().get_instrument()....

def probeOn():
    ldCont.write("LDON 1")
def probeOff():
    ldCont.write("LDON 0")

glbP=pmParams.GlbParams()

ampConvD=glbP.loadCoilCal()
def clear():
    try:
        outputTask.th.ClearTask()
    except Exception:
        pass;

    try:
        signalTask.ClearTask()
    except Exception:
        pass

    try:
        #trigTask.th.ClearTask()
        trigTask.clear()
    except Exception:
        pass;

    try:
        ps.close()
    except Exception:
        pass;

def getCurrentOutputs():
    outputs=Munch()
    outputs.fields=Munch(
                    vx=outputTask.Vx, 
                    vy=outputTask.Vy,
                    vz=outputTask.Vz,
                    )
    #outputs.trig=Munch(t=trigTask.t, 
    #            pmp=trigTask.wvfms[0], 
    #            init=trigTask.wvfms[1]
    #            )
    outputs.fg=Munch(t=fgCont.t,
                    vx=fgCont.VX, 
                    vy=fgCont.VY,
                    vz=fgCont.VZ,
                    )
    return outputs

outputTask=None
trigTask=None
fgCont=None
trigFG=None
def init():
    global outputTask, trigFG, fgCont
    outputTask=OutputTaskMod()
    #outputTask.setModFreq(3,4,5)
    outputTask.setModAmp(0.005,0.005,0.005)
    print("outputtask inited")
    #trigTask=PmTriggerTask()#TriggerOutputDig()
    #print("trigTask inited")
    trigFG=fgAgilent.AgilentFG()
    fgCont=PmFgController()
    print("fgController inited")
    acq.init(bRemote=True)
    #acq.subscribe(b'raw')
    print("acquisition communication inited")
    setupExperiment()
    print("Experiment setup")

def setupLasers(lasParams):
    ldCont.write('TEMP {}'.format(lasParams.prbT))
    ldCont.write('SILD {}'.format(lasParams.prbI))
    if 0:
        if lasParams.pmpI < 2:
            pmpCont.write('MAIN:CURRENT:SETPOINT {}'.format(lasParams.pmpI))
        else:
            raise ValueError("pump current is too high")
    

def setupDCFields(params):
    outputTask.write(**params.fields)

def convertAmps2Volts(pulseAmps):
    parNames=['piX', 'mPiX', 'piY', 'mPiY', 'pi2X', 'pi2Y', 'piYOff', 'piXOff','mPiXOff', 'mPiYOff', 'pi2YOff', 'pi2XOff', 
            'piX_B', 'mPiX_B', 'piY_B', 'mPiY_B']
    if pulseAmps.voltUnits:
       return pulseAmps

    cnvted=deepcopy(pulseAmps)
    for name in parNames:
        if pulseAmps[name] is None:
            continue
        if name in ['piX', 'mPiX', 'piYOff', 'mPiYOff', 'pi2X', 'piX_B', 'mPiX_B']:
            ax='X' 
        elif name in ['piY', 'mPiY', 'piXOff', 'pi2Y', 'mPiXOff', 'piY_B', 'mPiY_B']:
            ax="Y"
        elif name in ['pi2YOff', 'pi2XOff']:
            ax='Z'
        else:
            raise ValueError("Not sure which axis {} belongs to!".format(name))
        theta=pulseAmps[name]
        cnvted[name]=ampConvD[ax](theta/180.*np.pi)
    return cnvted

def setupFGs(params):
    sampRates=[fgCont.rateX, fgCont.rateY, fgCont.rateZ]
    tFg,Vx,Vy,Vz=generatePulseMagPattern(params.pulseSeqDesc, params.pulseTiming, params.pulseAmps, sampleRate=sampRates)
    fgCont.allOff()
    fgCont.setWaveForms(tFg, Vx, Vy, Vz)
    fgCont.updateOutputs()
    fgCont.allOn()

def setupTrigger(params):
    if 0:
        try:
            trigTask.stop()
        except DAQError:
            pass

    if 1: #the old way  using a digital wave form
        digSampRate=1e6
        tTrig, trigWvfmPump, trigWvfmInit = generateTriggerWaveforms(params.pulseTiming,params.pulseSeqDesc, Npts=(params.totalTime-30e-6)*digSampRate, sampleRate=digSampRate)
        #trigTask.setWaveforms(tTrig, trigWvfmPump, trigWvfmInit)
        trigWvfmPump[0]=0
        #trigWvfmPump[-200:-100]=1.
        #trigWvfmPump[-100:]=0.
        #trigWvfmPump=np.where(trigWvfmPump>0, 0., 1.0)
        
        trigFG.uploadWaveform(trigWvfmPump)
        trigFG.setPeriod(tTrig.size*(tTrig[1]-tTrig[0]))
        if params.pulseTiming.pumpTime==0:
            trigFG.setLH(-.01,.01)
        else:
            trigFG.setLH(0,4.0)
    if 0:

        trigTask.setParams(params.totalTime, pumpTime=params.pulseTiming.pumpTime, Npmp=len(params.pulseSeqDesc), pmpDelay=0, tBetweenPmp=params.pulseTiming.tLong )
        trigTask.start()

#Don't think I actually want thiese... use pmAcquire instead

def setupExperiment():
    _setupExperiment(glbP.p)
def _setupExperiment(params):
    params=deepcopy(params)
    setupTrigger(params)
    setupDCFields(params)
    if not params.pulseAmps.voltUnits:
        print("converting units")
        params.pulseAmps=convertAmps2Volts(params.pulseAmps)
        print("new amps: {}".format(params.pulseAmps))
    setupFGs(params)
    setupLasers(params.lasParams)
    acq.setScopeParams(acqTime=params.totalTime)
    sleep(0.2)

def setupWithChanged(bPermanent=False, **kwargs):
    if bPermanent:
        glbP.change(**kwargs)
        setupExperiment()
    else:
        _setupExperiment(glbP.getChanged(**kwargs))
def setupWithTweaked(bPermanent=False, frac=1, **kwargs):
    if bPermanent:
        glbP.tweak(kwargs,frac=frac)
        setupExperiment()
    else:
        _setupExperiment(glbP.getTweaked(frac=frac,**kwargs))
    print("tweaks: {}".format(kwargs))

def showWaveforms():
    o=getCurrentOutputs()

    if 0:
        pl.figure("Trig Task Outputs") 
        pl.suptitle("Triggers")
        ax1=pl.subplot(211)
        pl.plot(o.trig.t, o.trig.init)
        ax2=pl.subplot(212, sharex=ax1)
        pl.plot(o.trig.t, o.trig.pmp)

    pl.figure()
    pl.suptitle("FG outputs")
    axFG1=pl.subplot(311)
    pl.plot(o.fg.t, o.fg.vx)
    pl.subplot(312,sharex=axFG1)
    pl.plot(o.fg.t, o.fg.vy)
    pl.subplot(313,sharex=axFG1)
    tz=np.linspace(o.fg.t[0], o.fg.t[-1], o.fg.vz.size)
    pl.plot(tz, o.fg.vz)

    print(o.fields)

lasParamNames=['prbI', 'prbT', 'pumpTime']
fieldParamNames=['Vx', 'Vy', 'Vz']
pulseParamNames1=['piX', 'piY', 'mPiX', 'mPiY',]
pulseParamNames2=['piXOff', 'mPiXOff', 'piYOff', 'mPiYOff',
    'pi2XOff', 'pi2YOff']#,'pi2RelWidth']

def locAcquire(Naves=1, Nds=1, ):
    numAttempts=1
    while 1:
        numAttempts+=1
        try:
            if 0:
                while acq.glbSOCKET_PUBSUB.poll(50):
                    while acq.glbSOCKET_PUBSUB.poll(50):
                        print("Clearing old data...")
                        acq.checkForPublished()
            sleep(0.2)
            _,out,dt=acq.acquireStreaming(Nmin=Naves, bNoStale=True)#Raw()
            #out=MT.smoothalong(out,Nds, axis=-1)[...,::Nds]
            out=MT.downsample_npts(out,Nds, axis=-1)
            #outL.append(out)
            #out=np.mean(out, axis=0)
        except DAQError:
            print("trigger stopped, try again")
        else:
            break
    t=np.arange(out.shape[-1])*dt*Nds
    return t, out

def getZeroLevel(Naves=50, par=None):
    """Just turn off the pump for a little while and measure the level
    """
    if par is None:
        par=glbP.p
    oldPumpTime=par.pulseTiming.pumpTime
    par.pulseTiming.pumpTime=0
    _setupExperiment(par)#WithChanged(tau=4.1e-3, pulseSeqDesc=testSeq)
    t,dat=locAcquire(Naves)
    v0=dat.mean()
    par.pulseTiming.pumpTime=oldPumpTime
    _setupExperiment(par)#WithChanged(tau=4.1e-3, pulseSeqDesc=testSeq)
    return v0


lasDevDict=dict(
    prbI=1,
    prbT=0.5,
    pumpTime=-5e-6
    )
def acquireGradData( paramNames=fieldParamNames+pulseParamNames1,
        Nreps=2, Naves=200, dV=0.1, dVPulse=0.5, NdownSample=1, bSyncPulse1Amps=False):
    #acq.startStreaming(bKeepSampsPerSeg=True) #Reset
    outputTask.modOff()
    setupExperiment()
    sleep(0.4)
    _,dat,dt=acq.acquireStreaming()
    acquire=lambda: locAcquire(Naves)[1]
    dat=acquire()
    Npts=dat.shape[-1]
    t=np.arange(Npts)*dt
    #datA=zeros((4,t.size), dtype='f8' )
    #datD=defaultdict(lambda: zeros(t.size, dtype='f8'))
    #setupExperiment()
    #sleep(0.4)
    datD={parName:np.zeros(Npts,dtype='f8') for parName in ['ref']+paramNames}
    print("acq shape: {}".format(dat.shape))
    #if paramNames is None:
    #    paramNames=fieldParamNames+pulseParamNames
    for k in range(Nreps):
        if k>0:
            setupExperiment()
            sleep(0.4)
        raw=acquire().mean(axis=0)
        # Check in case the number of points has changed- it usually only gets smaller
        Nnew=min([raw.size, datD['ref'].size])
        datD['ref']=datD['ref'][:Nnew]
        datD['ref']+=raw[:Nnew]
        for paramName in paramNames:
            slpTime=0
            if paramName in fieldParamNames: 
                dev=dV
            elif paramName in lasParamNames:
                dev=lasDevDict[paramName]
            else:
                dev=dVPulse
            setupWithTweaked(bSyncPulse1Amps=bSyncPulse1Amps, **{paramName: dev})
            if paramName=='prbT':
                sleep(1.0)
            sleep(0.4)
            #acq.startStreaming() #Reset
            raw=acquire().mean(axis=0)
            Nnew=min([raw.size, datD[paramName].size])
            datD[paramName]=datD[paramName][:Nnew]
            datD[paramName]+=raw[:Nnew]
            if paramName=='prbT':
                sleep(1.0)
        
    for key in datD.keys():
        datD[key]/=Nreps
        datD[key]=MT.smooth(datD[key],NdownSample)[::NdownSample]
    dS_dB={}
    ref=datD.pop('ref')
    for key in datD.keys():
        if key in fieldParamNames:
            dev = dV  
        elif key in lasParamNames:
            dev=lasDevDict[key]
        else: 
            dev=dVPulse
            
        try:
            dS_dB[key]=(datD[key]-ref)/dev
        except ValueError:
            newN=min([ datD[key].size, ref.size ])
            datD[key]=datD[key][:newN]
            ref=ref[:newN]
            dS_dB[key]=(datD[key]-ref)/dev

    dS_dB['ref']=ref
    setupExperiment()
    glbP.saveCal(t, dS_dB)
    outputTask.modOn()
    return t, dS_dB


def acquireAndSave(filename, Nacqs, Nsegs=100, grad=None, notes='', Nds=1, gradKwargs={}):
    if grad=='load':
        grad=glbP.loadCal()['grad']
    if grad is None:
        if 'paramNames' not in gradKwargs.keys():
            t, grad=acquireGradData(['Vx', 'Vy', 'Vz', 'piX', 'mPiX', 'piY', 'mPiY'], NdownSample=Nds, **gradKwargs)
        else:
            t, grad=acquireGradData(**gradKwargs)

    datL=[]
    magAcq=AcquireToFileTask(targetFileName=None)
    magAcq.StartTask()
    t,dat=locAcquire(Naves=Nsegs, Nds=Nds)
    magAcq.StopTask()
    magAcq.ClearTask()
    sensorsD={'mag': np.vstack(magAcq.a)}
    saveDataSet(filename, t, dat, grad=grad, sensorsD=sensorsD)
    for n in range(Nacqs-1):
        magAcq=AcquireToFileTask(targetFileName=None)
        magAcq.StartTask()
        t,dat=locAcquire(Naves=Nsegs, Nds=Nds)
        magAcq.StopTask()
        magAcq.ClearTask()
        sensorsD={'mag': np.vstack(magAcq.a)}
        addSaved(filename, dat, sensorsD=sensorsD)
    return



datDir="c:/PulseSerfData/"
def saveDataSet(filename, t, dat, sensorsD=None, grad=None, params=None, notes=''):
    if grad is None:
        grad=glbP.loadCal()
    if params is None:
        params=glbP.p
    basePath=datDir+filename
    infoD={
            'params':params,
            'notes':notes,
        }

    baseDir=os.path.dirname(basePath)
    if not os.path.exists(baseDir):
        os.makedirs(baseDir)
    pickle.dump(infoD, open(basePath+'_pars.pkl', 'wb'), protocol=2)
    np.savez(basePath+'_t.npz', t)
    addSaved(filename, dat, sensorsD=sensorsD)
    #np.savez_compressed(basePath+'_dat000.npz', dat)
    np.savez(basePath+'_grad.npz', **grad)
    return

def addSaved(filename, dat, sensorsD=None):
    basePath=datDir+filename
    if not os.path.exists(basePath+'_pars.pkl'):
        raise ValueError("No base data set")
    for k in range(100): 
        fname=basePath+'_dat{:03d}.npz'.format(k)
        if not os.path.exists(fname):
            np.savez(fname, data=dat)
            if sensorsD is not None:
                fnameSensors=basePath+'_sensors{:03d}.npz'.format(k)
                np.savez(fnameSensors, **sensorsD)
            break
    else:
        raise ValueError("Can't find a free spot for the file, last tried was: {}".format(fname))
    return
    





def orientate():
    """Apply some modulation to each axis, and record which axes it appears on.
    results:
    Vx -> [Oxx, Oxy, Oxz]
    Vy -> [Oyx, Oyy, Oyz] -> Mxyz
    Vz -> [Ozx, Ozy, Ozz]
    Finding the eigenvectors of this should give me what I want? Or maybe this transposed?
    """



vSclFact=1.
p0=Munch(
    pulseTiming=Munch(
        t0=20e-6,
        width=4e-6,
        pi2RelWidth=1.,
        tau=1400e-6,
        tLong=1700e-6,
        tShort=300e-6,
        tauShort=150e-6,
        depolLength=0e-6,
        pumpTime=15e-6
    ),
    pulseSeqDesc=[ ('X', 'Y'), ('mX', 'Y'),
            ('Y', 'X'), ('mY', 'X'),
            ('X', 'mY'), ('mX', 'mY'),
            ('Y', 'mX'), ('mY', 'mX'),
    ],
    pulseAmps=Munch(
        piX=90.,
        piY=90.,
        mPiX=-90,
        mPiY=-90,
        pi2X=90,
        pi2Y=-90,
        piXOff=0.,
        mPiXOff=0.0,
        piYOff=0.0,
        mPiYOff=0.0, 
        pi2XOff=.0,
        pi2YOff=0.0,  
        bonusZHeight=(0,0,0),
        reverseHeight=-1,
        voltUnits=False,
    ),
    fields=Munch(
        Vx=1.00,
        Vy=-0.0,
        Vz=-0.00,
    ),
    totalTime=1./62, # Period of mains AC
)
import pmCalibrate as cal

if __name__=="__main__":
    from importlib import reload
    #glbP.change(Vx=5.0, Vz=-6.0, Vy=2.0)
    mgc=get_ipython().magic
    mgc(u"%load_ext autoreload")
    mgc(u"%autoreload 2")
    mgc(u"%matplotlib qt")
    #glbP.p=p0
    init()
    cal.c.acq=acq
    cal.c.setupExperiment=setupExperiment
    cal.c._setupExperiment=_setupExperiment
    cal.c.locAcquire=locAcquire
    cal.c.glbP=glbP
    cal.c.pulseParamNames1=pulseParamNames1
    cal.c.pulseParamNames2=pulseParamNames2
    cal.c.fieldParamNames=fieldParamNames
    cal.c.getZeroLevel=getZeroLevel
    cal.c.acquireGradData=acquireGradData
    #acq.init()
    #acq.startStreaming()
    #sleep(0.5)
    #zero1()
    #zeroMag1()
    #setupExperiment()
    #showWaveforms()
    #pl.show()0
