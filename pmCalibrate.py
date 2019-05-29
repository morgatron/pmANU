import pmParams
import pmProcessRaw as proc
import numpy as np
glbP=pmParams.GlbParams()

class Contr(object):
    def __init__(self):
        self.locAcquire=None
        self.setupExperiment=None
        self._setupExperiment=None
        self.acq=None
        self.pulseParamNames2=None
        self.pulseParamNames1=None
c=Contr()

       
def setCalParams():
    glbP.saveParams(glbP.p, 'calBackup')
    p=glbP.p
    p.pulseSeqDesc=pmParams.doubleFullSeqLong
    p.pulseTiming.tau=3.2e-3
    p.pulseTiming.tLong=3.4e-3
    p.pulseTiming.tShort=920e-6
    p.pulseTiming.tauShort=725e-6
    p.pulseTiming.pumpTime=25e-6
    p.t0=35e-6
    p.totalTime=34.65e-3
    glbP.p=p
    c.setupExperiment()

def set4PulseTiming():
    glbP.change(tau=3.4e-3,
                tauShort=0,
                tLong=4.12e-3,
                t0=20e-6,
                pumpTime=18e-6,
                tShort=0,
                pulseSeqDesc=[('X', 'Y'), ('mX', 'Y'), ('Y', 'X'), ('mY', 'X')],
                )

def revertCalParams():
    pCaled=glbP.p
    pOld=glbP.loadParams('calBackup')
    pOld.pulseAmps=pCaled.pulseAmps
    pOld.fields=pCaled.fields
    glbP.p=pOld

 

def zero1(Nsteps=1, gradD=None, Naves=50, frac=0.8, gradKwargs={'Naves':100, 'Nreps':1, 'dVPulse':1, 'dV':0.2}, bDoMinusToo=False):
    """  Set initial, +ve pulse amplitudes by trying to flatten the periods after the first pulses
    """
    #if gradD is None:
    #    cal=glbP.loadCalData()
    #    grad=cal['grad']
    #    tGrad=cal['t']
    par0=glbP.p
    lastTweak=None
    relevantPars=['piX', 'piY']#, 'mPiX', 'mPiY']
    if bDoMinusToo:
        relevantPars+=['mPiX', 'mPiY']
    for k in range(Nsteps):
        v0=c.getZeroLevel(Naves=Naves)
        t,grad=c.acquireGradData(paramNames=relevantPars, bSyncPulse1Amps=True, **gradKwargs)
        ref=grad['ref']-v0
        endDumpT=par0.pulseTiming.tLong-par0.pulseTiming.tau+20e-6
        print("endDumpT:{}".format(endDumpT))
        fitP=proc.fitPlain(t,ref, par0, grad, toFit=relevantPars,endDumpT=endDumpT, startDumpT=300e-6, bAddDC=False, bFitShort=False)
        #fitP=np.where(fitP>9.9, 9.9, fitP)
        #fitP=np.where(fitP<-9.9, -9.9, fitP)
        adjustD={k:-v for k,v in zip(relevantPars,fitP)}
        #fitD.pop('Vz')
        print("ADJD: {}".format(adjustD))
        if lastTweak is None:
            lastTweak=fitP
        else:
            #diffD={k:fitD[k]-lastTweak[k] for k in fitD.keys()}
            diffP=fitP-lastTweak
            if np.any(diffP>0):
                print("Stopping because step size is increasing (new step: {}, last step:{})".format(fitP, lastTweak))
                return
            lastTweak=fitP


        c.glbP.tweak(adjustD, frac=frac,)
        c.setupExperiment()
        #setupWithTweaked(bPermanent=True, **fitD)

def zero2(Nsteps=2, gradD=None, frac=0.8, gradKwargs={'Naves':50, 'Nreps':1, 'dVPulse':1, 'dV':0.2}):
    """ Set negative pulse amplitudes and Vx, Vy by trying to flatten the subtracted +- pulse windows
    """
    #if gradD is None:
    #    cal=glbP.loadCalData()
    #    grad=cal['grad']
    #    tGrad=cal['t']
    par0=glbP.p
    lastTweak=None
    fieldPars=['Vx', 'Vy', 'mPiX', 'mPiY']
    for k in range(Nsteps):
        t,grad=c.acquireGradData(paramNames=fieldPars, bSyncPulse1Amps=True, **gradKwargs)
        ref=grad['ref']
        endDumpT=par0.pulseTiming.tLong-par0.pulseTiming.tau+100e-6 
        print("endDumpT:{}".format(endDumpT))
        fitP=proc.fitSubbed(t,[ref], par0, gradD=grad, toFit=fieldPars,endDumpT=endDumpT, startDumpT=300e-6, bFitShort=False)[0]
        #fitP=np.where(fitP>9.9, 9.9, fitP)
        #fitP=np.where(fitP<-9.9, -9.9, fitP)
        adjustD={k:-v for k,v in zip(fieldPars,fitP)}
        #fitD.pop('Vz')
        print("fitD: {}".format(adjustD))
        
        if lastTweak is None:
            lastTweak=fitP
        else:
            #diffD={k:fitD[k]-lastTweak[k] for k in fitD.keys()}
            diffP=fitP-lastTweak
            if np.any(diffP>0):
                print("Stopping because step size is increasing (new step: {}, last step:{})".format(fitP, lastTweak))
                return
            lastTweak=fitP


        c.glbP.tweak(adjustD, frac=0.7,)
        c.setupExperiment()
        #setupWithTweaked(bPermanent=True, **fitD)

def zero3(Nsteps=1, gradD=None, Naves=50, gradKwargs={'Naves':100, 'Nreps':1, 'dVPulse':1, 'dV':0.2}):
    """ Set Vz and off-axis componetns for the initial pulses by zeroing difference between windows where the first pulse is identical, but the second pulse is opposite.
    Should bring the mean polarization after the first pulse to be orthogonal to the probe, and in the plane containing the probe and the X(Y) coil axis
    """
    #if gradD is None:
    #    cal=glbP.loadCalData()
    #    grad=cal['grad']
    #    tGrad=cal['t']
    par0=glbP.p
    lastTweak=None
    fieldPars=['Vz', 'piXOff', 'piYOff', 'mPiXOff', 'mPiYOff']
    for k in range(Nsteps):
        t,grad=c.acquireGradData(paramNames=fieldPars, bSyncPulse1Amps=True, **gradKwargs)
        ref=grad['ref']
        endDumpT=80e-6 
        print("endDumpT:{}".format(endDumpT))
        fitP=proc.fitSubbed(t,[ref], par0, gradD=grad, toFit=fieldPars,endDumpT=endDumpT, startDumpT=250e-6, bFitShort=True, toSubL=['x2', 'xm2', 'y2', 'ym2'])[0]
        #fitP=np.where(fitP>9.9, 9.9, fitP)
        #fitP=np.where(fitP<-9.9, -9.9, fitP)
        adjustD={k:-v for k,v in zip(fieldPars,fitP)}
        #fitD.pop('Vz')
        print("fitD: {}".format(adjustD))
        
        if lastTweak is None:
            lastTweak=-fitP
        else:
            #diffD={k:fitD[k]-lastTweak[k] for k in fitD.keys()}
            diffP=fitP+lastTweak
            if np.any(diffP>0):
                print("Stopping because step size is increasing (new step: {}, last step:{})".format(fitP, lastTweak))
                return


        c.glbP.tweak(adjustD, frac=0.7,)
        c.setupExperiment()
        #setupWithTweaked(bPermanent=True, **fitD)

def zero4(Nsteps=3, gradD=None, Naves=100, gradKwargs={'Naves':100, 'Nreps':1, 'dVPulse':7}):
    """ Adjust the off-axis component for the second pulse by adding a Z component to it. This assumes the probe and Bz direction are roughly aligned.
    """
    #if gradD is None:
    #    cal=glbP.loadCalData()
    #    grad=cal['grad']
    #    tGrad=cal['t']
    par0=glbP.p
    lastTweak=None
    relevantPars=['pi2XOff', 'pi2YOff']#, 'mPiX', 'mPiY']
    for k in range(Nsteps):
        v0=c.getZeroLevel(Naves=Naves)
        t,grad=c.acquireGradData(paramNames=relevantPars, bSyncPulse1Amps=True, **gradKwargs)
        ref=grad['ref']-v0
        endDumpT=100e-6
        print("endDumpT:{}".format(endDumpT))
        fitP=proc.fitPlain(t,ref, par0, grad, toFit=relevantPars,endDumpT=endDumpT, startDumpT=250e-6, bAddDC=False, bFitShort=False)
        #fitP=np.where(fitP>9.9, 9.9, fitP)
        #fitP=np.where(fitP<-9.9, -9.9, fitP)
        adjustD={k:-v for k,v in zip(relevantPars,fitP)}
        #fitD.pop('Vz')
        print("ADJD: {}".format(adjustD))
        if lastTweak is None:
            lastTweak=fitP
        else:
            #diffD={k:fitD[k]-lastTweak[k] for k in fitD.keys()}
            diffP=fitP-lastTweak
            if np.any(diffP>0):
                print("Stopping because step size is increasing (new step: {}, last step:{})".format(fitP, lastTweak))
                return
            lastTweak=fitP


        c.glbP.tweak(adjustD, frac=0.4,)
        c.setupExperiment()
        #setupWithTweaked(bPermanent=True, **fitD)
from collections import defaultdict
from scipy import interpolate
def calibrate(winLength=100e-6, vMax=2.0, Nsteps=50, Naves=50):
    """ Calibrate conversion from volts to degrees for the given parameters
    """
    par=glbP.p
    acq.startStreaming()
    par.pulseTiming.tau=50e-6
    par.pulseTiming.tLong=4e-3
    par.pulseTiming.tau=100e-6
    amps=par.pulseAmps
    par.pulseAmps.voltUnits=True
    c.acq.setScopeParams(VRange=10)

    yTestSeq=[('Y', 'Y'), ('X', 'X'), ('mX', 'mX'), ('mY', 'mY')]#,('X', 'Y'), ('X', 'mY'), ('mX', 'Y'), ('mX', 'mY')]

    par.pulseSeqDesc=yTestSeq
    #acquire with pump pulses off
    #trigTask.pumpOn(False)
    if 0:
        oldPumpTime=par.pulseTiming.pumpTime
        par.pulseTiming.pumpTime=0
        c._setupExperiment(par)#WithChanged(tau=4.1e-3, pulseSeqDesc=testSeq)
        t,dat=c.locAcquire(Naves)
        v0=dat.mean()
        par.pulseTiming.pumpTime=oldPumpTime
        c._setupExperiment(par)#WithChanged(tau=4.1e-3, pulseSeqDesc=testSeq)
    v0=c.getZeroLevel()

    resD=defaultdict(list)
    if 1:
        vA=np.linspace(-vMax, vMax, Nsteps)
        for v in vA:
            tp=par.pulseTiming
            par.pulseAmps.pi2X=v
            par.pulseAmps.pi2Y=v
            c._setupExperiment(par)
            #setupWithChanged(**{'pi2X':v, 'pi2Y':v})
            sleep(0.2)
            t,dat=c.locAcquire(Naves)
            dat=dat.mean(axis=0)
            magData,pulseData,depolData=proc.splitRawSubsecs(t, dat,par.pulseTiming, par.pulseSeqDesc)
            winStart=tp.t0+tp.tau+2*tp.width
            vals=magData.loc[winStart:winStart+winLength ].mean()-v0
            for key in vals.keys():
                resD[key].append(vals[key])

        #Now create fitting functions
        funcD={}
        for key, y in resD.items():
            y=np.array(y)
            y=MT.smooth(y, int(Nsteps/15) )
            amp=(y.max()-y.min())/2
            zero=(y.max()+y.min())/2
            yNorm=y.copy()
            yNorm-=zero
            yNorm/=amp
            minI=yNorm.argmin()
            maxI=yNorm.argmax()
            theta=np.arcsin(yNorm)
            theta[:maxI]=np.pi-theta[:maxI]
            theta[minI:]=-np.pi-theta[minI:]
            fitF=interpolate.interp1d(theta[abs(vA)<=0.15].mean()-theta, vA, fill_value='extrapolate')
            #pdb.set_trace()
            funcD[key]=fitF

    xFuncs=[funcD[key] for key in funcD if eval(key)[0].find('X')>=0]
    yFuncs=[funcD[key] for key in funcD if eval(key)[0].find('Y')>=0]
    fitFxAve=lambda theta: np.mean([func(theta) for func in xFuncs], axis=0)
    fitFyAve=lambda theta: np.mean([func(theta) for func in yFuncs], axis=0)
    #fitFzAve=lambda theta: np.mean([funcD[key](theta) for key in funcD if ])
        


    seqDesc=[('Y', 'X'), ('mX', 'Y')]#, ('Y', 'X'), ('mY', 'X')]#,('X', 'Y'), ('X', 'mY'), ('mX', 'Y'), ('mX', 'mY')]

    if 1:
        par.pulseSeqDesc=seqDesc
        l=[]
        resDz=defaultdict(list)
        for v in vA:
            tp=par.pulseTiming
            #pdb.set_trace()
            par.pulseAmps.pi2X=fitFxAve(np.pi/2)
            par.pulseAmps.pi2Y=fitFyAve(np.pi/2)
            par.pulseAmps.bonusZHeight=(0,0,v)
            c._setupExperiment(par)
            #setupWithChanged(**{'pi2X':v, 'pi2Y':v})
            t,dat=c.locAcquire(Naves)
            dat=dat.mean(axis=0)
            magData,pulseData,depolData=proc.splitRawSubsecs(t, dat,par.pulseTiming, par.pulseSeqDesc)
            #l.append(magData)
            winStart=tp.t0+tp.tau+2*tp.width
            vals=magData.loc[winStart:winStart+winLength ].mean()-v0
            for key in vals.keys():
                resDz[key].append(vals[key])

        funcDz={}
        for key, y in resDz.items():
            y=np.array(y)
            y=MT.smooth(y, int(Nsteps/15) )
            amp=(y.max()-y.min())/2
            zero=(y.max()+y.min())/2
            yNorm=y.copy()
            yNorm-=zero
            yNorm/=amp
            minI=yNorm.argmin()
            maxI=yNorm.argmax()
            theta=np.arcsin(yNorm)
            theta[:maxI]=np.pi-theta[:maxI]
            theta[minI:]=-np.pi-theta[minI:]
            fitF=interpolate.interp1d(theta[abs(vA)<0.15].mean()-theta, vA, fill_value='extrapolate')
            #pdb.set_trace()
            funcDz[key]=fitF
    #should think about consolidating here...
    fitFzAve=lambda theta: np.mean([func(theta) for func in funcDz.values()], axis=0)
    global ampConvD
    ampConvD={'X': fitFxAve,
              'Y': fitFyAve,
              'Z': fitFzAve}
    glbP.saveCoilCal(ampConvD)

    #setupExperiment()
    #Now do z:
    return xFuncs, yFuncs, ampConvD#resD, funcD, resDz, funcDz, fitFxAve, fitFyAve, fitFzAve#,resDz# par, funcD #convDict




if __name__=="__main__":
    zero1()
    zero2()
    zero1()
    zero2()
    zero3()
    zero4()
