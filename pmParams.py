""" Global Parameters """
#import pickle
import dill as pickle
from copy import deepcopy
import os

class GlbParams(object):
    filename=None
    curParFilename=None
    def __init__(self,  curParFilename="expParamsLast.pkl", curCalFilename="calLast.pkl", curCoilCalFilename='coilCalLast.pkl', initialParams=None):
        self.curParFilename=curParFilename
        self.curCalFilename=curCalFilename
        self.curCoilCalFilename=curCoilCalFilename
        if initialParams is not None:
            if isinstance(initialParams, str):
                self.saveParams(self.loadParams(initialParams))
            else:
                self.saveParams(initialParams)

    def loadParams(self, filename=None):
        if filename is None:
            filename=self.curParFilename
        return pickle.load(open(filename, 'br'))
        #epInit.pulseAmps.bonusZHeight=0

    def saveParams(self, params, filename=None):
        if filename is None:
            filename=self.curParFilename
        pickle.dump(params,open(filename, 'bw'))
        #epInit=pickle.load(open(pklFilename, 'br'))

    @property
    def p(self):
        return self.loadParams()

    @p.setter
    def p(self, newParams):
        self.saveParams(newParams)

    def change(self, bSyncPulse1Amps=True, **kwargs):
        self.p=self.getChanged(bSyncPulse1Amps=True, **kwargs)

    def tweak(self, tweakDict, frac=1.0, bSyncPulse1Amps=True):
        #newD={key:val*frac for key,val in tweakDict.items()}
        self.p=self.getTweaked(frac=frac, bSyncPulse1Amps=bSyncPulse1Amps, **tweakDict)

    def getChanged(self, bSyncPulse1Amps=True, **kwargs):
        newPars=deepcopy(self.p)
        for key in kwargs.keys():
            if key in newPars:
                newPars[key]=kwargs[key]
            else:
                for subParKey in ['fields', 'pulseAmps', 'pulseTiming', 'lasParams']:
                    if key in newPars[subParKey]:
                        newPars[subParKey][key]=kwargs[key]
                        if key in ['piX', 'mPiX', 'piY', 'mPiY'] and bSyncPulse1Amps:
                            keyB=key+'_B'
                            newPars[subParKey][keyB]=kwargs[key]
                        break
                else:
                    print("didn't find {}".format(key))
        #self.p=newPars
        return newPars

    def getTweaked(self, frac=1, bSyncPulse1Amps=True, **kwargs):
        newPars=deepcopy(self.p)
        for key in kwargs.keys():
            for subParKey in ['fields', 'pulseAmps', 'pulseTiming', 'lasParams']:
                if key in newPars[subParKey]:
                    newPars[subParKey][key]+=frac*kwargs[key]
                    if key in ['piX', 'mPiX', 'piY', 'mPiY'] and bSyncPulse1Amps:
                        keyB=key+'_B'
                        newPars[subParKey][keyB]+=frac*kwargs[key]
                    break
            else:
                print("didn't find {}".format(key))
        #self.p=newPars
        return newPars

    
    lastMTimeCoilCal=0;
    def loadCoilCal(self):
        coilCal=pickle.load(open(self.curCoilCalFilename, 'br'))
        self.lastMTimeCoilCal=os.path.getmtime(self.curCoilCalFilename)
        return coilCal

    def saveCoilCal(self,coilCalDict, filename=None):
        if filename is None:
            filename=self.curCoilCalFilename
        pickle.dump(coilCalDict,open(filename, 'bw'), protocol=-1)

    def isCoilCalUpdated(self):
        curMTime=os.path.getmtime(self.curCoilCalFilename)
        if curMTime > self.lastMTimeCoilCal:
            return True
        else:
            return False

    lastMTime=0;
    def loadCal(self):
        cal=pickle.load(open(self.curCalFilename, 'br'))
        self.lastMTime=os.path.getmtime(self.curCalFilename)
        return cal
        #epInit.pulseAmps.bonusZHeight=0
    def saveCal(self,t,grad, filename=None):
        if filename is None:
            filename=self.curCalFilename
        pickle.dump({'t': t, 'grad':grad, 'pars':self.p},open(filename, 'bw'), protocol=-1)
    def isCalUpdated(self):
        curMTime=os.path.getmtime(self.curCalFilename)
        if curMTime > self.lastMTime:
            return True
        else:
            return False
fullSeq=[('X', 'Y'), ('mX', 'Y'), ('Y', 'X'), ('mY', 'X'),]
fullSeqLong=fullSeq+[(el[0], 's'+el[1]) for el in fullSeq];
doubleFullSeq=[('X', 'Y'), ('mX', 'Y'), ('Y', 'X'), ('mY', 'X'),
        ('X_B', 'mY'), ('mX_B', 'mY'),  ('Y_B', 'mX'), ('mY_B', 'mX'),
        ]
doubleFullSeq2=[('X', 'Y'), ('mX', 'Y'), ('Y', 'X'), ('mY', 'X'),
        ('mX_B', 'mY'), ('X_B', 'mY'),  ('mY_B', 'mX'), ('Y_B', 'mX'),
        ]
doubleFullSeqLong=doubleFullSeq+[(el[0], 's'+el[1]) for el in doubleFullSeq]

from munch import Munch
vSclFact=10
p0=Munch(
    pulseTiming=Munch(
        t0=40e-6,
        width=2.5e-6,
        pi2RelWidth=1.,
        tau=1400e-6,
        tLong=1700e-6,
        tShort=150e-6,
        tauShort=60e-6,
        depolLength=50e-6,
    ),
    pulseSeqDesc=[ ('X', 'Y'), ('mX', 'Y'),
            ('Y', 'X'), ('mY', 'X'),
            ('X', 'mY'), ('mX', 'mY'),
            ('Y', 'mX'), ('mY', 'mX'),
    ],
    pulseAmps=Munch(
        piX=3.75/vSclFact, 
        piY=3.26/vSclFact, 
        mPiX=-3.75/vSclFact, 
        mPiY=-3.26/vSclFact, 
        piXOff=0.,
        mPiXOff=0.0,
        piYOff=0.0,
        mPiYOff=0.0, 
        pi2XOff=.0,
        pi2YOff=0.0,  
        bonusZHeight=(0,0,0),
        reversePulse=True,
    ),
    fields=Munch(
        Vx=1*-0.06,
        Vy=1*-0.034,
        Vz=1*-0.006,
    ),
    totalTime=1./60, # Period of mains AC
)

if 0:
    pklFilename='expParamsLast.pkl'
    def loadParams():
        global ep
        ep=pickle.load(open(pklFilename, 'br'))
        #epInit.pulseAmps.bonusZHeight=0

    def saveParams():
        pickle.dump(ep,open(pklFilename, 'bw'))
        #epInit=pickle.load(open(pklFilename, 'br'))

    def adjustParams(oldPars, fitVals,frac=0.5):
        newPars=deepcopy(oldPars)
        for key in fitVals.keys():
            for subParKey in ['fields', 'pulseAmps', 'pulseTiming']:
                if key in newPars[subParKey]:
                    newPars[subParKey][key]-=fitVals[key]*frac
                    break
            else:
                print("didn't find {}".format(key))
        return newPars

calFile="calfile.pkl"

if 0:
    calDict=loadCalData()
    def loadCalData(calFileName=calFile):
        #calDict=dict(load(calFile))
        calDict=pickle.load(open(calFile, 'rb'))
        return calDict
    def saveCalData(t, processedCalD, calFile=calFile):
        f=open(calFile,'wb')
        pickle.dump(processedD, f, protocol=-1)
        #savez_compressed(calFile, processedCalD)


    from matplotlib import gridspec
    def showCalData(data):
        """Somehow display all the calibration data in a useful way
        """
        gs=gridspec.GridSpec(4,4)
        for gradData in dat.magDataGrad:
            for k,key in enumerate(gradData.keys()):
                ax=subplot(gs[k])
                ax.plot(gradData[key])
                ax.set_ylim([-0.2,0.2])

        for k,key in enumerate(dat.magDataRaw.keys()):
            ax=subplot(gs[k+8])
            ax.plot(dat.magDataRaw[key])
            ax.set_ylim([-0.2,0.2])


if __name__=="__main__":
    glbP=GlbParams(initialParams=p0)
