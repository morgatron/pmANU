"""Fitting of pulsed, vector magnetometry data

This is the ANU version. Let's try and simplify if possible. 

Not all routines in here are needed. The main ones, currently, are: 
doAllTraceProcessing-> to convert raw signal data into subsections, some of which are subtracted off each other
doAllGradProcessing -> do the same for grad data
fitSubbed -> start with raw data, and fit the subtracted components with the desired gradients
fitPlain -> start with raw data, and fit the raw components with the desired gradients


These methods are used to enable the overall 'zeroing' process, which should be described elseqhere, but since it's not, here it is:
1. Get approximate values for piX, piY (and minus versions) by assuming things are roughly orthogonal and looking at raw data (can be done by hand, or in principle by fitting with fitPlain too)
2. Fit to subtracted long waiting to subtracted segments with Vx, Vy, Vz, mPiX, mPiY. This should zero out the early part of each, at least the subtracted and ideally the non-subtracted too.
(repeat step 1 and step 2)
3. Fit simultaneously to subtracted long and short waiting time segments, to determine Vx, Vy, Vz, mPiX, mPiY, piXOff, and piYOff. This should be able to zero out the subtracted signals completely. Could also just do Vz, piXOff, and piYOff in this step
4. To zero out the non-subtracted signals, do a fit to the non-subtracted versions with pi2XOff, pi2YOff
"""


import pdb
import MT
import pandas as pd
import numpy as np
from munch import Munch
from statsmodels import api as sm
import pdb
import pmParams
import pylab as pl
from MT import pruneRagged
from copy import copy

glbP=pmParams.GlbParams()

fitRegions=Munch(
      mag=[200e-6,np.inf],
      pulse=[40e-6,np.inf]
        )


def splitRawSubsecs(t, raw, timePars, seqDesc):
    """
    Output data frames (from a long time series with many frames in it). Each frame is made up 
    of several traces, according to @seqDesc
    """
    #NLongTraces=len(seqDesc)
    #tDeltas=[timePars.tShort, timePars.tLong]*NLongTraces + [timePars.depolLength]
    tDeltas=[timePars.tShort if seq[1].startswith('s') else timePars.tLong for seq in seqDesc ] + [timePars.depolLength]

    tBegins=np.cumsum([0]+tDeltas[:-1])
    tEnds=tBegins+tDeltas
    tEnds[tEnds>t.max()]=t.max()
    #indxL=MT.sliceSorted(t,csum-tBegins, csum)
    indxL=MT.sliceSorted(t,tBegins, tEnds)
    def trim(lst, warnThresh=0.2): #make all the sub-sections the same length
        if not len(lst):
            return lst
        lens=[l.size for l in lst]
        newN=np.min(lens)
        biggestN=np.max(lens)
        if 1-float(newN)/biggestN> warnThresh:
            print("doing a lot of trimming! Maybe a mistake...")
            print(lens)
            raise ValueError("doing a lot of trimming! Maybe a mistake...");
        lst=[l[:newN] for l in lst]
        return np.array(lst)
    #pdb.set_trace()
    #pulseInds=trim(indxL[:-1:2])
    shortPeriodMask=np.where([seq[1].startswith('s') for seq in seqDesc],True, False)
    shortMagInds=trim(indxL[:-1][shortPeriodMask])
    longMagInds=trim(indxL[:-1][~shortPeriodMask])
    depolInds=np.array(indxL[-1])

    
    #tSplt, datSplt=pSerf.splt(t,raw, tSplt=timePars.tLong+timePars.tShort, Nsplts=NLongTraces, keepFrac=0.95)

    #NPtsShort=t.searchsorted(timePars.tShort)
    longLabels=[str(desc) for desc in seqDesc if not desc[1].startswith('s')]
    #@longLabels=labels[~shortPeriodMask]
    shortLabels=[str((desc[0], desc[1].strip('s'))) for desc in seqDesc if desc[1].startswith('s')]

    #raw[magInds].T
    #pdb.set_trace()
    #print('shortMagInds'.format(shortMagInds))
    if not shortMagInds.size:
        shortMagData=pd.DataFrame()
    else:
        shortMagData=pd.DataFrame(raw[shortMagInds].T, index=t[:shortMagInds.shape[-1]], columns=shortLabels)
    longMagData=pd.DataFrame(raw[longMagInds].T, index=t[:longMagInds.shape[-1]], columns=longLabels)
    #pulseData=pd.DataFrame(raw[pulseInds].T, index=t[:pulseInds.shape[1]], columns=labels)
    depolData=pd.DataFrame(raw[depolInds], index=t[:depolInds.shape[0]], columns=['depol'])
    #pulseData=pd.DataFrame(datSplt[:,:NPtsShort].T, index=t[:NPtsShort], columns=labels)
    #magData=pd.DataFrame(datSplt[:,NPtsShort:].T, index=tSplt[NPtsShort:], columns=labels)
    #pulseData=pd.DataFrame(datSplt[:,:NPtsShort].T, index=t[:NPtsShort], columns=labels)
   
    #Split into segments
    #Sort out label order
    #Make labeled data frames

    #Should be data frames?
    return longMagData, shortMagData, depolData

def subtractSubsecs(df, tLim=None, toSubL=['y', 'ym', 'x', 'xm']):
    if df.shape[0]==0:
        return {}
    toSubD={
        'y':["('X', 'Y')", "('mX', 'Y')"],
        'ym':["('X_B', 'mY')","('mX_B', 'mY')"],
        'x':["('Y', 'X')","('mY', 'X')"],
        'xm':["('Y_B', 'mX')","('mY_B', 'mX')"],
        'y2':["('X', 'Y')", "('X_B', 'mY')"],
        'ym2':["('mX', 'Y')","('mX_B', 'mY')"],
        'x2':["('Y', 'X')","('Y_B', 'mX')"],
        'xm2':["('mY', 'X')","('mY_B', 'mX')"],

            }
    out={}
    for key in toSubL:
        val=toSubD[key]
        try:
            out[key]=df[val[0]] - df[val[1]]
        except KeyError as e:
            #print(e.args[0])
            pass
    if 0:
        out={
            #Z measuring combinations
            #'zmY':df["('X', 'Y')"]-df["('X', 'mY')"],
            #'zY':df["('mX', 'Y')"]-df["('mX', 'mY')"],
            #'zX':df["('Y', 'X')"]-df["('Y', 'mX')"],
            #'zmX':df["('mY', 'X')"]-df["('mY', 'mX')"],

            'y':df["('X', 'Y')"]-df["('mX', 'Y')"],
            'ym':df["('X', 'mY')"]-df["('mX', 'mY')"],
            'x':df["('Y', 'X')"]-df["('mY', 'X')"],
            'xm':df["('Y', 'mX')"]-df["('mY', 'mX')"],
        }
    return pd.DataFrame(out)

def doAllTraceProcessing(t, raw, expPars, toSubL=['x', 'xm', 'y', 'ym']):
    magDataRaw, pulseDataRaw,depolDataRaw=splitRawSubsecs(t,raw, expPars.pulseTiming, expPars.pulseSeqDesc)
    subPulseData=subtractSubsecs(pulseDataRaw, toSubL=toSubL)
    subMagData=subtractSubsecs(magDataRaw, toSubL=toSubL)
    lst=[magDataRaw, pulseDataRaw, depolDataRaw, subMagData, subPulseData]
    nms=["magDataRaw", 'pulseDataRaw', 'depolDataRaw', "subMagData", "subPulseData" ]
    processedD=Munch({nm:val for nm,val in zip(nms,lst)})
    return processedD

def doAllGradProcessing(t, gradRawD, expPars, toSubL=['x', 'xm', 'y', 'ym']):
    bigLst=[splitRawSubsecs(t, gradRawD[key],expPars.pulseTiming, expPars.pulseSeqDesc) for key in gradRawD]
    magDataGrad, pulseDataGrad, depolGrad= zip(*bigLst)
    magDataGrad= pd.Panel( dict(zip(gradRawD.keys(), magDataGrad))).to_frame()  
    pulseDataGrad= pd.Panel( dict(zip(gradRawD.keys(), pulseDataGrad)) ).to_frame()
    subMagGrad=subtractSubsecs(magDataGrad.stack().unstack("minor"), toSubL=toSubL).stack().unstack(1)
    if pulseDataGrad.shape[0]!=0:
        subPulseGrad=subtractSubsecs(pulseDataGrad.stack().unstack("minor"), toSubL=toSubL).stack().unstack(1)
        nms=["magDataGrad", "pulseDataGrad", "depolGrad", "subMagGrad", "subPulseGrad" ]
        lst=[ magDataGrad, pulseDataGrad, depolGrad, subMagGrad, subPulseGrad ]
    else:
        nms=["magDataGrad",  "depolGrad", "subMagGrad", ]
        lst=[ magDataGrad, depolGrad, subMagGrad ]

    processedD=Munch({nm:val for nm,val in zip(nms,lst)})
    return processedD





    if 0:
        outputD={};
        for key,val in gradRawD.items(): 
            processedD=doAllTraceProcessing(t, val, expPars)
        #for grad stuff
        if 0:
            magDataGrad= pd.Panel( dict(zip(gradRawD.keys(), magDataGrad))).to_frame()  
            pulseDataGrad= pd.Panel( dict(zip(gradRawD.keys(), pulseDataGrad)) ).to_frame()
            subMagGrad=getSubtracted(magDataGrad.stack().unstack("minor")).stack().unstack(1)
            subPulseGrad=getSubtracted(pulseDataGrad.stack().unstack("minor")).stack().unstack(1)
            subPulseData=getSubtracted(pulseDataRaw)
            subMagData=getSubtracted(magDataRaw)

            nms=["magDataGrad", "pulseDataGrad", "subMagGrad", "subPulseGrad", "depolDataRaw", "depolGrad"]
            lst=[magDataRaw, pulseDataRaw, subMagData, subPulseData, 
                    magDataGrad, pulseDataGrad, subMagGrad, subPulseGrad, depolDataRaw, depolGrad]
            processedD=Munch({nm:val for nm,val in zip(nms,lst)})
            return processedD


#TODO
def fitSimp(signal, exogD, addDC=False, bReturnObjs=False, bPlot=False):
    """Fit a list of signals to a linear combination of the curves named in exogD. This is mostly just a simple wrapper around statasmodles.WLS- but with less steps, and it could grow to include some fancier weighting stuff.

    """
    if not type(exogD) == pd.DataFrame:
        try:#if processedGradient is a dictionary
            paramNames=list(exogD.keys())
            paramNames.sort()
            exogD=[exogD[key] for key in paramNames]
        except AttributeError:
            paramNames=["Vx, Vy, Vz"]
            print("Gradient was not a dict")
            

        #Do linear fit of @sgGrad to the @sgDat
        #endog=np.hstack([sgZ, sgY])
        #exog=np.hstack([sgDZ, sgDY]).T
        
        exog=np.vstack(exogD).T
    else:
        exog=exogD
    if addDC:
        exog=sm.add_constant(exog)
    #Whatever the form of the return is
    resF=(lambda mod: mod.fit()) if bReturnObjs else (lambda mod: mod.fit().params)

    sig=signal[0]
    endog=sig.stack()
    model=sm.OLS(endog, exog, missing='drop')
    resL=[resF(model)]
    if len(signal)>1:
        for sig in signal[1:]:
            model.endog[:]=sig.stack()
            resL.append(resF(model))
    return resL
    #result=model.fit()
        #covL.append(result.cov())
        #resultL.append(result.params)
    #resultA=np.array(resultL)
    #covA=np.array(covL)
    #return result, Bunch(**locals())
    #paramD={key:vals for key, vals in zip(paramNames, resultA.T)}
    #covD={key:vals for key, vals in zip(paramNames, covA.T)}
    if bReturnObjs:
        return result#paramD, covD#, Bunch(**locals())
    else:
        return result.params

def pruneCrap(df, startDumpT=100e-6, endDumpT=50e-6):
    return df.loc[startDumpT:endDumpT]

#def doSimpleMagnetometry(t, raw, expPars=None,gradD=None, processedGrad=None, startDumpT=200e-6, endDumpT=500e-6, toFit=['Vx', 'Vy', 'Vz'], bSplit8=False):

def fitSubbed(t, raw, expPars=None,gradD=None, processedGrad=None, startDumpT=200e-6, endDumpT=500e-6, toFit=['Vx', 'Vy', 'Vz'], bSplit8=False, bRetProcessed=False, bFitShort=True, toSubL=['x', 'xm', 'y', 'ym'], addDC=False):
    """Should actually do 2 lots of magnetometry, one on the first section and one on the second, and just append them all together"""

    raw=np.array(raw)
    #raw=pruneRagged(raw, maxPts=20, axis=-1)
    #print("new shape: {}".format(raw.shape))
    if expPars is None:
        expPars= glbP.p
    if processedGrad is None:
        if gradD is None:
            cal= glbP.loadCal()
            gradD=cal['grad']
        processedGrad=doAllGradProcessing(t, gradD, expPars, toSubL=toSubL)

    ignoreTime=expPars.pulseTiming.t0+startDumpT
    finalTime=expPars.pulseTiming.t0+expPars.pulseTiming.tLong-endDumpT
    finalTimeShort=expPars.pulseTiming.t0+expPars.pulseTiming.tShort-endDumpT

    if bFitShort:
        exogD=pd.concat([processedGrad.subMagGrad[toFit].loc[ignoreTime:finalTime], processedGrad.subPulseGrad[toFit].loc[ignoreTime:finalTimeShort]], keys=['l', 's'])
    else:
        if bSplit8:
            toFit1=copy(toFit)
            toFit2=copy(toFit)
            plsFitL=['piX', 'piY', 'mPiX', 'mPiY']
            plsFitLB=[st+'_B' for st in plsFitL]
            for key in toFit:
                if key in plsFitLB:
                    toFit1.remove(key)
                elif key in plsFitL:
                    toFit2.remove(key)
            print('toFit: {}, {}'.format(toFit1, toFit2))

            exogD1=processedGrad.subMagGrad[toFit1].loc[(slice(ignoreTime,finalTime)),:]
            exogD2=processedGrad.subMagGrad[toFit2].loc[(slice(ignoreTime,finalTime)),:]
        else:
            exogD=processedGrad.subMagGrad[toFit].loc[ignoreTime:finalTime]

    outL=[]
    if raw.ndim>1:
        if bSplit8: #This branch will be broken currently
            outL1=[]
            outL2=[]
            procDL=[doAllTraceProcessing(t, dat, expPars, toSubL=toSubL) for dat in raw]
            labs=procDL[0].subMagData.keys()
            labs1=[l for l in labs if not l.endswith('m')]
            labs2=[l for l in labs if l.endswith('m')]
            subbedSigL1=[procD.subMagData.loc[ignoreTime:finalTime, labs1] for procD in procDL ]
            subbedSigL2=[procD.subMagData.loc[ignoreTime:finalTime, labs2] for procD in procDL ]
            outL1=fitSimp(subbedSigL1, exogD1.loc[(slice(None), labs1),:], addDC=addDC) 
            outL1=[[fitP[key] for key in toFit1] for fitP in outL1]
            outL2=fitSimp(subbedSigL2, exogD2.loc[(slice(None), labs2),:], addDC=addDC) 
            outL2=[[fitP[key] for key in toFit2] for fitP in outL2]

            outA=(np.array(outL1), np.array(outL2))

            #print("labs1: {}".format(labs1))
            #print("labs2: {}".format(labs2))
            #Weird double slicing seems to be required for no good reason
            if 0:
                p1=fitSimp([sigProc.subMagData.loc[ignoreTime:finalTime, labs1 ]], exogD1.loc[(slice(ignoreTime,finalTime)),:].loc[(slice(None), labs1),:] )
                outL1.append(p1)
                if labs2:
                    #p2=fitSimp(sigProc.subMagData.loc[ignoreTime:, labs2 ], exogD.loc[ignoreTime:finalTime])
                    p2=fitSimp([sigProc.subMagData.loc[ignoreTime:finalTime, labs2 ]], exogD2.loc[(slice(ignoreTime,finalTime)),:].loc[(slice(None), labs2),:] )
                    #p2=fitSimp(sigProc.subMagData.loc[ignoreTime:finalTime, labs2 ], exogD.loc[(slice(ignoreTime,finalTime), labs2),:] )
                    outL2.append(p2)

                outL=[outL1, outL2] 

        else:
            procDL=[doAllTraceProcessing(t, dat, expPars, toSubL=toSubL) for dat in raw]
            #subbedMagL=[dat.subMagData.loc[ignoreTime:finalTime] for dat in processedL ]
            if bFitShort:
                subbedSigL=[pd.concat([procD.subMagData.loc[ignoreTime:finalTime], procD.subPulseData.loc[ignoreTime:finalTimeShort]], keys=['l', 's']) for procD in procDL]
            else:
                subbedSigL=[procD.subMagData.loc[ignoreTime:finalTime] for procD in procDL ]

            outL=fitSimp(subbedSigL, exogD, addDC=addDC) 
            outL=[[fitP[key] for key in toFit] for fitP in outL]
            outA=np.array(outL)
    return outA


def fitPlain(t, raw, expPars=None,gradD=None, processedGrad=None, startDumpT=120e-6, endDumpT=60e-6, toFit=['Vx', 'Vy', 'Vz', 'piX', 'mPiX','piY', 'mPiY'], bFitShort=False, bSplit8=False, bAddDC=True):
    """Can probably use doFit in place of this"""
    gradD={key:gradD[key] for key in toFit}
            
    processedGrad=doAllGradProcessing(t, gradD, expPars)
    processedSig=doAllTraceProcessing(t,raw, expPars)
    tLong=expPars.pulseTiming.tLong
    tShort=expPars.pulseTiming.tShort
    if bFitShort:
        sig=pd.concat([processedSig.magDataRaw.loc[startDumpT:tLong-endDumpT], processedSig.pulseDataRaw.loc[startDumpT:tShort-endDumpT]], keys=['l', 's'])
    #sig2=pd.concat([processedSig.magDataRaw, processedSig.pulseDataRaw])
        grad=pd.concat([processedGrad.magDataGrad.loc[startDumpT:tLong-endDumpT], processedGrad.pulseDataGrad.loc[startDumpT:tShort-endDumpT]], keys=['l', 's'])
    else:
        sig=processedSig.magDataRaw.loc[startDumpT:tLong-endDumpT]
        grad=processedGrad.magDataGrad.loc[startDumpT:tLong-endDumpT]
    #sig=sig.loc[startDumpT:-endDumpT]
    #grad=grad.loc[startDumpT:-endDumpT]
    #return sig,sig2, grad
    fitP=fitSimp([sig], grad, addDC=bAddDC, bReturnObjs=False)[0]

    return fitP#np.array(outL)#, sig, grad



def showProcessing(t, raw, expPars, ylims=[-0.2, 0.2]):
    sigProcessedD=doAllTraceProcessing(t,raw, expPars)
    #Raw:
    pl.figure( )
    pl.plot(t,raw)
    pl.title('Raw')

    #Processed:
    for subGroup in sigProcessedD:
        subFrame=sigProcessedD[subGroup]
        if subFrame is not None:
            cols=subFrame.columns 
            Ncols=len(cols)
            pl.figure(subGroup, figsize=(8,10))
            pl.suptitle(subGroup)
            for k,col in enumerate(cols):
                pl.subplot(Ncols, 1, k+1)
                pl.plot(subFrame[col])
                pl.ylim(ylims)
                pl.title(col)
            pl.tight_layout()

def showGrad(t,gradRawD, ylims=[-1,1]):
    parNames=[ key  for key in gradRawD.keys() if key in ['Vx', 'Vy', 'Vz', 'piX', 'piY']]
    
    Npars=len(parNames)
   
    #Unprocessed
    for k,parName in enumerate(parNames):
        if k==0:
            ax=pl.subplot(Npars,1, 1)
        else:
            ax=pl.subplot(Npars,1, k+1,sharex=ax)
        ax.plot(t, gradRawD[parName])
        ax.set_title(parName)
        ax.set_ylim(ylims)
    pl.tight_layout()






if __name__=="__main__":
    import numpy as np
    class DockAreaManager(object):
        win=None
        area=None
        dockD=None

        def __init__(self, name='Dock window'):
            self.name=name
            area=DockArea()
            win = pg.QtGui.QMainWindow()
            win.setCentralWidget(area)
            win.resize(400,300)
            win.setWindowTitle(name)
            self.area=area
            self.dockD={}
            self.win=win

            #Save area
            saveDock=Dock("saveArea", size=(10,10))
            w1 = pg.LayoutWidget()
            label = pg.QtGui.QLabel("""Save/restore state""")
            saveBtn = pg.QtGui.QPushButton('Save state')
            restoreBtn = pg.QtGui.QPushButton('Restore state')
            restoreBtn.setEnabled(False)
            w1.addWidget(label, row=0, col=0)
            w1.addWidget(saveBtn, row=1, col=0)
            w1.addWidget(restoreBtn, row=2, col=0)
            saveDock.addWidget(w1)
            saveBtn.clicked.connect(self.save)
            restoreBtn.clicked.connect(self.load)
            self.saveBtn=saveBtn
            self.restoreBtn=restoreBtn
            self.area.addDock(saveDock)
            self.win.show()

        def save(self):
            self.state = self.area.saveState()
            pickle.dump(self.state, open("dockManager_{}_{}.pkl".format(__name__, self.name), 'wb') )
            self.restoreBtn.setEnabled(True)
        def load(self):
            try:
                if self.state is None:
                    state=pickle.load(open("dockManager_{}_{}.pkl".format(__name__, self.name), 'rb') )
                    state={k:v for k,v in state if k in self.dockD.keys()} 
                self.area.restoreState(self.state)
            except Exception as e:
                print(e.args[0])


        def addDockPlot(self, name, x=None,y=None, title=None, **kwargs):
            dock=Dock(name, size=(200, 200))
            if title is None:
                title=name
            w = pg.PlotWidget(title=title)
            if y is None and x is not None:
                y=x; x=None
            if x is None:
                x=arange(y.size)
            w.plot(x, y)
            dock.addWidget(w);
            self.dockD[name]=dock
            self.area.addDock(dock)
        def getPlotItem(self, name):
            return self.dockD[name].findChild(pg.PlotWidget).plotItem

        def setData(self, name, x,y=None):
            if y is None:
                y=x; x=np.arange(y.size)
            self.getPlotItem(name).curves[0].setData(x,y)
        def updateFromDict(self, D):
            """ Update/create plots from the dictionary D containing curves to be plotted
            """
            for key, val in D.items():
                if len(val)>2:
                    val=(val,)
                if key in self.dockD:
                    self.setData(key, *val)
                else:
                    self.addDockPlot(key, *val)

    class Graphy(object):
        import pmAcquire as acq
        import pyqtgraph as pg
        gwRaw=None
        gwSub=None
        def __init__(self):
            self.acq.subscribe(b'raw')
            self.gwRaw=DockAreaManager(name="Raw")
            self.gwRaw2=DockAreaManager(name="Raw2")
            self.gwSub=DockAreaManager(name="Subtracted")
            self.gwSub2=DockAreaManager(name="Subtracted2")
            self.gwGrad=DockAreaManager(name="Gradient")
            if 0:
                rawPltL=[]
                for col in range(3):
                    for row in range(3):
                        rawPltL.append(gwRaw.addPlot(col=col, row=row))
                self.rawPltL=rawPltL
                subPltL=[]
                for col in range(2):
                    for row in range(2):
                        subPltL.append(gwRaw.addPlot(col=col, row=row))
                self.gwRaw=gwRaw
                self.gwSub=gwSub

        def updateGradProc(self):
            if glbP.isCalUpdated():
                cal=glbP.loadCal()
                self.gradD=cal['grad']
                t=cal['t']
                self.gradProcessedD=proc.doAllGradProcessing(t, self.gradD, cal['pars'])


        def updateGraphs(self):
            chk=self.acq.checkForPublished()
            grad=glbP.loadCal()['grad']
            self.grad=grad
            if chk is not None:
                topic,D=chk
                tL=D['t']
                raw=np.vstack(D['data'])
                dt=D['dt']

                raw=raw[-1]#.mean(axis=0)
                t=np.arange(raw.size)*dt
                #self.t, self.raw= (t,raw)
                sigProcessedD=doAllTraceProcessing(t, raw, glbP.p)
                self.sigProcD=sigProcessedD
                self.gwSub.updateFromDict(sigProcessedD.subMagData)
                self.gwSub2.updateFromDict(sigProcessedD.subPulseData)
                self.gwRaw.updateFromDict(sigProcessedD.magDataRaw)
                self.gwRaw2.updateFromDict(sigProcessedD.pulseDataRaw)
                self.gwGrad.updateFromDict(grad)
    from pyqtgraph.dockarea import *
    import pyqtgraph as pg
    from numpy import *
    from time import sleep
    #app=pg.mkQApp()
    app = pg.QtGui.QApplication([])
    timer = pg.QtCore.QTimer()
    gr=Graphy()
    timer.timeout.connect(gr.updateGraphs)
    timer.start(300)
    while 0:
        gr.updateGraphs()
        sleep(0.5)
    #dm=DockAreaManager()
    #dm.addDockPlot("one", x, x**2)
    #dm.addDockPlot("two", x, sin(x**2))
    #dm.addDockPlot("three", x, x**2*sin(x**2))
    #dm.addDockPlot("four", x, x**2*sin(x**2)-x**3)


