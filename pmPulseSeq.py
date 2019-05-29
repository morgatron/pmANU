""" Plan: this should hold relevant functions/class for describing a pulsed-magnetometer pulse sequence.

The required outputs of this are:
* Three time series of signals between -1 and +1 representing the X, Y, Z voltages applied by the function generators.
* A dictionary of start and stop times showing where each time section starts and stops, 
* Bonus: the splitDataIntoSubs function that takes t,sig data and splits it into a dictionary with the releavant sections

This has significant overlap with the current pulse_controller module. 

The pulse-controller module is nice in that it allows an easy interface to make a small change, i.e. changing one pulse amplitude but not the rest. The pmSetup.py with a PulseMagController object (i.e. pretty much just the current PulseController but with the DAQ card included.

 Relevant height parameters:
piX, piY, mPiX, mPiY, 
piXOff, mPiXOff, piYOff, mPiYOff, 
pi2XOff,  pi2YOff, 



TODO: allow a different height first-pulse. This is necessary to allow measuring the gradient when a single parameter affects the pulse-height in 2 (or more) places. e.g. have piX, mPiX, and piX_B and mPiX_B
Have to keep in mind that changing the pulseSequence description will require changing the splitting/subtracting part too: likely the splitting will be ok, but will have to re-do which bits to subtract... and will have to redo the zeroing sequences.
So, let's have extra parameters piX_BD, mPiX_BD etc that give just the difference. Usually they're zero... but that won't work because of the conversions from volts to degrees (which this file knows nothing about)
So, could have completely different parameters, e.g. the piX_B, mPiX_B etc. The pulse sequence will be written differently. Can modify the parameter change method to always update both at the same time, unless a specific flag is sent. Sounds reasonable?
"""
import MT
import numpy as np
import pylab as pl

def generatePulseMagPattern(pulseSeqDesc, pulseTimeParams, 
        heightParams,
        sampleRate=2500000
    ):
    #paramNames=['piX', 'piY', 'mPiX', 'mPiY', 
    #'piXOff', 'mPiXOff', 'piYOff', 'mPiYOff', 
    #'pi2X', 'pi2Y', 'pi2XOff', 'pi2YOff'] 
    Nseq=len(pulseSeqDesc)
    hp=heightParams
    bonusZHeight=hp.bonusZHeight
    d1=dict( #First pulse
        X=(hp.piX, hp.piXOff,0),
        mX=(hp.mPiX, hp.mPiXOff,0),
        Y=(hp.piYOff, hp.piY, 0),
        mY=(hp.mPiYOff, hp.mPiY, 0.),
        X_B=(hp.piX_B, hp.piXOff,0),
        mX_B=(hp.mPiX_B, hp.mPiXOff,0),
        Y_B=(hp.piYOff, hp.piY_B, 0),
        mY_B=(hp.mPiYOff, hp.mPiY_B, 0.),
    )
    if 1:
        d2=dict( #Second pulse
            X=( hp.pi2X if hp.pi2X is not None else (hp.piX-hp.mPiX)/2., 0,hp.pi2XOff),
            mX=( -hp.pi2X if hp.pi2X is not None else -(hp.piX-hp.mPiX)/2.,0, -hp.pi2XOff),
            Y=( 0, hp.pi2Y if hp.pi2Y is not None else (hp.piY-hp.mPiY)/2., hp.pi2YOff),
            mY=(0, -hp.pi2Y if hp.pi2Y is not None else -(hp.piY-hp.mPiY)/2., -hp.pi2YOff),
    #y=log(exp(t*1e2) ) -log(log( exp(t*1e2) ))
        )
        d2copy=d2.copy()
        for k in d2copy.keys():
            d2['s'+k]=d2copy[k]
    print("Bonus z height: {}".format(bonusZHeight))
    d3=dict( #Bonuz Z-pulse
        X=tuple(bonusZHeight),
        mX=tuple(bonusZHeight),
        Y=tuple(bonusZHeight),
        mY=tuple(bonusZHeight),
        X_B=tuple(bonusZHeight),
        mX_B=tuple(bonusZHeight),
        Y_B=tuple(bonusZHeight),
        mY_B=tuple(bonusZHeight),
        )

    t0,tLong, tauLong, width,longWidth, pi2RelWidth, tShort, tauShort, depolLength, tInitDepol=pulseTimeParams.t0,pulseTimeParams.tLong,pulseTimeParams.tau, pulseTimeParams.width, pulseTimeParams.longWidth, pulseTimeParams.pi2RelWidth, pulseTimeParams.tShort, pulseTimeParams.tauShort, pulseTimeParams.depolLength, pulseTimeParams.tInitDepol
    
    startTimes=[]
    pulseWidths=[]

    initDepol = 0.0 if tInitDepol==0 else 1.0
    if 0:
        for subSeq in pulseSeqDesc:
            if type(subSeq) == tuple:
                # Do first pulse things
                delayType='' #just t0 usually
                pulseHeights.append(phD[subSeq[0]])
                pulseWidths.append(pwD[subSeq[0]])
                startTimes.append(pwD[delayType])

                #Second pulse things
                delayType,pulseType2=subSeq[-1].split()
                pulseHeights.append(phD[pulseType2])
                pulseWidths.append(pwD[pulseType2])
                startTimes.append(pwD[delayType])
                seqTotTimes.append(seqTD[delayType])
                
            else: #probably just depolX
                pulseHeights.append(phD[subSeq])
                pulseWidths.append(pwD[subSeq])
                seqTotTimes.append(seqTD[subSeq])
                
            for subSubSeq in pulseSeqDesc:
                bits=subSubSeq.split('_')
                pulseType=bits.pop()
                if len(bits)>0:
                    delayType=bits.pop()
                if pulseType=="depolX":
                    pulseHeight=phD['depolX']
                    pulseWidth=pwD['depol']
                startTime=t0 + pdD[delayType]
    if tShort >0 and 0:
        pulseHeights=[ (d1[plsPair[0]], d2[plsPair[1]], 
                        [-v for v in d1[plsPair[0]]], #undoing pulse
                        d3[plsPair[0]], #bonus z-pulse
                        d2[plsPair[1]],
                        [heightParams.reverseHeight*v for v in d1[plsPair[0]]], #undoing pulse
                        )  for plsPair in pulseSeqDesc]
    else:
        pulseHeights=[  (   d1[plsPair[0]], 
                            [initDepol*v for v in d2[plsPair[1]]],
                            d3[plsPair[0]], #bonus z-pulse
                            [1*v for v in d2[plsPair[1]]],
                            [heightParams.reverseHeight*v for v in d1[plsPair[0]]] #undoing
                        ) for plsPair in pulseSeqDesc]
    #import pdb
    #pdb.set_trace() 

    #Replace with something similar to the pulseHeights one
    tOffs=0
    for k in range(Nseq):
        #Short trace-------------
        bShortTrace=pulseSeqDesc[k][-1].startswith('s')
        thisTau= tauShort if bShortTrace else tauLong
        thisTTotal=tShort if bShortTrace else tLong

        #tOffs=(tLong+tShort)*k
        if tShort>0 and 0:
            startTimes.extend([t0+tOffs, t0+tauShort+tOffs])
            pulseWidths.extend([width, width])
            tOffs+=tShort

        #Long trace-------------
        #Could add in bonus Bz pulse here
        startTimes.extend([t0+tOffs, 
                        t0+tOffs+tInitDepol,
                        t0+tOffs+thisTau-width*2, #Bz pulse
                        t0+tOffs+thisTau,
                        tOffs+thisTTotal-width-40e-6, #revert pulse
                        ])
        pulseWidths.extend([width, 
                longWidth, #initDepol
                1*width, #Bonus Z
                pi2RelWidth*(width if tInitDepol==0 else longWidth),
                width
                ])
        tOffs += thisTTotal


    pulseHeights=np.array(pulseHeights).reshape(-1,3)
  
    #depolX
    #depolLength=300e-6
    if 0:
        k+=1
        tOffs=(tLong+tShort)*(k)
        pulseWidths.extend([depolLength/2]*2)
        startTimes.extend([tOffs+t0, tOffs+t0+depolLength/2])
        pulseHeights=np.vstack([pulseHeights, (.05,0,0),(-.05,0.,0) ])

    tLast=np.max(np.array(startTimes)+np.array(pulseWidths)) 
    print("latest thing in sequence: {}".format(tLast) )
    tSeqTotal=tLast+10e-6
    sampleRate=np.array(sampleRate)
    if sampleRate.size==1:
        sampleRate=np.ones(3)*sampleRate
    #Nsamples=32768*2
    tSeqTotal=tLast+10e-6
    Nsamples=(tSeqTotal*sampleRate).astype('i4')
    #tSeqTotal=int(tSeqTotal*sampleRate)/sampleRate
    #Nsamples=int(tSeqTotal*sampleRate)
    print("Nsamples: {}".format(Nsamples) )
    #sampleRate=Nsamples/tSeqTotal

    tx,y1=makePulseTrain(startTimes=startTimes, 
                            pulseWidths=pulseWidths, 
                            pulseHeights= pulseHeights[:,0],
                            sampleRate=sampleRate[0],
                            Nsamples=Nsamples[0],
                            )
    ty,y2=makePulseTrain(startTimes=startTimes, 
                            pulseWidths=pulseWidths, 
                            pulseHeights= pulseHeights[:,1],
                            sampleRate=sampleRate[1],
                            Nsamples=Nsamples[1],
                            )
    tz,y3=makePulseTrain(startTimes=startTimes, 
                            pulseWidths=pulseWidths, 
                            pulseHeights= pulseHeights[:,2],
                            sampleRate=sampleRate[2],
                            Nsamples=Nsamples[2],
                            )
    return tx, y1, y2,y3


##### TRIGGER PATTERNS:

from functools import partial
def makePulseTrain(startTimes, pulseWidths, pulseHeights, sampleRate, Nsamples, pulseFunc=partial(MT.tophat, bDigitizeFix=True)):
    Nsamples=int(Nsamples)
    Npulses=len(startTimes)
    if not hasattr(pulseWidths, "__iter__"):
        pulseWidths=[pulseWidths]*Npulses
    if not hasattr(pulseHeights, "__iter__"):
        pulseHeights=[pulseHeights]*Npulses
    if not (len(startTimes)==len(pulseWidths)==len(pulseHeights) ):
        raise ValueError("All sequences should be the same length, OR scalars")
    #t=np.linspace(0,tSeqTotal,tSeqTotal*sampleRate)*1.0;
    t=np.arange(Nsamples)/sampleRate
    y=np.zeros(Nsamples)
    for startT, tWidth, height in zip(startTimes, pulseWidths, pulseHeights):
        #y+=np.where( (t>startT) & (t<startT+tWidth), height, 0.)
        plsShape=pulseFunc(t, tWidth, startT+tWidth/2., bDigitizeFix=True)*height
        #print("area of pls: {:.5f}".format(np.sum(plsShape)))
        y+=plsShape
    return t, y

def generateTriggerWaveforms(pulseTimingParams, pulseSeqDesc, Npts, sampleRate):
    N=len(pulseSeqDesc)
    pt=pulseTimingParams
    pmpPulseWidth=pt.pumpTime

    tDeltas=[pt.tShort if seq[1].startswith('s') else pt.tLong for seq in pulseSeqDesc ] #+ [pt.depolLength]
    allPumpTimes=np.cumsum([0]+tDeltas[:-1])
    #auxPumpTimes=np.arange(N)*(pt.tShort+pt.tLong)
    #mainPumpTimes=np.arange(N)*(pt.tShort+pt.tLong) + pt.tShort
    #if pt.tShort>0:
    #    allPumpTimes=np.vstack([auxPumpTimes, mainPumpTimes]).T.ravel() #interleaved
    #else:
    #    allPumpTimes=mainPumpTimes
    allPumpTimes+=1.0/sampleRate

    #pmpPulseWidth=20*1e-6
    #sampleRate=1e6
    #t,pumpTrigWvfm=makePulseTrain(allPumpTimes+pmpPulseWidth/2, pmpPulseWidth, 1., Nsamples=Npts, sampleRate=sampleRate)
    #initPulseWidth=pmpPulseWidth

    #pmpPulseWidth=20*1e-6
    #sampleRate=1e6
    t,pumpTrigWvfm=makePulseTrain(allPumpTimes, pmpPulseWidth, 1., Nsamples=Npts, sampleRate=sampleRate)
    initPulseWidth=pmpPulseWidth
    initTrigWvfm=MT.tophat(t, initPulseWidth, initPulseWidth/2+ 1./sampleRate )
    return t, pumpTrigWvfm, initTrigWvfm

if 0:
    def squarePulse(x, width, center):
        return np.where( (x>center-width/2) & (x<center+width/2), 1, 0)
    def makePulseTrainNope(trigTimes, pulseWidths, Npts, sampleRate, pulseFunc=squarePulse):#, initTriggerWidth=10e-6):
        t=arange(Npts)/sampleRate
        pulseWidths=np.array(pulseWidths)
        if pulseWidths.ndim==0:
            pulseWidths=np.ones(len(trigTimes))*pulseWidths
        wvfmPump=np.zeros(Npts, dtype='u4')
        for center, width in zip(trigTimes, pulseWidths):
            #wvfmPump+=MT.tophat(t, width, center+width/2+(t[1]-t[0])).astype('u4')
            wvfmPump+=pulseFunc(t, width, center+(t[1]-t[0]) + initTriggerWidth).astype('u4')
        #First term is the overall trigger (should be line 0),
        # second term is line 1 and is the pumpb beam trigger
        #wvfmPort=2*MT.tophat(t, initTriggerWidth, (t[1]-t[0])) + 1*wvfmPump
        wvfmPort=2*pulseFunc(t, initTriggerWidth, initTriggerWidth/2+(t[1]-t[0])) + 1*wvfmPump
        wvfmPort=wvfmPort.astype('u4')
        #self.data[:]=[self.Vx, self.Vy, self.Vz]
        written = int32()
        self.th.WriteDigitalU32(wvfmPort.size, 1,10, DAQmx_Val_GroupByChannel, wvfmPort,byref(written), None)
        self.t=t
        self.wvfm=wvfmPort
        return self.t, self.wvfm


if __name__=="__main__":
    import pmParams
    glbP= pmParams.GlbParams()
    pTemp=glbP.p
    pTemp.pulseAmps.mPiX=0
    pTemp.pulseAmps.piX=1
    pTemp.pulseAmps.piY=0
    pTemp.pulseAmps.mPiY=0
    pTemp.pulseTiming.tShort=0
    from pylab import *
    figure()
    t,vx,vy,vz=generatePulseMagPattern(pTemp.pulseSeqDesc, pTemp.pulseTiming, pTemp.pulseAmps)
    plot(t,np.vstack([vx,vy,vz]).T)
    show()


    #Check splitting it up:
