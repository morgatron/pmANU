import zmq
import pmAcquire as acq
import pmProcessRaw as proc
import pmParams
import pickle
from time import sleep
import pdb
glbP=pmParams.GlbParams()


def zeroMiddle(t, dat, start,end):
    t.searchsorted([start,end])
    dat[start:end]=0
    return dat
class Serve(object):
    PORT = "5561"
    streamFile=None
    def __init__(self):
        acq.subscribe(b'raw')
        #acq.subscribe(b'seg')
        self.updateGradProc(True)
        self.SOCKET= zmq.Context().socket(zmq.PUB)
        self.SOCKET.set_hwm(10)
        self.SOCKET.bind("tcp://*:%s" % self.PORT)


        self.sampRate=29
        Nhist=300
        self.magHist=np.zeros((Nhist, 3))
        self.calFact=np.ones(3, dtype='f8')
        self.modAmpL=[0.05,0.05,0.05]
        self.modFreqL=[5,3,1]
        #self.tAx=arange(Nhist)/self.sampRate
        self.bStopStreaming=False
    def updateGradProc(self, justDoIt=False):
        #print("check for updated grad")
        if glbP.isCalUpdated() or justDoIt==True:
            cal=glbP.loadCal()
            self.gradD=cal['grad']
            t=cal['t']
            self.gradD={key: zeroMiddle(t, self.gradD[key], 100e-6,cal['pars'].pulseTiming.tau-100e-6) for key in self.gradD}
            self.gradProcessedD=proc.doAllGradProcessing(t, self.gradD, cal['pars'])
    def update(self):
        self.updateGradProc()
        #datL=acq.checkForPublished()
        #topic,(t0L,rawL,dt)=acq.checkForPublished()
        rep=acq.checkForPublished()
        if rep is None:
            #print("Nothing recieved")
            return
        topic,datD=rep
        dt=datD['dt']
        rawL=datD['data']
        #print("RawL[0,:5], {}".format(rawL[0][:10]))
        #print("t[:4], {}".format(datD['t'][:4]))

        #t
        #for raw in rawL:
        if len(rawL):
            try:
                rawL=np.vstack(rawL)
            except ValueError:
                newMaxL=min([r.size for r in rawL])
                rawL=[r[:newMaxL] for r in rawL]
            t=np.arange(rawL[0].size)*dt
            #print(t.max())
            #(t,raw)=dat
            #self.updateGradProc()
            #self.mag=proc.doSimpleMagnetometry(t, rawL, glbP.p, processedGrad=self.gradProcessedD, startDumpT=300e-6,endDumpT=500e-6, toFit=['Vx', 'Vy', 'Vz', 'piX', 'mPiX', 'piY', 'mPiY'])
            toFitDubL=['Vx', 'Vy', 'Vz', 'piX',  'piY','piX_B', 'piY_B']
            toFitNormL=['Vx', 'Vy','Vz', 'piX',  'piY']
            toFitNoPL=['Vx', 'Vy','Vz']
            self.mag=proc.fitSubbed(t, rawL, glbP.p, processedGrad=self.gradProcessedD, startDumpT=200e-6,endDumpT=50e-6, toFit=toFitNormL,# 'mPiY'],
                    bFitShort=False)
            #self.mag=proc.doSimpleMagnetometry(t, rawL, glbP.p, processedGrad=None, startDumpT=100e-6, endDumpT=50e-6, toFit=['Vx', 'Vy', 'Vz', 'piX', 'mPiX', 'piY', 'mPiY'])
            #print("got mag: {}...".format(self.mag[:2]))
            print("sent {} segs".format(len(self.mag)))
            mag2Send=self.mag
            #mag2Send[:,:3]/=self.calFact
            msg=b'mag '+ pickle.dumps(mag2Send)
            self.SOCKET.send(msg)
        if self.streamFile is not None:
            self.mag.tofile(self.streamFile)
        #else:
        #    self.updateGradProc()
        Nnew=self.mag.shape[0]
        self.magHist=np.roll(self.magHist, -Nnew, axis=0)
        self.magHist[-Nnew:]=self.mag[:,:3]
        if self.magHist[0].mean()!=0:
            #pdb.set_trace()
            self.updateCals()
            #self.bStopStreaming=True

    def updateCals(self):
        tAx=np.arange(self.magHist.shape[0])/self.sampRate
        self.magHist-=self.magHist.mean(axis=0)
        for k,f in enumerate(self.modFreqL):
            Tpts=1./f*self.sampRate
            N=np.floor(tAx.size/Tpts)*Tpts
            sinQuad=((np.sin(2*np.pi*f*tAx[:N])*self.magHist[:N,k]).mean())
            cosQuad=((np.cos(2*f*np.pi*tAx[:N])*self.magHist[:N,k]).mean())
            self.calFact[k]=np.sqrt( sinQuad**2  + cosQuad**2)/self.modAmpL[k]*2
            amp=sinQuad + 1j*cosQuad
            #print("amp: {}, phase: {}".format(amp, np.angle(amp)))
            print(self.calFact[k])

    def startSavingToFile(self, filename, notes=""):
        datDir="c:/pulseSerfData/magStreams/"
        self.streamFile=open(datDir+filename+"_strm.npz", 'wb')
        np.savez(datDir+filename+"_meta.npz", notes=notes, grad=self.gradD, pars=glbP.p)

    def stopStreaming(self):
        self.streamFile.close()
        self.streamFile=None


import time
import numpy as np
class Client(object):
    dt=1/60.
    def __init__(self):
        self.PORT=Serve.PORT
        self.t0=time.time()
        self.SOCKET= zmq.Context().socket(zmq.SUB)
        self.SOCKET.set_hwm(10)
        self.SOCKET.connect("tcp://localhost:%i" % self.PORT)
        self.SOCKET.setsockopt(zmq.SUBSCRIBE, b'mag')

    def getNewVals(self):
        if self.SOCKET.poll(10):
            topic,msg= self.SOCKET.recv().split(b' ', 1)
            fitPars=pickle.loads(msg).T
            t=np.arange(fitPars[0].size)*self.dt+time.time()-self.t0
            #print("got {} vals".format(t.size))
            return t, fitPars
        else:
            return None
    def close(self):
        self.SOCKET.close()

if __name__=="__main__":
    sv=Serve()
    sv.startSavingToFile("jul28_prbNoise_3", notes="")



    while 1 and not sv.bStopStreaming:
        sv.update()
        sleep(0.001)

