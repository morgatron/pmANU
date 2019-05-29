from fgAgilent import AgilentFG
from fg33500 import Agilent33500
from fg3390 import KiethleyFG
import numpy as np
import MT

class PicoDummyFG(object):
    waveformDuration=None
    wvfm=None
    triggerSource=3#"ScopeTrig"
    triggerType="Falling"
    maxPts=16384
    def __init__(self, ps):
        self.ps=ps
    def setPeriod(self, period):
        self.waveformDuration=period
        #self.reupload()
    def reupload(self):
        #duration=self.wvfm.size/2e6
        duration=self.waveformDuration
        if self.wvfm.size>self.maxPts:
            tInp=np.linspace(0,duration,self.wvfm.size)
            #indx=np.arange(self.wvfm.size)
            #wvfm=np.interp(np.linspace(0,indx[-1], 16384), indx, self.wvfm)
            wvfm=np.interp(0e-6+np.linspace(0,duration, self.maxPts),tInp, self.wvfm)
        else:
            wvfm=self.wvfm
        self.wvfmUpload=wvfm
        self.waveformDuration=duration
        self.ps.setAWGSimple(wvfm, duration, triggerSource=self.triggerSource, triggerType=self.triggerType, offsetVoltage=None )
    def uploadWaveform(self, wvfm):
        self.wvfm=wvfm
        self.reupload()
    def setLH(self, low, high):
        curAmpl=self.wvfm.max()-self.wvfm.min()
        self.wvfm*=abs(high-low)/curAmpl
        self.wvfm-=self.wvfm.min()+low
        self.reupload()
        #ps.setAWGSimple( ((2**15-1)*pumpPulseWvfm).astype('i2')+2**15,wvfmDuration, triggerSource="None", pkToPk=2., offsetVoltage=0.)
    def allOff(self):
        self.uploadWaveform(zeros(self.maxPts))


from FG import FG
class DummyFG(FG):
    def connect(self):
        pass;
    def __init__(self):
        pass;
    def close(self):
        pass;
    def setTriggerDelay(self, delay):
        pass;
    def setTriggerMode(self, mode):
        pass;
    def setOutputWaveForm(self, t, x, chNum):
        pass;
    def setOutputState(self, bOn, chNum=0):
        pass;
    def configureHandle(self):
        pass;
    def uploadWaveform(self, wvfm, name="VOLATILE"):
        pass;

class PmFgController(object):
    rateX=250000000/10
    rateY=250000000/10
    rateZ=125000000/100
    pulseHeightParams=None 
    pulseTimeParams=None
    t=np.zeros(1)
    VX=np.zeros(1)
    VY=np.zeros(1)
    VZ=np.zeros(1)
    bXChanged=None;
    bYChanged=None;
    bZChanged=None;
    bNoZ=False

    def __init__(self, ps=None):
        self.bXChanged=True;
        self.bYChanged=True;
        self.bZChanged=True;
        self.bXAmpChanged=True
        self.bYAmpChanged=True
        self.bZAmpChanged=True
        #self.fgP=PicoDummyFG(ps);
        #self.fgA=AgilentFG();
        self.fgK=KiethleyFG();
        self.fg33500=Agilent33500()
        self.allOff()

    def setRates(self, rateX=None, rateY=None, rateZ=None):
        if rateX is not None:
            self.rateX=rateX
        if rateY is not None:
            self.rateY=rateY
        if rateZ is not None:
            self.rateZ=rateZ

    def setWaveForms(self, t,VX, VY, VZ, bReorderOnly=False):
        self.t=t
        if 0:
            if t.size!= self.t.size or ~np.all(t==self.t):
                #print("t changed")            
                self.bXChanged=True
                self.bYChanged=True
                self.bZChanged=True
        if self.VX.size!= VX.size or ~np.all(self.VX==VX):
            #print("Xchanged")
            self.bXChanged=True
            self.bXAmpChanged=True
        if self.VY.size!= VY.size or ~np.all(self.VY==VY):
            #print("Ychanged")
            self.bYChanged=True
            self.bYAmpChanged=True
        if self.VZ.size!= VZ.size or ~np.all(self.VZ==VZ):
            #print("Zchanged")
            self.bZChanged=True
            self.bZAmpChanged=True
        

        if VX.size>0 and bReorderOnly:
            if self.bXChanged and self.VX.max()==VX.max() and self.VX.min()==VX.min():
                self.bXAmpChanged=False
            if self.bYChanged and self.VY.max()==VY.max() and self.VY.min()==VY.min():
                self.bYAmpChanged=False
            if self.bZChanged and self.VZ.max()==VZ.max() and self.VZ.min()==VZ.min():
                self.bZAmpChanged=False


        self.VX, self.VY, self.VZ = VX, VY, VZ
        self.updateOutputs()
        return

    def updateOutputX(self):
        fg=self.fg33500
        fg.allOff()
        fg.setRate(self.rateX, 0)
        if all(self.VX==0):
            fg.uploadWaveform(self.VX/max(abs(self.VX)), name='VOLX', chanNum=0);
            fg.setLH( -0.01, 0.01, chanNum=0 )
        else:
            #fg.uploadWaveform(self.VY/max(abs(self.VY)));
            fg.uploadWaveform(self.VX/max(abs(self.VX)),name='VOLX',chanNum=0);
            if self.bXAmpChanged:
                fg.setLH( self.VX.min(), self.VX.max(), chanNum=0 )
        errStr=fg.getErr()
        errVal=int(errStr.split(',')[0])
        #if errVal!= 0 and errVal != -221:
        if errVal:
            raise ValueError(errStr.split(',')[1])
        self.bXChanged=False
        self.bXAmpChanged=False

    def updateOutputY(self):
        fg=self.fg33500
        fg.allOff()
        fg.setRate(self.rateY, 1)
        if all(self.VY==0):
            fg.uploadWaveform(self.VY,name='VOLY',chanNum=1);
            fg.setLH( -0.01, 0.01, chanNum=1 )
        else:
            #fg.uploadWaveform(self.VY/max(abs(self.VY)));
            fg.uploadWaveform(self.VY/max(abs(self.VY)),name='VOLY',chanNum=1);
            if self.bYAmpChanged:
                fg.setLH( self.VY.min(), self.VY.max(), chanNum=1 )
        errStr=fg.getErr()
        errVal=int(errStr.split(',')[0])
        #if errVal!= 0 and errVal != -221:
        if errVal:
            raise ValueError(errStr.split(',')[1])
        self.bYChanged=False
        self.bYAmpChanged=False

    def updateOutputZ(self): 
        fg=self.fgK
        if self.bNoZ:
            return

        fg.setOutputState(False)
        #fg.setRate(self.rateZ)
        #self.VZ=self.VZ[:32000]
        fgZPeriod=1./self.rateZ*self.VZ.size
        print("fgZPeriod: {}".format(fgZPeriod))
        fg.setPeriod(fgZPeriod)
        if 1:
            if abs(max(self.VZ)-min(self.VZ))<0.05:
                fg.uploadWaveform(self.VZ*0, scl=False);
                fg.setOffset(0)
                fg.setAmp( 0.05 )
            else:
                fg.uploadWaveform(self.VZ/max(abs(self.VZ)),scl=True);
                if self.bZAmpChanged:
                    fg.setLH( self.VZ.min(), self.VZ.max() )

        errStr=fg.getErr()
        print(errStr)
        errVal=int(errStr.split(',')[0])
        #if errVal!= 0 and errVal != -221:
        if errVal!=0:
            raise ValueError(errStr.split(',')[1])
        self.bZChanged=False
        self.bZAmpChanged=False

    def updateOutputs(self):
        if self.bXChanged:
            self.updateOutputX()
        if self.bYChanged:
            self.updateOutputY()
        if self.bZChanged:
            self.updateOutputZ()

    def allOn(self):
        #self.fgP.allOn()
        self.fgK.allOn()
        self.fg33500.allOn()
        #self.fgK.allOn()

    def allOff(self):
        #self.fgP.allOff()
        self.fgK.allOff()
        self.fg33500.allOff()


    def setWaveformsResponseCorrected(t, Vx,Vy, Vz):
        """Should take the desired output waveforms (Vx,Vy, Vz) and change the actual output according to the known response.
        """
        raise NotImplementedError



if __name__=="__main__":
    from numpy import pi, sin, cos 
    t=np.arange(0,15e-3, 1e-6)
    y=np.sin(2*pi*t*1000)
    pfg=PmFgController()
    pfg.setRates(1e6,1e6,1e6)
    pfg.setWaveForms(y,y,y)
    
