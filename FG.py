import visa
from pyvisa import VisaIOError
from pylab import fromstring, linspace, floor
import numpy as np
import abc
import pdb
from time import sleep

def list_instruments():
    rm=visa.ResourceManager();
    resL=rm.list_resources()
    return resL

class FG(object):
    __metaclass__ = abc.ABCMeta
    #ip_address="136.159.248.161"
    addr="USB0::0x0699::0x0365::C034484::INSTR"
    handle=None;
    rm=None
    numChans=1.
    
    @staticmethod
    def array_to_text_block(data, minVal,maxVal, scl=True):
        """We'll assume it'll be sent as integers"""
        
        valRange=maxVal-minVal
        data=np.array(data,dtype='f8')
        if scl:
            lowV=data.min()
            highV=data.max()
            data=(data-lowV)/np.abs(highV-lowV) #Now between 0 and 1
            data*=valRange + minVal
        dataInt=np.rint(data).astype('i2')
        #pdb.set_trace()
        datStr=','.join([str(num) for num in dataInt])
        #print(datStr[:30])
        return datStr

    #@abc.abstractmethod
    def connect(self, address=None):
        """ Connect to the instrument. 
        After this, self.handle will be an actual handle
        """
        if address is not None:
            self.addr=address
        #self.handle=visa.instrument("TCPIP::{0}::INSTR".format(self.ip_address));
        self.handle=self.rm.open_resource("{0}".format(self.addr));
        self.configureHandle()

    @abc.abstractmethod
    def configureHandle(self):
        """ Make sure handle is configured correctly for IO, e.g. set the query delay etc.
        """
        #self.handle.query_delay=0.1

    def __init__(self, addr=None):
        self.rm=visa.ResourceManager();
        if addr is not None:
            self.addr=addr
        self.connect();
        

    def close(self):
        self.handle.close()
        
    def check_connection(self):
        try:
            ret_val=self.handle.query("*IDN?");
            print(ret_val)
        except VisaIOError:
            print("not connected! But lets try to fix it:")
            del self.handle
            self.handle=None;
            self.connect();
    
    def allOn(self):
        """Turn all outputs on"""
        for chNum in range(self.numChans):
            self.setOutputState(True, chNum);

    def allOff(self):
        """Turn all outputs off"""
        for chNum in range(self.numChans):
            self.setOutputState(False, chNum);

    @abc.abstractmethod
    def setOutputState(self, bOn, chNum=0):
        """ Turn that channel on/off
        """

    def setOutputWaveForm(self, t, x, chNum=0):
        """Upload a waveform (t, x) and set it as active on channel chNum
        """
        self.setOutputState(0, chNum)
        self.uploadWaveform(x, scl=True)
        self.setPeriod(t[-1]-t[0])
        self.setLH( x.min(), x.max() )
        errStr=self.getErr()
        errVal=int(errStr.split(',')[0])
        #if errVal!= 0 and errVal != -221:
        if errVal:
            raise ValueError(errStr.split(',')[1])
        self.setOutputState(1, chNum)

    @abc.abstractmethod
    def setTriggerMode(self, mode="SINGLE"):
        """ Set the trigger mode: e.g. continuous or whatever
        """
        pass;
    @abc.abstractmethod
    def setTriggerDelay(self, delay):
        """ Amount of time to wait after a trigger
        """
    

    def trigger(self):
        """Trigger it once
        """
        raise NotImplementedError()
    
    def setLH(self, low, high, chanNum=0):
        #self.setAmp(0.01)
        offs=(low+high)/2.
        print("Offs:{}, low:{}, high:{}".format(offs,low,high))
        if high < low:
            newLow=high#-(2*low-high)
            newHigh=low
            print("newlow, newhigh: {} {}".format(newLow, newHigh))
            #print("newLow: {}, newHigh: {}".format(newLow, newHigh))
            self.setInverted(False, chanNum=chanNum);
            self.setOffset(offs, chanNum=chanNum)
            self.setLowHigh(newLow, newHigh,chanNum=chanNum)
            self.setInverted(True, chanNum=chanNum);
        else:
            self.setInverted(False);
            self.setOffset(offs, chanNum=chanNum)
            sleep(0.1)
            self.setLowHigh(low,high, chanNum=chanNum)
        #self.allOn()

    #def setLow(self,val):
        
    #    def setHigh(self,val):
    def setLowHigh(self, low, high, chanNum=0):
        """Set the low and high values for the waveform"""
        raise NotImplementedError()

    def setOffset(self, offset, chanNum=0):
        """Set the offset for the waveform"""
        raise NotImplementedError()

    def setPeriod(self, period, chanNum=0):
        """Set the period for the waveform"""
        raise NotImplementedError()

    def setAmp(self, amp, chanNum=0):
        """Set the amp for the waveform"""
        raise NotImplementedError()

    def uploadWaveform(self, wvfm, name="VOLATILE"):
        """Set the amp for the waveform"""
        raise NotImplementedError()

    def setInverted(self, bInvert=True, chanNum=0):
        if chanNum>0:
            self.handle.write("OUTP{}:POL {}".format(int(chanNum+1), "INV" if bInvert else "NORM"))
        else:
            self.handle.write("OUTP:POL {}".format("INV" if bInvert else "NORM"))
        
    def wait(self):
        self.handle.query("*OPC?");
    
    def getErr(self):
        return self.handle.query("SYST:ERR?")
    
