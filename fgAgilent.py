# -*- coding: utf-8 -*-

import visa

import numpy as np
from time import sleep, time
import pdb

import FG

def list_instruments():
    rm=visa.ResourceManager();
    print(rm.list_resources())
    return

class AgilentFG(FG.FG):
    #ip_address="136.159.248.161"
    #addr="USB0::0x0699::0x0365::C034484::INSTR"
    addr="USB0::0x0957::0x0407::MY44017778::INSTR"
    handle=None;
    rm=None
    numChans=1

    def configureHandle(self):
        self.handle.timeout=5000
    @staticmethod
    def array_to_text_block(data, scl=True):
        data=np.array(data,dtype='f8')
        if scl:
            lowV=data.min()
            highV=data.max()
            data=(data-lowV)/np.abs(highV-lowV)*2.-1.0
            #data/=abs(data).max()
            data*=8191
        dataInt=np.rint(data).astype('i2')
        #pdb.set_trace()
        datStr=','.join([str(num) for num in dataInt])
        #print(datStr[:100])
        return datStr
        #return "#{0}{1}{2}".format(len(Nstr), Nstr), data.tobytes())

    @staticmethod
    def array_to_binary_block(data, scl=True):
        data=np.array(data)
        if scl:
            data/=abs(data).max()
            
        data=np.rint(data).astype('i2')
        dataBytes=bytes(data)
        N=len(dataBytes)
        Nstr=str(N)
        return ( "#{0}{1}".format(len(Nstr), Nstr),  dataBytes )
        #return "#{0}{1}{2}".format(len(Nstr), Nstr), data.tobytes())

    def uploadWaveform(self, data, scl=True, name="VOLATILE"):
        #binBlockHead, binBlockValues=self.array_to_binary_block(data,scl=scl)
        datStr=self.array_to_text_block(data,scl=scl)
        self.handle.write("DATA:DAC {}, {}".format(name,datStr))
        #self.handle.write_raw(bytes( "DATA:DAC VOLATILE, {}".format(binBlockHead), 'ascii' )+ binBlockValues)
        sleep(.5)
        #pdb.set_trace()
        

    def setOutputState(self, bOn, chNum=0):
        stateStr= 'ON' if bOn else 'OFF'
        out=self.handle.write("OUTPUT {}".format(stateStr));
        return out
        

    def setOutputWaveForm(self, t, x, chNum):
        """Upload a waveform (t, x) and set it as active on channel chNum
        """
        pass;

    def setTriggerDelay(self, delay):
        """ Amount of time to wait after a trigger
        """
        raise NotImplementedError('Kiethey 3390 has no trigger delay')
    
    def setTriggerMode(self, mode="SINGLE"):
        allowed_modes=['SINGLE', 'NORM', 'SINGLE', 'STOP']
        if mode in allowed_modes:
            self.handle.write("TRIG_MODE {0}".format(mode)) 
        else:
            raise Exception("trigger mode must be one of {0}".format(allowed_modes))
    
    def setPeriod(self, T):
        self.handle.write('FREQ {:.5f}'.format(1./T))
        
    def setLowHigh(self, low, high, chanNum=0):
        self.handle.write('VOLT:LOW {:.5f}'.format(low) )
        self.handle.write('VOLT:HIGH {:.5f}'.format(high) )
        
    def setOffset(self, offset, chanNum=0):
        self.handle.write('VOLT:OFFS {:.5f}'.format(offset) )
        
    def setLoad(self, load=50):
        """If load is -ve, infinite is assumed
        """
        if load >0:
            loadStr="50"
        else:
            loadStr="INF"
        self.handle.write('OUTP:LOAD {}'.format(loadStr) );        
        
    def trigger(self):
        self.handle.write("ARM");
    def stop(self):
        self.handle.write("STOP");
    def wait(self):
        self.handle.query("*OPC?");
    


## def log_channel(trace, sample_time=2, num_samples=10, filename="biglog"):
##     from time import sleep, time
##     scope=Scope();
##     scope.connect()
##     
##     fout=open("{0}.npz".format(filename), 'wb') #delete the old file
##     fout.close()
##     
##     traces=[]
##     start_time=time();
##     cur_time=start_time;
##     for k in range(num_samples):
##         while cur_time-start_time < k*sample_time:
##             sleep(0.01)
##             cur_time=time()
##         
##         t, y =scope.get_trace(trace=trace)
##         scope.handle.write("CLSW");
##         traces.append(y)
##         print("Got sample {0}".format(k))
##         
##         #Save to a file for safety if things break
##         fout=open("{0}.npz".format(filename), 'ab')
##         np.save(fout, y)
##         fout.close()
##         
##         
##         
##     return t, traces
    
## def scan_channel(trace, sample_time=2,  f_handle=None, params=[], filename="biglog"):
##     from time import sleep, time
##     scope=Scope();
##     scope.connect()
##     
##     fout=open("{0}.npz".format(filename), 'wb') #delete the old file
##     fout.close()
##     
##     traces=[]
##     
##     
##     num_samples=len(params)
##     for k in range(num_samples):
##         f_handle(params[k])
##         sleep(0.1)
##         scope.handle.write("CLSW");
##         start_time=time();
##         cur_time=start_time;
##         while cur_time-start_time < sample_time:
##             sleep(0.01)
##             cur_time=time()
##         
##         t, y =scope.get_trace(trace=trace)
##         
##         traces.append(y)
##         print("Got sample {0}".format(k))
##         
##         #Save to a file for safety if things break
##         fout=open("{0}.npz".format(filename), 'ab')
##         np.save(fout, y)
##         fout.close()
##         
##         
##         
##     return t, traces
    
    




if __name__=="__main__":
    fg=AgilentFG();
    def sendPulse(tDelay=1, tWidth=3000, tTotal=4096):

        t=np.linspace(0,tTotal,tTotal*20)*1.0;
        y=np.where( (t>tDelay) & (t<tDelay+tWidth), 5.0, 0.)

        #fg.allOff()
        fg.setLoad(50)
        from pylab import plot,show
        plot(t,y)
        fg.uploadWaveform(y);
        fg.setPeriod(tTotal*1e-6);
        fg.setLowHigh(0,4.5)
        print(fg.handle.query("VOLT:OFFS?"))
        #print (rfg.setOutputWaveForm(t, y2, 1))
        #fg.allOn()
        #print("stuff")
        show()

    sendPulse(10.,10.,1000)
    fg.setOffset(-0.01);
