import fgAgilent

class PmTriggerTask(object):
    initTH=None;
    pmpTH=None
    def __init__(self):
        self.fg=fgAgilent.AgilentFG()
        pass;
    def setParams(self, pulseTiming, seqDesc):
        self.fg
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
