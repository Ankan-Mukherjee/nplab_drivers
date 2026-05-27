import qcodes as qc
import numpy as np
import time
from math import sqrt
from datetime import datetime

class CapacitanceBridge:
    """
    Object for the capacitance bridge in experiment
    Controls the specified channel (default: channel 2) of the AC box

    To initialize the object:
        tolerance: maximum tolerance for the output voltage
        ac_device: object for the device controlling the AC source at reference capacitor
        lockin_device: object for the device measuring the output voltage of the capacitance bridge
        wait_time: waiting time (in seconds) after chaning the AC source but before reading outputs from lockin
        ref_channel: the channel of the function generator which will be varied during balancing (i.e. channel for reference voltage)
        sample_channel: the channel of the function generator for exciting the sample
        sample_attenuation: attenuation (in dB) attached to the sample input
        ref_attenuation: attenuation (in dB) attached to the reference input
    """

    WAIT_TIME_BETWEEN_OPERATIONS = 0.1 # wait time between each operation to the function generator (in seconds). Try to increase if the operation failed during balancing.

    def __init__(self, tolerance, ac_device, lockin_device, wait_time, ref_channel=2, sample_channel=1):
        self.tolerance      = tolerance
        self.ac_device      = ac_device
        self.lockin_device  = lockin_device
        self.wait_time      = wait_time
        self.channel        = ref_channel
        self.count          = 0
        self.sample_channel = sample_channel
        
        self.ac_scale = 1                       # just some dummy value

    def setACScale(self, sample_attentuation=30, ref_attenuation=30):
        if self.sample_channel == 1:
            sample_volt = self.ac_device.C1_volt()
        if self.sample_channel == 2:
            sample_volt = self.ac_device.C2_volt()

        self.ac_scale = 10**(-(ref_attenuation - sample_attentuation) / 20.0) / sample_volt

    def setReferenceACVoltage(self, ac_volt):
        """
        Changes the function generator to the given parameters

        Argument:
        ac_volt: Parameters of the function generator in a NumPy array (format: ac_volt=numpy.array([[X], [Y]])). Will be converted to amplitude and phase by the program

        Returns: 
            amplitude: amplitude of ac_volt
            phase: phase of ac_volt
        """

        self.referenceVoltage = ac_volt
        amplitude, phase = CapacitanceBridge.getVectorPhaseAndMagnitude(ac_volt)
        # print("Set Reference AC Voltage: Amplitude = %.6e, Phase = %.1f \n" % (amplitude, phase))

        if self.channel == 1:
            time.sleep(CapacitanceBridge.WAIT_TIME_BETWEEN_OPERATIONS)
            self.ac_device.C1_volt(amplitude)
            time.sleep(CapacitanceBridge.WAIT_TIME_BETWEEN_OPERATIONS)
            self.ac_device.C1_phase(phase)
            time.sleep(CapacitanceBridge.WAIT_TIME_BETWEEN_OPERATIONS)

        if self.channel == 2:
            time.sleep(CapacitanceBridge.WAIT_TIME_BETWEEN_OPERATIONS)
            self.ac_device.C2_volt(amplitude)
            time.sleep(CapacitanceBridge.WAIT_TIME_BETWEEN_OPERATIONS)
            self.ac_device.C2_phase(phase)
            time.sleep(CapacitanceBridge.WAIT_TIME_BETWEEN_OPERATIONS)

        return amplitude, phase

    def measureLockinVout(self):
        """
        Measures the output voltage (X & Y) from the lockin amplifier. The sensitivity is adjusted automatically when the output is overloaded.

        Returns:
            Numpy Array numpy.array([[X], [Y]])
                X: Measured X from lockin amplifier
                Y: Measured Y from lockin amplifier
        """    
       
        X, Y = self.lockin_device.X(), self.lockin_device.Y()
        #print("Measured Lockin Output Voltage: X = %.6e, Y=%.6e, \n\n" % (X, Y))
        
        return np.array([[X], [Y]], dtype=np.longdouble)

    # def auto_range(self, max_changes=10):   # for sr830
    #     """
    #     Find the optimal sensitivity for the lockin amplifier at a given output. Increases/Decreases the sensitivity by one step and re-measures to check if overload has occurred.

    #     Argument:
    #     max_changes: Maximum number of steps for finding the suitable sensitivity (default: 10)
    #     """

    #     def autorange_once() -> bool:
    #         r = self.lockin_device.R()
    #         sens = self.lockin_device.sensitivity()
    #         if r > 0.9 * sens:
    #             return self.lockin_device.increment_sensitivity()
    #         elif r < 0.1 * sens:
    #             return self.lockin_device.decrement_sensitivity()
    #         return False

    #     sets = 0
    #     while autorange_once() and sets < max_changes:
    #         sets += 1
    #         time.sleep(1)

    def calculateResponseMatrix(self, initRefVoltage1, initRefVoltage2):
        """
        Calculates the response matrix M from two given voltages. The output voltage will be measured from the two input voltages to construct M.

        Arguments:
        initRefVoltage1: One guess of reference voltage (format: numpy.array([[X], [Y]]))
        initRefVoltage2: Another guess of reference voltage (format: numpy.array([[X], [Y]]))

        Returns:
            M, Vout1, Vout2:
                M: Response matrix
                Vout1: Measured output voltage using initRefVoltage1 (format: numpy.array([[X], [Y]]))
                Vout2: Measured output voltage using initRefVoltage2 (format: numpy.array([[X], [Y]]))
        """

        self.setReferenceACVoltage(initRefVoltage1)
        time.sleep(self.wait_time)
        self.lockin_device.auto_range()
        Vout1 = self.measureLockinVout()

        self.setReferenceACVoltage(initRefVoltage2)
        time.sleep(self.wait_time)
        self.lockin_device.auto_range()
        Vout2 = self.measureLockinVout()
        
        dS_in  = initRefVoltage2 - initRefVoltage1
        dS_out = Vout2 - Vout1
        magnitude = np.linalg.norm(dS_out) / np.linalg.norm(dS_in)
        phase_shift = np.arctan2(dS_out[1, 0] , dS_out[0, 0]) - np.arctan2(dS_in[1, 0], dS_in[0, 0])
        self.responseMatrix = magnitude * np.matrix(([np.cos(phase_shift), -1.0 * np.sin(phase_shift)],
                                    [np.sin(phase_shift), np.cos(phase_shift)]))
        return self.responseMatrix, Vout1, Vout2

    def getOffsetFromReferenceVoltage(self, Vref, Vout):
        """
        Calculate the offset from the equation Vout = M Vref + Voffset.

        Arguments:
        Vref: Reference voltage (format: numpy.array([[X], [Y]]))
        Vout: Output voltage measured from the lockin amplifier (format: numpy.array([[X], [Y]]))   

        Return:
            Voffset 
        """

        self.Voffset = Vout - self.responseMatrix.dot(Vref)
        return self.Voffset

    def calculateBalancedCapacitanceAndDissipation(self):
        """ 
        Calculate the sample's capacitance and dissipation from a BALANCED bridge at the balance point. 
        Note: The capacitance measured is in unit of Cref
        """

        self.balanced_capacitance = -1.0 * self.referenceVoltage[0, 0] * self.ac_scale
        self.balanced_dissipation =  1.0 * self.referenceVoltage[1, 0] * self.ac_scale
        print("Balanced Capacitance = %.6f, Dissipation = %.6f" % (self.balanced_capacitance, self.balanced_dissipation))

        # self.calculateOffbalanceVariations()
        return self.balanced_capacitance, self.balanced_dissipation # , self.delta_C, self.delta_D

    def calculateOffbalanceVariations(self):
        """
        Calculate the off balance capacitance and dissipation (\delta C & \delta D) (1st order change when Vout is varied in subsequent measurements)
        """

        determinant = np.linalg.det(self.responseMatrix)

        offset_x, offset_y       = self.Voffset[0, 0], self.Voffset[1, 0]
        reference_x, reference_y = self.referenceVoltage[0, 0], self.referenceVoltage[1, 0]

        self.delta_C = -1.0 * (reference_x * offset_x + reference_y * offset_y) / (reference_x**2 + reference_y**2) / determinant * self.ac_scale
        self.delta_D =  1.0 * (reference_y * offset_x - reference_x * offset_y) / (reference_x**2 + reference_y**2) / determinant * self.ac_scale
        print("Off Balance delta Capacitance = %.6f, delta Dissipation = %.6f" % (self.delta_C, self.delta_D))

    def getOffBalanceCapacitanceAndDissipation(self):
        """
        Apply linear approximation for subsequent offbalance measurements.
        Note: The capacitance measured is in unit of Cref
        
        Returns:
            SampleCapacitance, SampleDissipation
        """ 
        Vout = self.measureLockinVout()

        capacitance = self.balanced_capacitance + (self.delta_C * Vout[0, 0] + self.delta_D * Vout[1, 0])
        dissipation = self.balanced_dissipation + (self.delta_D * Vout[0, 0] - self.delta_C * Vout[1, 0])
        
        #print("Measured off-balance Capacitance = %.6f, Dissipation = %.6f" % (capacitance, dissipation))
        return capacitance, dissipation

    @staticmethod
    def getVectorPhaseAndMagnitude(vec):
        """
        Calculates the phase and magnitude from a given vector in cartesian coordinate (X & Y)
        """
        vec = np.array(vec)
        phase = 1.0 * np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
        if phase < 0:
            phase += 360
        return (np.linalg.norm(vec), phase)

    @staticmethod
    def getVectorXY(amplitude, phase):
        """
        Converts a vector in polar coordinate to cartesian coordinate
        phase: in degrees
        """
        phase = np.radians(phase)
        return amplitude * np.cos(phase), amplitude * np.sin(phase)

def balance_voltage(bridge, finalSensitivity, final_timeConstant, initVolt1, initVolt2, bruteForceTolerance=7e-7, time_constant=1, max_iteration=15):
    """
    Balance the reference voltage such that the output voltage < tolerance

    Algorithm for balancing: 
    Overview: An implementation of 2d Newton's method for solving the root of the matrix equation: Vout = M Vref + Voffset. Here M and Voffset is assumed to be constant in Vref. 
              Note: The convergence rate is highly dependent on the initial guess for Vref. Try to have an idea of the apporixmate balance point and start from there may help speed up the process

    Procedures:
        1. Starts with two initial guesses for Vref
        2. Using the two initial Vref, the response matrix can be calculated. (Note: This is possible because the response matrix assumes the form [[a, -b]
                                                                                                                                                    [b,  a]])
        3. Voffset can then be calculated using Voffset = Vout - M Vref
        4. (Starts the loop) Get another Vref from  Vref = - M^(-1) Voffset
        5. Measure Vout using the new Vref from Step 4   
        6. Check the magnitude of Vout from Step 5. If the magnitude is smaller than the tolerance specified, the bridge is balanced and the loop will be terminated
        7. If the magnitude is larger than the tolerance, calculate a new Voffset from the Vout and Vref of Step 4 & 5
        8. If the iteration count is smaller than the specified maximum iteration, repeats Step 4 using the Voffset from Step 7. Otherwise, the balancing failed and the loop terminates.
        9. If the balancing is successful, calculate the capacitance, dissipation and off balance variations measured at the balance point.

    Arguments:
    bridge: CapacitanceBridge object that controls the capacitance bridge in the experiment
    final_timeConstant: Time constant after balancing
    initVolt1: Initial reference AC voltage (in format [[V_x], [V_y]]). Change the initial reference voltage for faster convergence if necessary
    initVolt2: Another initial reference AC voltage (in format [[V_x], [V_y]])
    time_constant: Time constant used during balancing (default 1 second)
    max_iteration: Maximum number of iterations before terminating the balancing loop

    Return: Reference voltage that makes the output voltage < tolerance (or None if the balancing failed)
    """

    time.sleep(1)
    bridge.lockin_device.sensitivity(0.5)
    time.sleep(1)
    bridge.lockin_device.time_constant(time_constant)

    initVolt1 = np.array(initVolt1, dtype=np.longdouble)
    initVolt2 = np.array(initVolt2, dtype=np.longdouble)
    Vref_new = initVolt1

    _, Vout, _ = bridge.calculateResponseMatrix(initVolt1, initVolt2)    
    Voffset = bridge.getOffsetFromReferenceVoltage(Vref_new, Vout)      

    count = 1
    balanced = False
    while (count <= max_iteration) and (not balanced):
        print("Starting Iteration: %d" % count)
        
        Vref_new = -1.0 * (bridge.responseMatrix.I).dot(Voffset)       
        bridge.setReferenceACVoltage(Vref_new)
        
        mag, phase = CapacitanceBridge.getVectorPhaseAndMagnitude(Vref_new)
        print("Set Vref to Amplitude = %.4f, Phase = %.1f" % (mag, phase))

        if mag > 5:
            print("Vref Diverged")
            return None 

        time.sleep(bridge.wait_time)
        bridge.lockin_device.auto_range()

        Vout = bridge.measureLockinVout()
        Voffset = bridge.getOffsetFromReferenceVoltage(Vref_new, Vout)

        if np.linalg.norm(Vout) < bridge.tolerance:
            balanced = True
            mag, phase = CapacitanceBridge.getVectorPhaseAndMagnitude(bridge.referenceVoltage)
            print("The output voltage is balanced.")
            print("Reference Voltage: Amplitude = %.3f, Phase = %.1f" % (mag, phase))
            print("Output Voltage: X = %.3e, Y=%.3e" % (Vout[0, 0], Vout[1, 0]))
            
            bridge.balanced = True #may can remove

        count += 1

    if not balanced:
        # try brute force before give up
        old_tol = bridge.tolerance
        bridge.tolerance = bruteForceTolerance

        print("Try to brute force now")
        prev_vout = bridge.measureLockinVout()
        current_vout = 0
        count = 0
        direction = +1

        step_size = 0.2e-3

        while count < 25 and np.linalg.norm(prev_vout) > bridge.tolerance:         
            print("Brute Force Count: %d" % count)   
            mag, phase = CapacitanceBridge.getVectorPhaseAndMagnitude(bridge.referenceVoltage)
            mag += step_size * direction   # move one step in amplitude
            print("Change Reference Voltage: Amplitude = %.4f, Phase = %.1f" % (mag, phase))

            X, Y = bridge.getVectorXY(mag, phase)
            bridge.setReferenceACVoltage(np.array([[X], [Y]]))
            time.sleep(bridge.wait_time)

            current_vout = bridge.measureLockinVout()

            if np.linalg.norm(current_vout) < bridge.tolerance: # we're done, output is smaller than tolerance
                bridge.lockin_device.sensitivity(finalSensitivity)
                bridge.lockin_device.time_constant(final_timeConstant)
                bridge.tolerance = old_tol

                return np.array([[X], [Y]])

            if np.linalg.norm(current_vout) > np.linalg.norm(prev_vout): # output is increasing, reverse the direction
                direction *= -1
                mag += step_size * direction
                X, Y = bridge.getVectorXY(mag, phase)
                bridge.setReferenceACVoltage(np.array([[X], [Y]]))
            else:
                prev_vout = current_vout  # output is decreasing, try to further decrease in another iteration

            count += 1

        print("Still doesn't work. Now try smaller steps")

        prev_vout = bridge.measureLockinVout()
        current_vout = 0
        count = 0
        direction = +1

        step_size = 0.1e-3

        while count < 5 and np.linalg.norm(prev_vout) > bridge.tolerance:         
            print("Brute Force Count: %d" % count)   
            mag, phase = CapacitanceBridge.getVectorPhaseAndMagnitude(bridge.referenceVoltage)
            mag += step_size * direction   # move one step in amplitude
            print("Change Reference Voltage: Amplitude = %.4f, Phase = %.1f" % (mag, phase))

            X, Y = bridge.getVectorXY(mag, phase)
            bridge.setReferenceACVoltage(np.array([[X], [Y]]))
            time.sleep(bridge.wait_time)

            current_vout = bridge.measureLockinVout()

            if np.linalg.norm(current_vout) < bridge.tolerance: # we're done, output is smaller than tolerance
                bridge.lockin_device.sensitivity(finalSensitivity)
                bridge.lockin_device.time_constant(final_timeConstant)
                bridge.tolerance = old_tol

                return np.array([[X], [Y]])

            if np.linalg.norm(current_vout) > np.linalg.norm(prev_vout): # output is increasing, reverse the direction
                direction *= -1
                mag += step_size * direction
                X, Y = bridge.getVectorXY(mag, phase)
                bridge.setReferenceACVoltage(np.array([[X], [Y]]))
            else:
                prev_vout = current_vout  # output is decreasing, try to further decrease in another iteration

            count += 1

        bridge.tolerance = old_tol
        print("Maximum iterations reached")
        return None

    bridge.lockin_device.sensitivity(finalSensitivity)
    print("Sensitivity", finalSensitivity)
    bridge.lockin_device.time_constant(final_timeConstant)
    
    return bridge.referenceVoltage

def createLiveSingleParamPlotCapacitance(bridge, sensitivity, time_constant, SetParam, SetArray, *MeasParams, SetDelay=0.0,
                       DataName='', XParam=None, YParam=None,
                       plot_results=True, save_plots=True):
    """ Single parameter sweep, single measure (for more measurements, add
    parameters to the .each() part). Includes live plot.

    Returns: data (a qcodes DataSet object), plot

    Arguments:
    bridge: The bridge object to balance
    sensitivity: Measurement sensitivity
    time_constant: Measurement time constant
    SetParam: The parameter to sweep (such as a voltage)
    SetArray: should be a list or numpy array of values you want to set
                SetParam to.
    SetDelay: The delay time between when SetParam is set till the MeasParams
                are measured (0 by default).
    *MeasParam: The comma-separated parameters you want to measure at each
                setpoint

    Keyword Arguments:
    DataName: A name to tag the data (defaults to nothing)
    XParam: Optional, the x parameter to be used in plotting (if not used, will
                default to the set parameter for every plot). Must be either a
                list that is the same length as YParam, a single parameter, or
                None.
    YParam: Allows you to pick only a few parameters to plot out of those
                measured. (if not mentioned, will plot all *MeasParams)
    plot_results: True by default, if false, suppresses plotting
    save_plots: True by default. If false, doesn't save plots at the end of the
                sweep
    """
    SetParam(SetArray[0])
    bridge.lockin_device.time_constant(time_constant)

    loop = qc.Loop(SetParam[SetArray], delay=SetDelay).each(*MeasParams)
    data = loop.get_data_set(name=DataName)
    plot = []

    def _plot_update():
        if type(plot) is list:
            for p in plot:
                p.update()
        else:
            plot.update()

    def _plot_save():
        if type(plot) is list:
            for i in range(len(plot)):
                fname = '{}_{}.png'.format(plot[i].get_default_title(), str(XParam[i])+'vs'+str(YParam[i]))
                plot[i].save(filename=fname)
        else:
            fname = '{}_{}.png'.format(plot.get_default_title(), str(XParam)+'vs'+str(*MeasParams))
            plot.save(filename=fname)

    if plot_results:
        if XParam is None:
            XParam = SetParam

        if len(MeasParams) == 1:
            plot = qc.QtPlot(getattr(data, str(XParam)+'_set'),
                             getattr(data, str(*MeasParams)),
                             window_title=str(XParam)+' vs. '+str(*MeasParams))
            loop.with_bg_task(plot.update)
        else:
            if YParam is None:
                YParam = MeasParams
            if type(XParam) is not list and type(XParam) is not tuple:
                if type(YParam) is not list and type(YParam) is not tuple:
                    XParam = [XParam]
                    YParam = [YParam]
                else:
                    XParam = [XParam]*len(MeasParams)
            elif len(XParam) != len(YParam):
                raise ValueError('length of XParam list must be the same as' +
                                 'length of YParam list')

            # Create a str for XParam so we can account for _set in the str
            XParamStr = []
            for i in range(len(XParam)):
                xpi = str(XParam[i])
                if xpi == str(SetParam):
                    XParamStr.append(xpi + '_set')
                else:
                    XParamStr.append(xpi)

            for i in range(len(YParam)):
                title = str(YParam[i]) + ' vs. ' + str(XParam[i])
                plot.append(qc.QtPlot(getattr(data, XParamStr[i]),
                            getattr(data, str(YParam[i])), window_title=title))

            loop.with_bg_task(_plot_update)
    try:
        loop.run()
        if save_plots and plot_results:
            _plot_save()
        return data, plot
    except KeyboardInterrupt:
        if plot_results:
            _plot_update()
            if save_plots:
                _plot_save()
        print('Keyboard Interrupt')
        return data, plot

def createLive2DPlotCapacitance(bridge, sensitivity, time_constant, initVolt1, initVolt2, SetParam1, SetArray1, SetParam2, SetArray2, *MeasParams, bruteForceTolerance=7e-7,
                     SetDelay1=0, SetDelay2=0, Param2_SetBetween=None,
                     DataName='', ZParam=None,
                     plot_results=True, save_plots=True, max_iteration=15):  
    """ Two parameter sweep, multiple measure. Includes live plot. Will balance the bridge before starting the sweep using the conditions for the first data point. Note: if the SetParam1
    array is nonuniform, the y axis of the plot will be messed up. Try MatPlot
    instead of QtPlot in that situation.

    Returns: data (a qcodes DataSet object), plot

    Arguments:
    bridge: The bridge object to measure
    sensitivity: Measurement sensitivity
    time_constant: Measurement time constant
    SetParam1: The outer parameter to sweep (such as a temperature)
    SetArray1: should be a list or numpy array of values you want to set
                SetParam1 to. This array will be run through once
    SetParam2: The inner parameter to sweep (such as a voltage)
    Param2_SetBetween: Sets parameter 2 to this value at the end of each
                sweep of the parameter (completion of one row) and before
                changing parameter 1.
    SetArray2: should be a list or numpy array of values you want to set
                SetParam2 to. This array will be run through for each value of
                SetArray1
    MeasParams: The parameter(s) you want to measure at each setpoint

    Keyword Arguments:
    SetDelay1: The delay time between when SetParam1 is set till the SetParam2
                is set to its first value (0 by default)
    SetDelay2: Delay time between when SetParam2 is set and the MeasParam
                is measured (0 by default)
    DataName: A name to tag the data (defaults to nothing)
    ZParam: Allows you to pick only a few parameters to plot out of those
                measured. (if not mentioned, will plot all *MeasParams)
    plot_results: True by default, if false, suppresses plotting
    save_plots: True by default. If false, doesn't save plots at the end of the
                sweep
    max_iteration: Maximum number of iterations for balancing (15 by default)
    """
    
    temp = []  # dummy parameter which do the same thing as usual if balancing is successful (i.e. balanced[0] is True), otherwise just reads 0 
    for _, param in enumerate(MeasParams):
        def skipLine(p=param): # don't remove p =param
            if not balanced[0]:
                return 0
            else:
                return p()
        temp.append(qc.Parameter(str(param) + "_", get_cmd=skipLine))

    def logging():
        print("Measure Using", "P1", SetParam1(), "P2", SetParam2())

    index = [0]
    balanced = [True]

    def between_func():
        if index[0] < len(SetArray1):
            SetParam1(SetArray1[index[0]])
            if not (Param2_SetBetween is None):
                SetParam2(Param2_SetBetween)
            else:
                SetParam2(SetArray2[0])

            time.sleep(0.5)
            print("Balance Using", "P1", SetParam1(), "P2", SetParam2())
        result = balance_voltage(bridge, sensitivity, time_constant, initVolt1=initVolt1, initVolt2=initVolt2, bruteForceTolerance=bruteForceTolerance, max_iteration=max_iteration)

        if result is None: # balancing failed
            balanced[0] = False
        else:
            balanced[0] = True

        if index[0] == 0:
            bridge.calculateBalancedCapacitanceAndDissipation()
        bridge.calculateOffbalanceVariations()
        
        index[0] +=1
        bridge.lockin_device.time_constant(time_constant)

    innerloop = qc.Loop(SetParam2[SetArray2],
                        delay=SetDelay2).each(*temp, qc.Task(logging))
    twodloop = qc.Loop(SetParam1[SetArray1],
                       delay=SetDelay1).each(qc.Task(between_func), innerloop)
    data = twodloop.get_data_set(name=DataName)
    plot = []

    def _plot_update():
        if type(plot) is list:
            for p in plot:
                p.update()
        else:
            plot.update()

    def _plot_save():
        if type(plot) is list:
            for i in range(len(plot)):
                fname = '{}_{}.png'.format(plot[i].get_default_title(), str(ZParam[i]))
                plot[i].save(filename=fname)
        else:
            fname = '{}_{}.png'.format(plot.get_default_title(), str(*MeasParams))
            plot.save(filename=fname)

    if plot_results:
        if len(MeasParams) == 1:
            plot = qc.QtPlot(getattr(data, str(*MeasParams)+"_"), window_title=str(*MeasParams))
            twodloop.with_bg_task(plot.update)
        else:
            if ZParam is None:
                ZParam = MeasParams
            if type(ZParam) is not list and type(ZParam) is not tuple:
                ZParam = [ZParam]

            for zp in ZParam:
                plot.append(qc.QtPlot(getattr(data, str(zp)+"_"), window_title=str(zp)))

            twodloop.with_bg_task(_plot_update)

    try:
        twodloop.run()
        if save_plots and plot_results:
            _plot_save()
        return data, plot
    except KeyboardInterrupt:
        if plot_results:
            _plot_update()
            if save_plots:
                _plot_save()
        print('Keyboard Interrupt')
        return data, plot