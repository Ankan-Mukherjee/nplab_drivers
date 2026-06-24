""" Copied from the triton model in the Oxford instrument folder, with
some variations and restrictions specifically applicable to our Triton.

This has an extra magnetic field sweep protection that disallows sweeping when
the temperature is too high"""

import configparser
import re
from functools import partial
import logging
from traceback import format_exc

from qcodes import IPInstrument
from qcodes.validators import Enum, Ints, Numbers

from time import sleep, monotonic
import numpy as np


def parse_outp_bool(value):
    if type(value) is float:
        value = int(value)
    elif type(value) is str:
        value = value.lower()

    if value in {1, 'on', True}:
        return 1
    elif value in {0, 'off', False}:
        return 0
    else:
        print(value)
        raise ValueError('Must be boolean, on or off, 0 or 1, True or False')


def parse_inp_bool(value):
    if type(value) is float:
        value = int(value)
    elif type(value) is str:
        value = value.lower()

    if value in {1, 'on', True}:
        return 'ON'
    elif value in {0, 'off', False}:
        return 'OFF'
    else:
        print(value)
        raise ValueError('Must be boolean, on or off, 0 or 1, True or False')


boolcheck = (0, 1, 'on', 'off', 'ON', 'OFF', False, True)


class Triton(IPInstrument):
    r"""
    Triton Driver

    Args:
        address: IP or host address for the ethernet connection
        port: the connection port number

        TODO:
        fetch registry directly from fridge-computer
    """

    def __init__(self, name, address, port, timeout=20, **kwargs):
        super().__init__(name, address=address, port=port,
                         terminator='\r\n', timeout=timeout, **kwargs)

        self._heater_range_auto = False
        self._heater_range_temp = [0.03, 0.1, 0.3, 1, 12, 40]
        self._heater_range_curr = [0.316, 1, 3.16, 10, 31.6, 100]
        self._control_channel = 5
        self._first_magnet_use = False

        self.add_parameter(name='time',
                           label='System Time',
                           get_cmd='READ:SYS:TIME',
                           get_parser=self._parse_time)

        self.add_parameter(name='action',
                           label='Current action',
                           get_cmd='READ:SYS:DR:ACTN',
                           get_parser=self._parse_action)

        self.add_parameter(name='status',
                           label='Status',
                           get_cmd='READ:SYS:DR:STATUS',
                           get_parser=self._parse_status)

        self.add_parameter(name='pid_control_channel',
                           label='PID control channel',
                           get_cmd=self._get_control_channel,
                           set_cmd=self._set_control_channel,
                           vals=Ints(1, 16))

        self.add_parameter(name='pid_mode',
                           label='PID Mode',
                           get_cmd=partial(self._get_control_param, 'MODE'),
                           get_parser=parse_outp_bool,
                           set_cmd=partial(self._set_control_param, 'MODE'),
                           set_parser=parse_inp_bool,
                           vals=Enum(*boolcheck))

        self.add_parameter(name='pid_ramp',
                           label='PID ramp enabled',
                           get_cmd=partial(self._get_control_param,
                                           'RAMP:ENAB'),
                           get_parser=parse_outp_bool,
                           set_cmd=partial(self._set_control_param,
                                           'RAMP:ENAB'),
                           set_parser=parse_inp_bool,
                           vals=Enum(*boolcheck))

        self.add_parameter(name='pid_setpoint',
                           label='PID temperature setpoint',
                           unit='K',
                           get_cmd=partial(self._get_control_param, 'TSET'),
                           set_cmd=partial(self._set_control_param, 'TSET'))

        self.add_parameter(name='pid_rate',
                           label='PID ramp rate',
                           unit='K/min',
                           get_cmd=partial(self._get_control_param,
                                           'RAMP:RATE'),
                           set_cmd=partial(self._set_control_param,
                                           'RAMP:RATE'))

        self.add_parameter(name='pid_range',
                           label='PID heater range',
                           # TODO: The units in the software are mA, how to
                           # do this correctly?
                           unit='mA',
                           get_cmd=partial(self._get_control_param, 'RANGE'),
                           set_cmd=partial(self._set_control_param, 'RANGE'),
                           vals=Enum(*self._heater_range_curr))

        self.add_parameter(name='magnet_status',
                           label='Magnet status',
                           get_cmd=partial(self._get_control_B_param, 'ACTN'))

        self.add_parameter(name='magnet_sweeprate',
                           label='Magnet sweep rate',
                           unit='T/min',
                           get_cmd=partial(
                               self._get_control_B_param, 'RVST:RATE'),
                           set_cmd=partial(self._set_control_magnet_sweeprate_param))
                           
        self.add_parameter(name='magnet_sweeprate_insta',
                           label='Instantaneous magnet sweep rate',
                           unit='T/min',
                           get_cmd=partial(self._get_control_B_param, 'RFST'))

        self.add_parameter(name='magnet_swh',
                           label='Magnet persistent switch heater',
                           set_cmd=self._set_swh,
                           get_cmd='READ:SYS:VRM:SWHT',
                           get_parser=self._parse_swh,
                           vals=Enum(*boolcheck))

        self.add_parameter(name='magnet_POC',
                           label='Persistent after completing sweep?',
                           set_cmd='SET:SYS:VRM:POC:{}',
                           set_parser=parse_inp_bool,
                           get_cmd='READ:SYS:VRM:POC',
                           get_parser=self._parse_state,
                           vals=Enum(*boolcheck))

        self.add_parameter(name='B',
                           label='Magnetic field',
                           unit='T',
                           get_cmd=partial(self._get_control_B_param, 'VECT'))

        self.add_parameter(name='sweep_Bx',
                           label='Magnetic field x-component',
                           unit='T',
                           get_cmd=partial(
                               self._get_Bx),
                           set_cmd=self._set_Bx_stable)
        self.add_parameter(name='sweep_By',
                           label='Magnetic field y-component',
                           unit='T',
                           get_cmd=partial(
                               self._get_By),
                           set_cmd=partial(self._set_By_stable))
                           
        self.add_parameter(name='sweep_Bz',
                           label='Magnetic field z-component',
                           unit='T',
                           get_cmd=partial(
                               self._get_Bz),
                           set_cmd=partial(self._set_Bz_stable))
                           
        self.add_parameter(name='field',
                           label='B',
                           unit='T',
                           get_cmd=self._get_field)

        self.add_parameter(name='field_Bx',
                           label='B',
                           unit='T',
                           get_cmd=self._get_field_Bx,
                           set_cmd=partial(self._set_field_return_Bx))
                         
        self.add_parameter(name='field_By',
                           label='B',
                           unit='T',
                           get_cmd=self._get_field_By,
                           set_cmd=partial(self._set_field_return_By))
                           
        self.add_parameter(name='field_Bz',
                           label='B',
                           unit='T',
                           get_cmd=self._get_field_Bz,
                           set_cmd=partial(self._set_field_return_Bz))

        self.add_parameter(name='field_set_stable',
                           label='B',
                           unit='T',
                           get_cmd=self._get_field,
                           set_cmd=partial(self._set_field_stable))

        self.add_parameter(name='magnet_sweep_time',
                           label='Magnet sweep time',
                           unit='T/min',
                           get_cmd=partial(self._get_control_B_param, 'RVST:TIME'))

        self.add_parameter(name='MC_heater',
                           label='Mixing chamber heater power',
                           unit='uW',
                           get_cmd='READ:DEV:H1:HTR:SIG:POWR',
                           set_cmd='SET:DEV:H1:HTR:SIG:POWR:{}',
                           get_parser=self._parse_htr,
                           set_parser=float,
                           vals=Numbers(0, 300000))

        self.add_parameter(name='still_heater',
                           label='Still heater power',
                           unit='uW',
                           get_cmd='READ:DEV:H2:HTR:SIG:POWR',
                           set_cmd='SET:DEV:H2:HTR:SIG:POWR:{}',
                           get_parser=self._parse_htr,
                           set_parser=float,
                           vals=Numbers(0, 300000))

        self.add_parameter(name='turbo_speed',
                           unit='Hz',
                           get_cmd='READ:DEV:TURB1:PUMP:SIG:SPD',
                           get_parser=self._parse_pump_speed)

        self.chan_alias = {'MC': 'T8', 'MC_cernox': 'T5', 'still': 'T3',
                           'cold_plate': 'T4', 'magnet': 'T13', 'PT2h': 'T1',
                           'PT2p': 'T2', 'PT1h': 'T6', 'PT1p': 'T7'}
        self._get_named_temp_channels()
        self._get_temp_channels()
        self._get_pressure_channels()
        self._get_valve_channels()
        self._get_pump_channels()

        self.connect_message()
        self._init_precool_thresholds()
        self.state='DEFAULT'

    # def set_B(self, x, y, z, s):
    #     if 0 < s <= 0.205:
    #         self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
    #                    ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']\r\n')
    #         self.write('SET:SYS:VRM:ACTN:RTOS\r\n')
    #         t_wait = self.magnet_sweep_time() * 60 + 10
    #         print('Please wait ' + str(t_wait) +
    #               ' seconds for the field sweep...')
    #         sleep(t_wait)
    #     else:
    #         print('Warning: set magnet sweep rate in range (0 , 0.205] T/min')

    def read_valves(self):
        for i in range(1, 10):
            print('V{}:  {}'.format(i, getattr(self, 'V%d' % i)()))

    def read_pumps(self):
        print('Turbo: {},  speed: {} Hz'.format(self.turbo(), self.turbo_speed()))
        print('KNF: {}'.format(self.knf()))
        print('Forepump: {}'.format(self.forepump()))

    def read_temps(self):
        for i in self.chan_alias:
            stat = 'off'
            if getattr(self, i+'_temp_enable')() == 0:
                stat = 'off'
            elif getattr(self, i+'_temp_enable')() == 1:
                stat = 'on'
            else:
                print('Temp reading status not determined')
            print('{} - {}:  {} K'.format(i, stat, getattr(self, self.chan_alias[i])()))

    def read_pressures(self):
        for i in range(1,6):
            print('P{}:  {}'.format(i, getattr(self, 'P'+str(i))()))

        print('POVC:  {}'.format(getattr(self, 'POVC')()))

    def tempdisable_excMC_magnet(self):
        for i in self.chan_alias:
            if i not in ('MC', 'magnet'):
                getattr(self, i + '_temp_enable')('off')

    def tempdisable_excMC(self):
        for i in self.chan_alias:
            if i != 'MC':
                getattr(self, i + '_temp_enable')('off')

    def alltempsenable(self):
        for i in self.chan_alias:
            getattr(self, i + '_temp_enable')('on')

    def magnet_hold(self):
        """Stop any sweeps"""
        self.write('SET:SYS:VRM:ACTN:HOLD')

    def _get_control_B_param(self, param):
        cmd = 'READ:SYS:VRM:{}'.format(param)
        return self._get_response_value(self.ask(cmd))

    # def _get_control_Bcomp_param(self, param):
    #     cmd = 'READ:SYS:VRM:{}'.format(param)
    #     return self._get_response_value(self.ask(cmd[:-2]) + cmd[-2:])

    def _get_field(self):
        field = self.ask('READ:SYS:VRM:VECT')
        Bx, By, Bz = field.split(' ')
        
        Bx = float(Bx.strip("STAT:SYS:VRM:VECT:[").strip('T]'))
        By = float(By.strip("T"))
        Bz = float(Bz.strip('T]'))

        return (Bx, By, Bz)
    
    def _get_field_Bx(self):
        field = self.ask('READ:SYS:VRM:VECT')
        Bx, By, Bz = field.split(' ')
        
        Bx = float(Bx.strip("STAT:SYS:VRM:VECT:[").strip('T]'))
        By = float(By.strip("T"))
        Bz = float(Bz.strip('T]'))

        return Bx
    
    def _get_field_By(self):
        field = self.ask('READ:SYS:VRM:VECT')
        Bx, By, Bz = field.split(' ')
        
        Bx = float(Bx.strip("STAT:SYS:VRM:VECT:[").strip('T]'))
        By = float(By.strip("T"))
        Bz = float(Bz.strip('T]'))

        return By
    
    def _get_field_Bz(self):
        field = self.ask('READ:SYS:VRM:VECT')
        Bx, By, Bz = field.split(' ')
        
        Bx = float(Bx.strip("STAT:SYS:VRM:VECT:[").strip('T]'))
        By = float(By.strip("T"))
        Bz = float(Bz.strip('T]'))

        return Bz
        
    def _get_response(self, msg):
        return msg.split(':')[-1]

    def _get_response_value(self, msg):  #TODO need to correct this to make it more readable and include NPERS and PERS, HOLD, SAFE, etc.
        msg = self._get_response(msg)
        if msg.endswith('NOT_FOUND'):
            return None
        elif msg.endswith('IDLE'):
            return 'IDLE'
        elif msg.endswith('RTOS'):
            return 'RTOS'
        # elif msg.endswith('Bx'):
        #     return float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[0])
        # elif msg.endswith('By'):
        #     return float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[1])
        # elif msg.endswith('Bz'):
        #     return float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[2])
        elif len(re.findall(r"[-+]?\d*\.\d+|\d+", msg)) > 1:
            return [float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[0]), float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[1]), float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[2])]
        try:
            return float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[0])
        except Exception:
            return msg

    def get_idn(self):
        """ Return the Instrument Identifier Message """
        idstr = self.ask('*IDN?')
        idparts = [p.strip() for p in idstr.split(':', 4)][1:]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def _get_control_channel(self, force_get=False):

        # verify current channel
        if self._control_channel and not force_get:
            tempval = self.ask(
                'READ:DEV:T{}:TEMP:LOOP:MODE'.format(self._control_channel))
            if not tempval.endswith('NOT_FOUND'):
                return self._control_channel

        # either _control_channel is not set or wrong
        for i in range(1, 17):
            tempval = self.ask('READ:DEV:T{}:TEMP:LOOP:MODE'.format(i))
            if not tempval.endswith('NOT_FOUND'):
                self._control_channel = i
                break
        return self._control_channel

    def _set_control_channel(self, channel):
        self._control_channel = channel
        self.write('SET:DEV:T{}:TEMP:LOOP:HTR:H1'.format(channel))

    def _get_control_param(self, param):
        chan = self._get_control_channel()
        cmd = 'READ:DEV:T{}:TEMP:LOOP:{}'.format(chan, param)
        return self._get_response_value(self.ask(cmd))

    def _set_control_param(self, param, value):
        chan = self._get_control_channel()
        cmd = 'SET:DEV:T{}:TEMP:LOOP:{}:{}'.format(chan, param, value)
        self.write(cmd)

    def _set_control_magnet_sweeprate_param(self, s):
        sweep_limit = 0.205 
        
        if 0 < s <= sweep_limit:
            x = round(self.field()[0], 4)
            y = round(self.field()[1], 4)
            z = round(self.field()[2], 4)
            self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
                       ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']\r\n')
        else:
            print('Warning: set sweeprate in range (0 , {}] T/min, not setting sweeprate'.format(sweep_limit))
                
## We don't have the vector magnet option.
    # def _set_control_Bx_param(self, x):
    #     s = self.magnet_sweeprate()
    #     y = round(self.By(), 4)
    #     z = round(self.Bz(), 4)
    #     self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
    #                ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']\r\n')
    #     self.write('SET:SYS:VRM:ACTN:RTOS\r\n')
    #     # just to give an time estimate, +10s for overhead
    #     t_wait = self.magnet_sweep_time() * 60 + 10
    #     print('Please wait ' + str(t_wait) + ' seconds for the field sweep...')
    #     while self.magnet_status() != 'IDLE':
    #         pass
    #
    # def _set_control_By_param(self, y):
    #     s = self.magnet_sweeprate()
    #     x = round(self.Bx(), 4)
    #     z = round(self.Bz(), 4)
    #     self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
    #                ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']\r\n')
    #     self.write('SET:SYS:VRM:ACTN:RTOS\r\n')
    #     # just to give an time estimate, +10s for overhead
    #     t_wait = self.magnet_sweep_time() * 60 + 10
    #     print('Please wait ' + str(t_wait) + ' seconds for the field sweep...')
    #     while self.magnet_status() != 'IDLE':
    #         pass

    def _set_field_stable(self, B):
        x, y, z = B
        if self._first_magnet_use is False:
            usecheck = input('Are you sure you want to use the magnet? [y/n]: ')
            if usecheck.lower() == 'y':
                self._first_magnet_use = True
                pass
            else:
                print('Magnet will not be used')
                return
        
        
        ## Turn this off for now. Just be cautious when using the magnet
        # maxtempHon8T = 4.87
        # maxtempHon0T = 4.6
        # maxtempHoff8T = 4.7
        # maxtempHoff0T = 4.3
        # magtemp = self.magnet_temp()
        # if self.magnet_swh():
        #     f = np.abs(self.field())
        #     if f < 0.4:
        #         condit_temp = maxtempHon0T + np.sqrt(0.02*f)
        #     else:
        #         p4temp = maxtempHon0T + np.sqrt(0.02*0.4)
        #         sl = (maxtempHon8T - p4temp)/(8-0.4)
        #         interc = p4temp - sl*0.4
        #         condit_temp = sl*f + interc
        # else:
        #     f = np.abs(self.field())
        #     if f < 0.4:
        #         condit_temp = maxtempHoff0T + np.sqrt(0.02*f)
        #     else:
        #         p4temp = maxtempHoff0T + np.sqrt(0.02*0.4)
        #         sl = (maxtempHoff8T - p4temp)/(8-0.4)
        #         interc = p4temp - sl*0.4
        #         condit_temp = sl*f + interc

        # while magtemp >= condit_temp:
        #     print('The magnet temperature is {:.4f} K. '.format(magtemp) +
        #           'Waiting for it to drop < {:.4f} K'.format(condit_temp))
        #     sleep(15)
        #     magtemp = self.magnet_temp()

        s = self.magnet_sweeprate()
        self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
                   ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']')
        self.write('SET:SYS:VRM:ACTN:RTOS')
        # just to give an time estimate, +10s for overhead
        # t_wait = self.magnet_sweep_time() * 60 + 10
        # print('Please wait ' + str(t_wait) + ' seconds for the field sweep, ' +
        #       'plus the time required for operating the switch...')
        while self.magnet_status() != 'IDLE':
            pass
    
    def _get_Bx(self):
        return self.field()[0]
        
    def _get_By(self):
        return self.field()[1]
    
    def _get_Bz(self):
        return self.field()[2]
        
    def _set_Bx_stable(self, Bx, holdZValue=None):
        x, y, z = self.field()
        y = round(y, 4)
        z = round(z, 4)
        
        if holdZValue != None: # sometimes the magnet fluctuates and Bz slowly ramps up during sweeps
            z = holdZValue
        self.field_set_stable((Bx, y, z))
    
    def _set_By_stable(self, By):
        x, y, z = self.field()
        x = round(x, 4)
        z = round(z, 4)
        
        self._set_field_stable((x, By, z))
        
    def _set_Bz_stable(self, Bz):
        x, y, z = self.field()
        x = round(x, 4)
        y = round(y, 4)
        
        self._set_field_stable((x, y, Bz))
        
    def _set_field_return_Bx(self, Bx):
        x, y, z = self.field()
        x = Bx
        if self._first_magnet_use is False:
            usecheck = input('Are you sure you want to use the magnet? [y/n]: ')
            if usecheck.lower() == 'y':
                self._first_magnet_use = True
                pass
            else:
                print('Magnet will not be used')
                return

        ## Turn this off for now. Just be cautious when using the magnet
        # maxtempHon8T = 4.87
        # maxtempHon0T = 4.6
        # maxtempHoff8T = 4.7
        # maxtempHoff0T = 4.3
        # magtemp = self.magnet_temp()
        # if self.magnet_swh():
        #     f = np.abs(self.field())
        #     if f < 0.4:
        #         condit_temp = maxtempHon0T + np.sqrt(0.02*f)
        #     else:
        #         p4temp = maxtempHon0T + np.sqrt(0.02*0.4)
        #         sl = (maxtempHon8T - p4temp)/(8-0.4)
        #         interc = p4temp - sl*0.4
        #         condit_temp = sl*f + interc
        # else:
        #     f = np.abs(self.field())
        #     if f < 0.4:
        #         condit_temp = maxtempHoff0T + np.sqrt(0.02*f)
        #     else:
        #         p4temp = maxtempHoff0T + np.sqrt(0.02*0.4)
        #         sl = (maxtempHoff8T - p4temp)/(8-0.4)
        #         interc = p4temp - sl*0.4
        #         condit_temp = sl*f + interc

        # while magtemp >= condit_temp:
        #     print('The magnet temperature is {:.4f} K. '.format(magtemp) +
        #           'Waiting for it to drop < {:.4f} K'.format(condit_temp))
        #     sleep(15)
        #     magtemp = self.magnet_temp()

        s = self.magnet_sweeprate()
        self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
                   ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']')
        self.write('SET:SYS:VRM:ACTN:RTOS')
        # just to give an time estimate, +10s for overhead
        # t_wait = self.magnet_sweep_time() * 60 + 10
        # print('Sweep time approximately ' + str(t_wait) + ' seconds')
        return
        
    def _set_field_return_By(self, By):
        x, y, z = self.field()
        y = By
        if self._first_magnet_use is False:
            usecheck = input('Are you sure you want to use the magnet? [y/n]: ')
            if usecheck.lower() == 'y':
                self._first_magnet_use = True
                pass
            else:
                print('Magnet will not be used')
                return

        ## Turn this off for now. Just be cautious when using the magnet
        # maxtempHon8T = 4.87
        # maxtempHon0T = 4.6
        # maxtempHoff8T = 4.7
        # maxtempHoff0T = 4.3
        # magtemp = self.magnet_temp()
        # if self.magnet_swh():
        #     f = np.abs(self.field())
        #     if f < 0.4:
        #         condit_temp = maxtempHon0T + np.sqrt(0.02*f)
        #     else:
        #         p4temp = maxtempHon0T + np.sqrt(0.02*0.4)
        #         sl = (maxtempHon8T - p4temp)/(8-0.4)
        #         interc = p4temp - sl*0.4
        #         condit_temp = sl*f + interc
        # else:
        #     f = np.abs(self.field())
        #     if f < 0.4:
        #         condit_temp = maxtempHoff0T + np.sqrt(0.02*f)
        #     else:
        #         p4temp = maxtempHoff0T + np.sqrt(0.02*0.4)
        #         sl = (maxtempHoff8T - p4temp)/(8-0.4)
        #         interc = p4temp - sl*0.4
        #         condit_temp = sl*f + interc

        # while magtemp >= condit_temp:
        #     print('The magnet temperature is {:.4f} K. '.format(magtemp) +
        #           'Waiting for it to drop < {:.4f} K'.format(condit_temp))
        #     sleep(15)
        #     magtemp = self.magnet_temp()

        s = self.magnet_sweeprate()
        self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
                   ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']')
        self.write('SET:SYS:VRM:ACTN:RTOS')
        # just to give an time estimate, +10s for overhead
        # t_wait = self.magnet_sweep_time() * 60 + 10
        # print('Sweep time approximately ' + str(t_wait) + ' seconds')
        return
        
    def _set_field_return_Bz(self, Bz):
        x, y, z = self.field()
        z = Bz
        if self._first_magnet_use is False:
            usecheck = input('Are you sure you want to use the magnet? [y/n]: ')
            if usecheck.lower() == 'y':
                self._first_magnet_use = True
                pass
            else:
                print('Magnet will not be used')
                return

        ## Turn this off for now. Just be cautious when using the magnet
        # maxtempHon8T = 4.87
        # maxtempHon0T = 4.6
        # maxtempHoff8T = 4.7
        # maxtempHoff0T = 4.3
        # magtemp = self.magnet_temp()
        # if self.magnet_swh():
        #     f = np.abs(self.field())
        #     if f < 0.4:
        #         condit_temp = maxtempHon0T + np.sqrt(0.02*f)
        #     else:
        #         p4temp = maxtempHon0T + np.sqrt(0.02*0.4)
        #         sl = (maxtempHon8T - p4temp)/(8-0.4)
        #         interc = p4temp - sl*0.4
        #         condit_temp = sl*f + interc
        # else:
        #     f = np.abs(self.field())
        #     if f < 0.4:
        #         condit_temp = maxtempHoff0T + np.sqrt(0.02*f)
        #     else:
        #         p4temp = maxtempHoff0T + np.sqrt(0.02*0.4)
        #         sl = (maxtempHoff8T - p4temp)/(8-0.4)
        #         interc = p4temp - sl*0.4
        #         condit_temp = sl*f + interc

        # while magtemp >= condit_temp:
        #     print('The magnet temperature is {:.4f} K. '.format(magtemp) +
        #           'Waiting for it to drop < {:.4f} K'.format(condit_temp))
        #     sleep(15)
        #     magtemp = self.magnet_temp()

        s = self.magnet_sweeprate()
        self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
                   ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']')
        self.write('SET:SYS:VRM:ACTN:RTOS')
        # just to give an time estimate, +10s for overhead
        # t_wait = self.magnet_sweep_time() * 60 + 10
        # print('Sweep time approximately ' + str(t_wait) + ' seconds')
        return

    def _set_swh(self, val):
        val = parse_inp_bool(val)
        if val == 'ON':
            self.write('SET:SYS:VRM:ACTN:NPERS')
            print('Wait 5 min for the switch to warm')
            sleep(10)
            while self.magnet_status() != 'IDLE':
                pass
        elif val == 'OFF':
            self.write('SET:SYS:VRM:ACTN:PERS')
            print('Wait 5 min for the switch to cool')
            sleep(10)
            while self.magnet_status() != 'IDLE':
                pass
        else:
            raise ValueError('Should be a boolean value (ON, OFF)')

    def _get_named_temp_channels(self):
        for al in tuple(self.chan_alias):
            chan = self.chan_alias[al]
            self.add_parameter(name=al+'_temp',
                               unit='K',
                               get_cmd='READ:DEV:%s:TEMP:SIG:TEMP' % chan,
                               get_parser=self._parse_temp)
            self.add_parameter(name=al+'_temp_enable',
                               get_cmd='READ:DEV:%s:TEMP:MEAS:ENAB' % chan,
                               get_parser=self._parse_state,
                               set_cmd='SET:DEV:%s:TEMP:MEAS:ENAB:{}' % chan,
                               set_parser=parse_inp_bool,
                               vals=Enum(*boolcheck))
            if al == 'MC':
                self.add_parameter(name='MC_Res',
                                   unit='Ohms',
                                   get_cmd='READ:DEV:%s:TEMP:SIG:RES' % chan,
                                   get_parser=self._parse_res)

    def _get_pressure_channels(self):
        self.chan_pressure = []
        for i in range(1, 6):
            chan = 'P%d' % i
            self.chan_pressure.append(chan)
            self.add_parameter(name=chan,
                               unit='mbar',
                               get_cmd='READ:DEV:%s:PRES:SIG:PRES' % chan,
                               get_parser=self._parse_pres)

        chan = 'P6'
        self.chan_pressure.append('POVC')
        self.add_parameter(name='POVC',
                           unit='mbar',
                           get_cmd='READ:DEV:%s:PRES:SIG:PRES' % chan,
                           get_parser=self._parse_pres)
        self.chan_pressure = set(self.chan_pressure)

    def _get_valve_channels(self):
        self.chan_valves = []
        for i in range(1, 10):
            chan = 'V%d' % i
            self.chan_valves.append(chan)
            self.add_parameter(name=chan,
                               get_cmd='READ:DEV:%s:VALV:SIG:STATE' % chan,
                               set_cmd='SET:DEV:%s:VALV:SIG:STATE:{}' % chan,
                               get_parser=self._parse_valve_state,
                               vals=Enum('OPEN', 'CLOSE', 'TOGGLE'))
        self.chan_valves = set(self.chan_valves)

    def _get_pump_channels(self):
        self.chan_pumps = ['turbo', 'knf', 'forepump']
        self.add_parameter(name='turbo',
                           get_cmd='READ:DEV:TURB1:PUMP:SIG:STATE',
                           set_cmd='SET:DEV:TURB1:PUMP:SIG:STATE:{}',
                           get_parser=self._parse_state,
                           set_parser=parse_inp_bool,
                           vals=Enum(*boolcheck))
        self.add_parameter(name='knf',
                           get_cmd='READ:DEV:COMP:PUMP:SIG:STATE',
                           set_cmd='SET:DEV:COMP:PUMP:SIG:STATE:{}',
                           get_parser=self._parse_state,
                           set_parser=parse_inp_bool,
                           vals=Enum(*boolcheck))
        self.add_parameter(name='forepump',
                           get_cmd='READ:DEV:FP:PUMP:SIG:STATE',
                           set_cmd='SET:DEV:FP:PUMP:SIG:STATE:{}',
                           get_parser=self._parse_state,
                           set_parser=parse_inp_bool,
                           vals=Enum(*boolcheck))
        self.chan_pumps = set(self.chan_pumps)

    def _get_temp_channels(self):
        self.chan_temps = []
        for i in range(1, 17):
            chan = 'T%d' % i
            self.chan_temps.append(chan)
            self.add_parameter(name=chan,
                               unit='K',
                               get_cmd='READ:DEV:%s:TEMP:SIG:TEMP' % chan,
                               get_parser=self._parse_temp)
            self.add_parameter(name=chan+'_enable',
                               get_cmd='READ:DEV:%s:TEMP:MEAS:ENAB' % chan,
                               get_parser=self._parse_state,
                               set_cmd='SET:DEV:%s:TEMP:MEAS:ENAB:{}' % chan,
                               set_parser=parse_inp_bool,
                               vals=Enum(*boolcheck))
        self.chan_temps = set(self.chan_temps)

    def fullcooldown(self):
        "Starts the full cooldown automation"
        self.write('SET:SYS:DR:ACTN:CLDN')

    def condense(self):
        "Starts condensing (use only if < about 15K)"
        self.write('SET:SYS:DR:ACTN:COND')

    def mixture_collect(self):
        "Starts collecting the mixture into the tank"
        self.write('SET:SYS:DR:ACTN:COLL')

    def precool(self):
        "Starts a pre-cool (doesn't continue to the next step automatically)"
        self.write('SET:SYS:DR:ACTN:PCL')

    def pause_precool(self):
        "Pauses the pre-cool automation"
        self.write('SET:SYS:DR:ACTN:PCOND')

    def resume_precool(self):
        "Resumes the pre-cool automation"
        self.write('SET:SYS:DR:ACTN:RCOND')

    def stopcool(self):
        "Stops any running automation"
        self.write('SET:SYS:ACTN:STOP')

    def _parse_action(self, msg):
        """ Parse message and return action as a string

        Args:
            msg (str): message string
        Returns
            action (str): string describing the action
        """
        action = msg[17:]
        if action == 'PCL':
            action = 'Precooling'
        elif action == 'EPCL':
            action = 'Empty precool loop'
        elif action == 'COND':
            action = 'Condensing'
        elif action == 'NONE':
            if self.MC_temp.get() < 2:
                action = 'Circulating'
            else:
                action = 'Idle'
        elif action == 'COLL':
            action = 'Collecting mixture'
        else:
            action = 'Unknown'
        return action

    def _parse_status(self, msg):
        return msg[19:]

    def _parse_time(self, msg):
        return msg[14:]

    def _parse_temp(self, msg):
        if 'NOT_FOUND' in msg:
            return None
        return float(msg.split('SIG:TEMP:')[-1].strip('K'))

    def _parse_pres(self, msg):
        if 'NOT_FOUND' in msg:
            return None
        return float(msg.split('SIG:PRES:')[-1].strip('mB'))

    def _parse_state(self, msg):
        if 'NOT_FOUND' in msg:
            return None
        state = msg.split(':')[-1].strip()
        return parse_outp_bool(state)

    def _parse_valve_state(self, msg):
        if 'NOT_FOUND' in msg:
            return None
        state = msg.split(':')[-1].strip()
        return state

    def _parse_pump_speed(self, msg):
        if 'NOT_FOUND' in msg:
            return None
        return float(msg.split('SIG:SPD:')[-1].strip('Hz'))

    def _parse_res(self, msg):
        if 'NOT_FOUND' in msg:
            return None
        return float(msg.split(':')[-1].strip('Ohm'))

    def _parse_swh(self, msg):
        if 'NOT_FOUND' in msg:
            return None
        elif msg.split(' ')[-1].strip(']') == 'ON':
            return 1
        elif msg.split(' ')[-1].strip(']') == 'OFF':
            return 0
        else:
            print('unknown switch heater state')
            return msg

    def _parse_htr(self, msg):
        if 'NOT_FOUND' in msg:
            return None
        return float(msg.split('SIG:POWR:')[-1].strip('uW'))

    def _recv(self):
        return super()._recv().rstrip()


    def cooldown_default(self):
        print('Starting Default state')
        self._set_many([
            ("turbo", 0),
            ("forepump", 0)
        ])
        for i in range(1,10):
            self._set_many([(f'V{i}',"CLOSE")])
        sleep(5)
        self.state = 'DEFAULT'
        print('In Default state')

    def pressurize_precool(self):
        if self.state!='DEFAULT':
            self.cooldown_default()
        print('Precool Pressurizing')
        if getattr(self, "P4")()<110:
            for valve_to_open in ["V7","V8","V2","V4"]:
                self._set_many([(valve_to_open,"OPEN")])
            self._wait_until("P4", max=110)
            for valve_to_close in ["V4","V7","V8","V2"]:
                self._set_many([(valve_to_close,"CLOSE")])
        sleep(10)
        self._set_many([
            ("V5", "OPEN"),
            ("V2", "OPEN"),
        ])
        self._set_many([("forepump",1)])
        self._wait_until("P2", max=1800)
        self._set_many([("forepump",0)])
        self._set_many([
            ("V5", "CLOSE"),
            ("V3", "OPEN"),
            ("V2", "OPEN")
        ])
        self.state='PRECOOL'
        self._init_precool_thresholds()
        print('Precool Pressurized')
    
    def manual_precool(self):
        if self.state!='PRECOOL':
            self.pressurize_precool()
        print('Precooling')
        T0 = self.T8()
        if T0<=5:
            print('Precool not necessary')
            return
        for ind, T, P in self.precool_thresholds:
            if T0>T:
                self.precool_step = ind
                self.target_P2 = P*1000
                self.target_T = 5 if ind==31 else self.precool_thresholds[ind][1]
                break
        while(True):
            print(self.precool_thresholds[ind])
            sleep(8)
            self._do_until([('V7','OPEN'),('V7','CLOSE')],'P2',min=self.target_P2,poll=5,delay=0.5)
            self._wait_until('T8', min=self.target_T, timeout=30000)
            ind+=1
            self.target_P2 = self.precool_thresholds[ind][2]*1000
            self.target_T = 5 if ind==31 else self.precool_thresholds[ind][1]
            if self.T8()<5:
                break
        if self.T8()<5:
            print('Precool complete')
            return



    def _set_many(self,actions,delay=0.5):
        for name, value in actions:
            obj = getattr(self,name)
            obj.set(value)
            sleep(delay)


    def _wait_until(self, name, *, min=None, max=None,
                    timeout=300, poll=0.5):
        if min is None and max is None:
            raise ValueError("At least one of min or max must be specified")
        t0 = monotonic()
        while True:
            val = getattr(self, name)()
            if min is not None and val < min:
                return val
            if max is not None and val > max:
                return val
            if monotonic() - t0 > timeout:
                raise TimeoutError(f"Timeout waiting for {name}. Current value is {val}.")
            sleep(poll)
        
    def _do_until(self, actions, name, *, min=None, max=None,
                    timeout=300, poll=0.5, delay=0.5):
        if min is None and max is None:
            raise ValueError("At least one of min or max must be specified")
        t0 = monotonic()
        while True:
            val = getattr(self, name)()
            if min is not None and val < min:
                return val
            if max is not None and val > max:
                return val
            if monotonic() - t0 > timeout:
                raise TimeoutError(f"Timeout waiting for {name}. Current value is {val}.")
            sleep(poll)
            self._set_many(actions,delay)

    def _init_precool_thresholds(self):
        self.precool_thresholds = [
            (0, 290, 3.15),
            (1, 280, 3.10),
            (2, 270, 3.10),
            (3, 260, 3.00),
            (4, 250, 3.00),
            (5, 240, 3.00),
            (6, 230, 3.00),
            (7, 220, 2.93),
            (8, 210, 2.84),
            (9, 200, 2.76),
            (10, 190, 2.67),
            (11, 180, 2.60),
            (12, 170, 2.50),
            (13, 160, 2.40),
            (14, 150, 2.30),
            (15, 140, 2.20),
            (16, 130, 2.10),
            (17, 120, 2.00),
            (18, 110, 2.00),
            (19, 100, 2.00),
            (20, 90, 1.80),
            (21, 80, 1.80),
            (22, 70, 1.50),
            (23, 60, 1.50),
            (24, 52, 1.50),
            (25, 42, 1.40),
            (26, 35, 1.30),
            (27, 24, 1.00),
            (28, 19, 0.80),
            (29, 15, 0.50),
            (30, 12, 0.40),
            (31, 5, 0.30),
        ]
        self.precool_step=None
            



