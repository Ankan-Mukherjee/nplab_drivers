# import configparser
# import re
# from functools import partial
# import logging
# from traceback import format_exc

# from qcodes import IPInstrument
# from qcodes.validators import Enum, Ints, Numbers

# import time
# import numpy as np
# from qcodes.instrument_drivers.nplab_drivers.NPTriton import Triton

# class Cooldown(Triton):
#     def set_default(self):
#         delay = 0.5
#         self.fore_pump(0)
#         time.sleep(delay)
#         self.turbo(0)
#         time.sleep(delay)
#         for i in range(9):
#             setattr(self, f'V{i}','CLOSED')
#             time.sleep(delay)
#         time.sleep(delay*5)

#     def set_pressurize_precool(self):
#         self.set_default()
#         self.