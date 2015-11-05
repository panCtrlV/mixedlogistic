from mixedlogistic_separate.dataSimulators import *


"""
Simulation study to investigate the Fisher information for the model.
"""

dxm = 0
dxr = 0

simulator = MixedLogisticDataSimulator(5000, dxm, dxr, c=3)
simulator.simulate(25)

