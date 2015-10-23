from mixedlogistic_separate.dataSimulators import *


dxm = 0
dxr = 0

simulator = MixedLogisticDataSimulator(5000, dxm, dxr, c=3)
simulator.simulate(25)

