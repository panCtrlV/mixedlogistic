from mixedlogistic_separate.DataSimulators import *


simulator = MixedLogisticDataSimulator(5000, 3, 3, 3)
simulator.simulate(25)

params = simulator.parameters
params.flattenHiddenLayerParameters()

print params
