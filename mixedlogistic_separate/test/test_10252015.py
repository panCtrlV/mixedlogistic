from mixedlogistic_separate.dataSimulators import *


simulator = MixedLogisticDataSimulator(5000, 3, 3, 3)
simulator.simulate(25)
params = simulator.parameters
# print params

# test proper ordering of flatten method
flatHiddenParameterValues = params.flattenHiddenLayerParameters()
# print flatHiddenParameterValues
#
# # test getHiddenParametersFromFlatInput
# a, Alpha = params.getHiddenParametersFromFlatInput(flatHiddenParameterValues)
# print a
# print Alpha
#
# # test getObservedParametersFromFlatInput_j
# flatObservedParameterValues_group1 = params.flattenObservedLayerParametersj(0)
# print flatObservedParameterValues_group1
# b1, beta1 = params.getObservedParametersFromFlatInput_j(flatObservedParameterValues_group1)
# print b1
# print beta1

# compare Q functions and gradients for old and new flatten orders
data = simulator.data
f1, f2, g1, g2 = generateNegQAndGradientFunctions(data, params)

print f1(flatHiddenParameterValues)
# 324.02717253007876 with new flatten order
# 324.02717253007876 with old flatten order
print g1(flatHiddenParameterValues)
# with new flatten order
# [ -8.70493927  -2.29011511  11.11152495   1.45180871 -10.7422279
#   -6.46820163   3.59912219   9.94588311]
# with old flatten order
# [ -8.70493927 -10.7422279   -2.29011511  11.11152495   1.45180871
#   -6.46820163   3.59912219   9.94588311]
for i in range(3):
    print f2(params.getParameters(2, i+1), i)
# with new order
# 99.0807910159
# 628.083046038
# 1004.34363431
# with old order
# 99.0807910159
# 628.083046038
# 1004.34363431
for i in range(3):
    print g2(params.getParameters(2, i+1), i)
# with new order
# [-2.93917082 -4.46982209  7.76596323 -2.8103049 ]
# [-190.56471846  -27.0654797     1.5935964   -13.50550239]
# [-194.62063037   24.8841458   -81.27015453   13.28693948]
# with old order
# [-2.93917082 -4.46982209  7.76596323 -2.8103049 ]
# [-190.56471846  -27.0654797     1.5935964   -13.50550239]
# [-194.62063037   24.8841458   -81.27015453   13.28693948]

# compare trainEM for old and new orders
trainEM(data, )