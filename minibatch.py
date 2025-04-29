import time

from PyLTSpice import SimRunner
from PyLTSpice import SpiceEditor
from spicelib import RawRead
import random
import numpy as np

'''XOR truth table, noting -2 represents 0 and 2 represents 1, these are voltages fed through into SPICE (ref pg 29)'''
truthTable = [[-2, -2, 0], [-2, 2, 1], [2, -2, 1], [2, 2, 0]]
LTC = SimRunner(output_folder='./temp')


'''Takes index for latest raw file, finds voltage drops across all resistors (weights)'''
def measureVoltages(index, path):
    freeVoltages = np.array([])
    time.sleep(2)

    raw_data = RawRead(f'./{path}{index}.raw')
    R1 = raw_data.get_trace('V(n003)').get_wave(0)[0] - raw_data.get_trace('V(n004)').get_wave(0)[0]
    R2 = raw_data.get_trace('V(n012)').get_wave(0)[0] - raw_data.get_trace('V(n004)').get_wave(0)[0]
    R3 = raw_data.get_trace('V(n015)').get_wave(0)[0] - raw_data.get_trace('V(n004)').get_wave(0)[0]
    R4 = raw_data.get_trace('V(n003)').get_wave(0)[0] - raw_data.get_trace('V(n006)').get_wave(0)[0]
    R5 = raw_data.get_trace('V(n012)').get_wave(0)[0] - raw_data.get_trace('V(n006)').get_wave(0)[0]
    R6 = raw_data.get_trace('V(n015)').get_wave(0)[0] - raw_data.get_trace('V(n006)').get_wave(0)[0]
    R7 = raw_data.get_trace('V(n001)').get_wave(0)[0] - raw_data.get_trace('V(n005)').get_wave(0)[0]
    R8 = raw_data.get_trace('V(n011)').get_wave(0)[0] - raw_data.get_trace('V(n005)').get_wave(0)[0]
    R9 = raw_data.get_trace('V(n001)').get_wave(0)[0] - raw_data.get_trace('V(n009)').get_wave(0)[0]
    R10 = raw_data.get_trace('V(n011)').get_wave(0)[0] - raw_data.get_trace('V(n009)').get_wave(0)[0]

    freeVoltages = np.append(freeVoltages, [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10])
    return freeVoltages

#does predictions for all cases and provides total loss for given circuit
def predictAll(name, truthTable, accessCounter):
    predictIT = SpiceEditor(f'./temp/{name}.net')
    loss = 0
    # set default arguments
    #predictIT.set_parameters(res=0, cap=100e-6)
    for x in range(4):
        predictIT.reset_netlist()
        predictIT.set_component_value(f'V1', f'{truthTable[x][0]}')
        predictIT.set_component_value(f'V2', f'{truthTable[x][1]}')

        predictIT.set_component_value(f'I1', 0)
        predictIT.set_component_value(f'I2', 0)

        LTC.run(predictIT)

        time.sleep(2)

        outPuts = getOutputs(('_' + str(x + accessCounter)), '/temp/' + name)

        print('prediction', outPuts[0] - outPuts[1], truthTable[x][2])

        loss += (truthTable[x][2] - outPuts[0] + outPuts[1])**2

        predictIT.reset_netlist()
    print('Loss', loss)

'''Provides voltages of output node to see prediction'''
def getOutputs(index, path):
   raw_data = RawRead(f'./{path}{index}.raw')
   p = raw_data.get_trace('V(n005)').get_wave(0)[0]
   n = raw_data.get_trace('V(n009)').get_wave(0)[0]
   return n, p #should be p, n reversed as using pretrained from paper


def train(name):
    #Setting Up Simulation
    path = f'./testfiles/{name}.net'
    mainPath = f'./temp/{name}_'

    netlist = SpiceEditor(path)
    # set default arguments
    netlist.set_parameters(res=0, cap=100e-6)

    '''Hyper parameters of B to scale current, and a for learning rate as stated in paper (ref pg 29)'''
    B = 0.001
    a = 0.001

    conduct = np.array([])

    '''Generates set of uniformallyish distributed conductances as inital weight values'''
    for x in range(1, 11):
        ohm = random.randint(11, 9999)
        conduct = np.append(conduct, [1/ohm])

    #overides auto generation with pretrained conductancecs
    conduct = np.array([0.0987166831194, 0.0859845227859, 0.0000126135964731, 0.0611995104039, 0.0494559841741, 0.0583430571762, 0.100200400802, 0.011961722488, 0.00564015792442, 0.114547537228])

    print('PRE-UPDATE CONDUCTANCES', 1/conduct)

    accessCounter = 5

    miniBatch = np.zeros(10)
    for batch in range(0, 1000):
        print(f'Epoch: {batch}')

        m = 50

        #random sample of XOR cases for training
        random_array = np.random.choice([0, 1, 2, 3], size=m, p=[0.25, 0.25, 0.25, 0.25])

        for ind, values in enumerate(truthTable):
            print(f"For entry: {values}")
            #Ensure circuit is reset between simulations
            netlist.reset_netlist()
            #Loads in weight values into simuation
            print("LOADING")
            for x in range(1, 11):
                '''Converts conductances (not supported in SPICE) to resistance with relation: 1/conductance = resistance'''
                ohm = 1/conduct[(x - 1)]
                print(x, ohm)
                netlist.set_component_value(f'R{x}', f'{ohm}')

            #Initalises free phase
            netlist.set_component_value(f'I1', 0)
            netlist.set_component_value(f'I2', 0)

            #Sets inputs
            netlist.set_component_value(f'V1', f'{values[0]}')
            netlist.set_component_value(f'V2', f'{values[1]}')
            '''Runs first simulation of training cycle'''
            LTC.run(netlist)

            '''Function returns array of voltage drops across all resistors'''
            freeVoltages1 = measureVoltages(accessCounter, mainPath)
            print("freeVoltages1", freeVoltages1)

            '''Returns voltages of output nodes, vOutp = positive output node, vOutn = negative output node'''
            vOutp, vOutn = getOutputs(accessCounter, mainPath)
            accessCounter += 1

            #nuged phase current for pre trained
            Ink = B * (values[2] - vOutn + vOutp)
            Ipk = B * (vOutn - vOutp - values[2])

            '''
            For normal operation 
            Ipk = B * (values[2] - vOutp + vOutn)
            Ink = B * (vOutp - vOutn - values[2])
            '''

            #Nudged phase, injects current
            netlist.set_component_value(f'I1', f'{Ipk}')
            netlist.set_component_value(f'I2', f'{Ink}')

            LTC.run(netlist)
            '''Calculates new voltage drops with circuit in nudged conditions'''
            nudgedVoltages = measureVoltages(accessCounter, mainPath)
            print("nudgedVoltages", nudgedVoltages)
            accessCounter += 1

            #Updates minibatch for each resistor according to equation 77
            miniBatch = miniBatch + (nudgedVoltages ** 2 - freeVoltages1 ** 2) * np.sum(random_array == ind)

            netlist.reset_netlist()

        #Final conductance update according to equation 77
        print('Pre Update Conduct', conduct)
        conduct = conduct - (a/(m * B)) * miniBatch
        #Clips conductance, p.g. 29
        print('Pre Clip Conduct', conduct)
        conduct[conduct < 0.0000001] = 0.0000001
        print("Final conductances", conduct)
        miniBatch = np.zeros(10)
        #Finds total loss for minibatch
        print("Prediction for Epoch")
        predictAll(f'{name}_{accessCounter - 1}', truthTable, accessCounter)
        accessCounter += 4

    #Saves trained circuit
    netlist.write_netlist('./cool.net')

print("PREUPDATE PREDICTION")
predictAll("trainedt", truthTable, 1)
print("COMMENCING TRAINING")
train('trained')
