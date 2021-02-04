# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 21:20:36 2020

@author: Najmeh
"""



import numpy as np
import traci
import random
from collections import deque
import tensorflow.keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt


sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo"
sumoCmd = [sumoBinary, "-c", "C:/Users/RL/Sumo-GUI/crossactuated.sumocfg"]

class DQNAgent:
    # initialize the attribitues
    def __init__(self):
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.5  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.02
        self.memory = deque(maxlen=200)
        self.model = self._build_model()
        self.target_model = self._build_model()       
        self.action_size = 7
        self.tau = 0.05
        self.tau_num = 0

        
    def _build_model(self):
        model = Sequential()

        model.add(Dense(16, activation='relu', input_dim=6))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(7, activation='linear'))

        model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=self.learning_rate), loss='mse', metrics=['mse'])
        print(model.summary())
        return model
        
    # Agent sample <s,a,r,s'> to the replay memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def getaction(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        return np.argmax(self.model.predict(state)[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        self.tau_num += 1
        for state, action, reward, next_state, done in minibatch:
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else: 
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
                
            self.model.fit(state, target, epochs=1, verbose=0)

            
    def target_train(self):
        self.tau_num = 0
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
        

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def getstate():
        stateMatrix = []
        for i in range(6):
            stateMatrix.append(0)
    
        # number of vehicles in each edge of junction 420
        stateMatrix[0] = traci.edge.getLastStepVehicleNumber('1si')
        stateMatrix[1] = traci.edge.getLastStepVehicleNumber('2si')
    
        stateMatrix[2] = traci.edge.getLastStepVehicleNumber('3si')
        stateMatrix[3] = traci.edge.getLastStepVehicleNumber('4si')
    
        # when the light is green for east-west traffic
        if (traci.trafficlight.getPhase('0') == 0):
            stateMatrix[4] = 0
            stateMatrix[5] = (traci.trafficlight.getNextSwitch("0") - traci.simulation.getTime())
        # when the light is green for north-south traffic
        else:
            stateMatrix[4] = 1
            stateMatrix[5] = (traci.trafficlight.getNextSwitch("0") - traci.simulation.getTime())
    
        stateMatrix = np.array(stateMatrix)
        stateMatrix = np.reshape(stateMatrix, [1, 6])
    
        return stateMatrix


StartAddingVehichleStep = 5
episodes = 1000

batch_size = 32
xepisode = []
ywaiting = []
zvehicle_num = []
    
agent = DQNAgent()
history = []
    
for episode in range(episodes):
    
        stepz = 0
        a, b, c, d = 0, 0, 0, 0
        action = 0
        reward_stop = 0
        reward_move = 0
        waiting_time = 0
        reward = reward_stop - reward_move
        reward_total = 0
        
        added_vehlist = []
        insimulation_veh = []

        traci.start(sumoCmd)
        state = getstate()        
        
        if episode == 0 :
        
            while traci.simulation.getMinExpectedNumber() > 0 and stepz < 1200:
                stepz +=1
                if state[0][4] == 0:
                    reward_move = traci.edge.getLastStepVehicleNumber('1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')

                if state[0][4] == 1: 
                    reward_move = traci.edge.getLastStepVehicleNumber('3si') + traci.edge.getLastStepVehicleNumber('4si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si')
                traci.simulationStep()
                waiting_time += reward_stop
                reward = reward_stop - reward_move
                reward_total += reward
            traci.close()

        else:
            if episode % 20 == 0 and episode < 110:
                agent.epsilon = 0.5
            while traci.simulation.getMinExpectedNumber() > 0 and stepz < 1200:
                # print(routes)
                action = agent.getaction(state)
                #print(action, 'ACTION')
                
                # Add vehicles from East
                if action == 1 and state[0][4] == 0:
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                    if stepz >= StartAddingVehichleStep:
                        traci.vehicle.add('ve-%d' % (a), "horizontal")
                        traci.vehicle.setColor('ve-%d' % (a), (0, 0, 225, 255))
                        added_vehlist.append('ve-%d' % (a))
                        a += 1
                    traci.simulationStep()
                        
    
                if action == 1 and state[0][4] == 1:
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('3si') + traci.edge.getLastStepVehicleNumber('4si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si')
                    if stepz >= StartAddingVehichleStep:
                        traci.vehicle.add('ve-%d' % (a), "horizontal")
                        traci.vehicle.setColor('ve-%d' % (a), (0, 0, 225, 255))
                        added_vehlist.append('ve-%d' % (a))
                        a += 1
                    traci.simulationStep()
                  
                    # No vehicle to add
                if action == 0 and state[0][4] == 0:
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                    traci.simulationStep()        
                            
                if action == 0 and state[0][4] == 1:
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('3si') + traci.edge.getLastStepVehicleNumber('4si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si')                 
                    traci.simulationStep()    
                   
                    # Add vehicles from West
                if action == 2 and state[0][4] == 0:
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                    if stepz >= StartAddingVehichleStep:
                        traci.vehicle.add('vw-%d' % (b), "always_right")
                        traci.vehicle.setColor('vw-%d' % (b), (0, 0, 225, 255))
                        added_vehlist.append('vw-%d' % (b))
                        b += 1
                    traci.simulationStep()
    
                if action == 2 and state[0][4] == 1:
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('3si') + traci.edge.getLastStepVehicleNumber('4si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si')
                    if stepz >= StartAddingVehichleStep:
                        traci.vehicle.add('vw-%d' % (b), "always_right")
                        traci.vehicle.setColor('vw-%d' % (b), (0, 0, 225, 255))
                        added_vehlist.append('vw-%d' % (b))
                        b += 1
                    traci.simulationStep()
                                             
                    # Add vehicles from North
                if action == 3 and state[0][4] == 0:
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                    if stepz >= StartAddingVehichleStep:
                        traci.vehicle.add('vn-%d' % (c), "always_left")
                        traci.vehicle.setColor('vn-%d' % (c), (0, 0, 225, 255))
                        added_vehlist.append('vn-%d' % (c))
                        c += 1
                    traci.simulationStep()                          
                        
                if action == 3 and state[0][4] == 1:
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('3si') + traci.edge.getLastStepVehicleNumber('4si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si')
                    if stepz >= StartAddingVehichleStep:
                        traci.vehicle.add('vn-%d' % (c), "always_left")
                        traci.vehicle.setColor('vn-%d' % (c), (0, 0, 225, 255))
                        added_vehlist.append('vn-%d' % (c))
                        c += 1
                    traci.simulationStep()
                         
                    # Add vehicles from South
                if action == 4 and state[0][4] == 0:
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                    if stepz >= StartAddingVehichleStep:
                        traci.vehicle.add('vs-%d' % (d), "vertical")
                        traci.vehicle.setColor('vs-%d' % (d), (0, 0, 225, 255))
                        added_vehlist.append('vs-%d' % (d))
                        d += 1                  
                    traci.simulationStep()    
    
                if action == 4 and state[0][4] == 1:
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('3si') + traci.edge.getLastStepVehicleNumber('4si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si')
                    if stepz >= StartAddingVehichleStep:
                        traci.vehicle.add('vs-%d' % (d), "vertical")
                        traci.vehicle.setColor('vs-%d' % (d), (0, 0, 225, 255))
                        added_vehlist.append('vs-%d' % (d))
                        d += 1
                    traci.simulationStep()                       
                                              
                        
                    # Add one vehicles from East and one from West
                if action == 5 and state[0][4] == 0:
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                    if stepz >= StartAddingVehichleStep:
                        traci.vehicle.add('vew-%d' % (a), "horizontal")
                        traci.vehicle.setColor('vew-%d' % (a), (0, 0, 225, 255))
                        added_vehlist.append('vew-%d' % (a))
                        a += 1
                        traci.vehicle.add('vwe-%d' % (b), "always_right")
                        traci.vehicle.setColor('vwe-%d' % (b), (0, 0, 225, 255))
                        added_vehlist.append('vwe-%d' % (b))
                        b += 1
                    traci.simulationStep()
                        
                        
                if action == 5 and state[0][4] == 1: 
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('3si') + traci.edge.getLastStepVehicleNumber('4si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si')                   
                    if stepz >= StartAddingVehichleStep:
                        traci.vehicle.add('vew-%d' % (a), "horizontal")
                        traci.vehicle.setColor('vew-%d' % (a), (0, 0, 225, 255))
                        added_vehlist.append('vew-%d' % (a))
                        a += 1
                        traci.vehicle.add('vwe-%d' % (b), "always_right")
                        traci.vehicle.setColor('vwe-%d' % (b), (0, 0, 225, 255))
                        added_vehlist.append('vwe-%d' % (b))
                        b += 1
                    traci.simulationStep()     
    
                    # Add one vehicles from North and one from South
                if action == 6 and state[0][4] == 0:
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                    if stepz >= StartAddingVehichleStep:
                        traci.vehicle.add('vns-%d' % (c), "always_left")
                        traci.vehicle.setColor('vns-%d' % (c), (0, 0, 225, 255))
                        added_vehlist.append('vns-%d' % (c))
                        c += 1
                        traci.vehicle.add('vsn-%d' % (d), "vertical")
                        traci.vehicle.setColor('vsn-%d' % (d), (0, 0, 225, 255))
                        added_vehlist.append('vsn-%d' % (d))
                        d += 1                  
                    traci.simulationStep()   
                        
                if action == 6 and state[0][4] == 1:   
                    stepz +=1
                    reward_move = traci.edge.getLastStepVehicleNumber('3si') + traci.edge.getLastStepVehicleNumber('4si')
                    reward_stop = traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si')                
                    if stepz >= StartAddingVehichleStep:
                        traci.vehicle.add('vns-%d' % (c), "always_left")
                        traci.vehicle.setColor('vns-%d' % (c), (0, 0, 225, 255))
                        added_vehlist.append('vns-%d' % (c))
                        c += 1
                        traci.vehicle.add('vsn-%d' % (d), "vertical")
                        traci.vehicle.setColor('vsn-%d' % (d), (0, 0, 225, 255))
                        added_vehlist.append('vsn-%d' % (d))
                        d += 1
                    traci.simulationStep()
                        
                reward = reward_stop - reward_move
                waiting_time += reward_stop
                for i in range(4):
                    stepz += 1
                    if state[0][4] == 0:         
                        for v in added_vehlist:
                            if v in traci.vehicle.getIDList():
                                if v in traci.edge.getLastStepVehicleIDs('1si') or traci.edge.getLastStepVehicleIDs('2si'):
                                    reward += 1 
                                    insimulation_veh.append(v)
                                else:
                                    reward -= 1
                                    insimulation_veh.append(v)
                            elif v in insimulation_veh:
                                added_vehlist.remove(v)
                        reward_move = traci.edge.getLastStepVehicleNumber('1si') + traci.edge.getLastStepVehicleNumber('2si')
                        reward_stop = traci.edge.getLastStepVehicleNumber('3si') + traci.edge.getLastStepVehicleNumber('4si')
                        reward += reward_stop - reward_move
                        waiting_time += reward_stop
                    if state[0][4] == 1:
                        for v in added_vehlist:
                            if v in traci.vehicle.getIDList():
                                if v in traci.edge.getLastStepVehicleIDs('3si') or traci.edge.getLastStepVehicleIDs('4si'):
                                    reward += 1 
                                    insimulation_veh.append(v)
                                else:
                                    reward -= 1
                                    insimulation_veh.append(v)
                            elif v in insimulation_veh:
                                added_vehlist.remove(v)
                            added_vehlist.remove(v)
                        reward_stop = traci.edge.getLastStepVehicleNumber('1si') + traci.edge.getLastStepVehicleNumber('2si')
                        reward_move = traci.edge.getLastStepVehicleNumber('3si') + traci.edge.getLastStepVehicleNumber('4si')        
                        reward += reward_stop - reward_move
                        waiting_time += reward_stop
                    traci.simulationStep()                     
                        
                new_state = getstate()
                
    
                # Consider a cost function
                if action == 1 or action == 2 or action == 3 or action == 4 :
                    reward -= 4
                elif action == 5 or action == 6 :
                    reward -= 8
                

                reward_total += reward
                
                agent.remember(state, action, reward, new_state, False)
                # Randomly Draw 32 samples and train the neural network by RMS Prop algorithm
                if (len(agent.memory) > batch_size) :
                    agent.replay(batch_size)
                

                state = new_state
                if agent.tau_num == 10:
                    agent.target_train()                
                
            mem = agent.memory[-1]
            del agent.memory[-1]
            agent.memory.append((mem[0], mem[1], reward, mem[3], True))
        
        vehicle_num = a + b + c + d
        waiting_time = waiting_time
        xepisode.append(episode)
        ywaiting.append(waiting_time)
        plt.scatter(xepisode, ywaiting)
        plt.xlabel("Episodes")
        plt.ylabel("Waiting Time")
        
        zvehicle_num.append(vehicle_num)
        plt.scatter(xepisode, zvehicle_num)
        plt.xlabel("Episodes")
        plt.ylabel("Number of Vehicles Added")
        
        
        print('episode - ' + str(episode) + ' total vehicle number added - ' + str(vehicle_num) + 
              ' total waiting time - ' + str(waiting_time) + ' Total Reward - ' + str (reward_total))
        traci.close()

