# Exploring Reinforcement Learning Driven Cyber-Attacks on Intelligent Transportation Systems


Intelligent Transportation Systems (ITSs) are a combination of communication, computer and system technologies aiming at perfect decision making for transportation infrastructure. They create efficient and safe transportation networks. The advancement of connected vehicles, generating dynamic data through wireless communications enables ITSs to improve their efficiency especially in Traffic Signal Control (TSC) which is the backbone for traffic flow scheduling. However, wireless communications channels are vulnerable to various types of cyber-security attacks and can pose severe threats to dynamic TSC systems. Attackers may attempt to create higher latency traffic flow which leads to severe traffic congestion.

Deep Reinforcement Learning (RL) is a powerful technique that has been used to improve TSC systems performance in real-time environments such as ITSs. In this work, we aim at implementing an RL algorithm to create a Sybil attack, a cyber-security attack on TSC systems. Our goal is to exploit existing vulnerabilities in TSC systems and train our attack agent at a single intersection using Deep QNetwork (DQN) to launch a cyber-attack by sending falsified data. The results show that this attack leads to notable increase in vehicles’ average travel time and makes disastrous traffic congestion. Moreover, in crowded cities, this will give rise to serious problems such as more fuel consumption and air pollution.


Setup

This project was developed using Python 3. Install from Python.org

Prerequisite Packages :
1) Tensorflow (pip install tensorflow) or (pip install tensorflow-gpu)
2) Keras (pip install keras)

This simulatons were done on SUMO Traffic Simulator and it is available at: http://sumo.dlr.de/wiki/Downloads

For SUMO installation, please refer to the website.



Running

twoNetworks.py
