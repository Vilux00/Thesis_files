"""
Module for the LPAgent class that can be subclassed by agents.
"""
from SimPy import Simulation as Sim
import csv
from conf import LABELS, GRAPH_TYPE

class LPAgent(Sim.Process):

    # Variables shared between all instances of this class
    TIMESTEP_DEFAULT = 1.0

    def __init__(self, initialiser, name='network_process'):   
        Sim.Process.__init__(self, name)
        self.initialize(*initialiser)
        
    def initialize(self, id, sim, LPNet):             
        self.id = id
        self.sim = sim
        self.LPNet = LPNet
        self.VL = {l: LPNet.nodes[id][l] for l in LABELS}

    """
    Start the agent execution
    it executes a state change then wait for the next step
    """
    def Run(self):
        while True:
            self.updateStep()
            yield Sim.hold, self, LPAgent.TIMESTEP_DEFAULT/2
            self.state_changing()
            yield Sim.hold, self, LPAgent.TIMESTEP_DEFAULT/2
           
    """
    Updates all the belonging coefficients
    """
    def state_changing(self):
        if GRAPH_TYPE == "U":
            neighbours = list(self.LPNet.neighbors(self.id))
        else:
            neighbours = list(self.LPNet.predecessors(self.id))
        neighboursSize = len(list(neighbours))

        for j in LABELS:
            ssum  = 0
            # Calculate the second term of the update rule
            for i in list(neighbours):
                ssum += float(self.LPNet.node[i][j]) / float(neighboursSize + 1)

            # Update str(j)'s belonging coefficient
            self.VL[j] = self.LPNet.node[self.id][j] / float(neighboursSize + 1) + ssum 

    """
    Update the VL used by other agents to update themselves
    """
    def updateStep(self):
        for j in LABELS:
            self.LPNet.node[self.id][j] = self.VL[j]               
