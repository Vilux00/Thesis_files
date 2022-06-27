import os
from SimPy import Simulation as Sim
import NETWORKLOGGER
import csv
from conf import TRIALS, RESULTS_DIR, LABELS_INIT_VALUES_FILE, LABELS
 
class NetworkSimulation(Sim.Simulation):
    """
    Simulation support for agents in a complex network.
    Can run multiple fresh trials with the same input parameters. Writes system
    state evolution to file (states & network topologies)
    """
    def __init__(self, LPNet, LPAgent, maxTime):
        Sim.Simulation.__init__(self)
        self.LPNet = LPNet
        self.LPAgent = LPAgent
        self.until = maxTime 

    """
    Run a simulation for TRIALS trials
    """
    def runSimulation(self):
        print ("Starting simulation...")      
        for i in range(TRIALS):
            print ("--- Trial %i ---" % i)
            self.runTrial(i)
        print ("Simulation completed.")
     
    """
    Run a single trial
    """    
    def runTrial(self, id):   
        self.initialize()
                      
        print ("Set up LP agents...")              

        # Initialize agents
        for i in self.LPNet.nodes():
            agent = self.LPAgent.LPAgent((i, self, self.LPNet))
            self.LPNet.node[i]['agent'] = agent
            self.activate(agent, agent.Run())

        print ("Set up logging...")
        
        # Set up logging
        logging_interval = 1
        logger = NETWORKLOGGER.NetworkLogger(self, logging_interval)
        self.activate(logger, logger.Run(), prior=True)
        
        # Run simulation
        self.simulate(self.until)

        # Write log files
        logger.logTrialToFiles(id)