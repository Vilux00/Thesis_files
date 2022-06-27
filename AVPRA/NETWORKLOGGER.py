import networkx as nx
from SimPy import Simulation as Sim
import UTILS
from conf import TRIALS, RESULTS_DIR, LABELS
import time

# The iterations whose VLs have to be stored
tt = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]

class NetworkLogger(Sim.Process):
        
    def __init__(self, sim, logging_interval):
        Sim.Process.__init__(self, sim=sim)
        self.sim = sim
        self.interval = logging_interval
        self.LPNet = nx.Graph()
        self.LPStatesTuples = []
    
    def Run(self):
        i = 0
        start_time = time.time()
        while True:
            self.logCurrentState()
            print ("--- %i iterations completed in %fs ---" % (i, (time.time() - start_time)))
            i += 1
            yield Sim.hold, self, self.interval

    def logCurrentState(self):
        LPNodes = sorted(self.sim.LPNet.nodes(data=True), key=lambda x: x[0])

        # Actual VL belonging coefficients
        VLs = [[float(node[1]["agent"].VL[str(i)]) for i in LABELS] for node in LPNodes]

        # Add actual VL value to logs
        if self.sim.now() in tt:
            self.LPStatesTuples.append([self.sim.now(), VLs]) 

    def logTrialToFiles(self, id):   
        UTILS.storeAllToFile(self.LPStatesTuples, RESULTS_DIR, id)
        
 
