import pickle
import os

STATE = "_LPStates"
# / + "log_trial_"
BASE = os.sep + "trial_"

"""
Creates a directory if not exists
"""
def createIfNotExist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

"""
Create a file named dir/_log_trial/{id}_LPStates.pickled
"""      
def makeFileNameLPState(dir, id):
    return dir + BASE + str(id) + STATE + ".pickled"
    
"""
Store LPStateTuples in a file identified by trial_id
"""
def storeAllToFile(LPStatesTuples, directory, trial_id):
    storeToFile(LPStatesTuples, makeFileNameLPState(directory, str(trial_id)))                         

"""
Store one item (e.g. state list or networkx graph) in a specific pickle file.
"""            
def storeToFile(stuff, filename, verbose=True):   
    filename = os.path.normcase(filename)   
    directory = os.path.dirname(filename)
    createIfNotExist(directory)
    
    f = open(filename, 'wb')
    pickle.dump(stuff, f)
    f.close()
    
    if verbose:
        total = len(stuff)
        print ("Written %i items to pickled binary file: %s" % (total, filename)) 
    return filename