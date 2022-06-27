import csv
import sys

### Useful global variables ###

# Simulation parameters
# Max simulation time(number of iteration of the algorithm)
ITERATION_NUM = 30
# The number of simulations
TRIALS = 1

try:
    # Set files path and graph type
    LABELS_NAMES_FILE = sys.argv[1]
    LABELS_INIT_VALUES_FILE = sys.argv[2]
    EDGES_FILE = sys.argv[3]
    GRAPH_TYPE = sys.argv[4]   
    RESULTS_DIR = sys.argv[5]
except IndexError:
    print("Command line arguments not provided correctly")
    sys.exit(1)

# Creates the list of unique labels
with open(LABELS_NAMES_FILE, 'r') as f:
    csv_reader = csv.reader(f, delimiter=';')
    LABELS = [x.lstrip() for x in list(csv_reader)[0]]
