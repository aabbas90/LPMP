import numpy as np
from lpmp_py import amc_solver
import argparse
import os
import pickle
import time

parser = argparse.ArgumentParser(description='AMC disk instance runner')
parser.add_argument('path', metavar='path', type=str, help='Path to AMC instance file (.pkl)')
args = parser.parse_args()

input_path = args.path
print("Loading instance file")
instance = pickle.load(open(input_path, 'rb'))
print("Calling solver")
node_costs = instance['node_costs']
edge_costs = instance['edge_costs']
start_time = time.time()
node_labels, node_instance_ids, edge_labels, solver_cost = amc_solver(node_costs, edge_costs)
end_time = time.time()
print(f"Solver finished in {end_time - start_time:0f}secs, number of components: {np.unique(node_instance_ids).shape[0]}")