import numpy as np
from lpmp_py import mc_solver, amc_solver
import argparse
import os
import pickle
import time
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Multiway cut disk instance runner')
parser.add_argument('path', metavar='path', type=str, help='Path to multiway cut instance file (.pkl)')
args = parser.parse_args()

input_path = args.path
print("Loading instance file")
instance = pickle.load(open(input_path, 'rb'))
print("Calling solver")
node_costs = instance['node_costs']
edge_costs = instance['edge_costs']
start_time = time.time()

# AMC solver can also be called like:
# node_labels, _, edge_labels, solver_cost = amc_solver(node_costs, edge_costs)

# Call multiway solver:
node_labels, edge_labels, solver_cost = mc_solver(node_costs, edge_costs)

end_time = time.time()
print(f"Solver finished in {end_time - start_time:0f}secs. Solver cost: {solver_cost}")

# Transform node labels to 2D:
num_classes = node_costs.shape[1]
num_rows = 512
num_cols = 1024
node_costs = node_costs.transpose().reshape((num_classes, num_rows, num_cols))
node_labels = node_labels.transpose().reshape((num_classes, num_rows, num_cols))

# Transform edge labels to 2D: 
edge_distances = [1, 4, 16, 64, 128, 256]
edge_sampling_intervals = [1, 1, 4, 4, 4, 4]
edge_labels_2d = []
edge_costs_2d = []
_, _, edge_only_costs = zip(*edge_costs)
edge_only_costs = np.array(list(edge_only_costs))
start_index_1d = 0
for (i, (e_d, s_i)) in enumerate(zip(edge_distances, edge_sampling_intervals)):
    output_shape = [(num_rows - e_d) // s_i, num_cols // s_i]
    current_numel = np.prod(output_shape)
    end_index_1d = current_numel + start_index_1d
    edge_labels_2d.append(edge_labels[start_index_1d:end_index_1d].reshape(output_shape))
    edge_costs_2d.append(edge_only_costs[start_index_1d:end_index_1d].reshape(output_shape))
    start_index_1d += current_numel

    output_shape = [num_rows // s_i, (num_cols - e_d) // s_i]
    current_numel = np.prod(output_shape)
    end_index_1d = current_numel + start_index_1d
    edge_labels_2d.append(edge_labels[start_index_1d:end_index_1d].reshape(output_shape))
    edge_costs_2d.append(edge_only_costs[start_index_1d:end_index_1d].reshape(output_shape))
    start_index_1d += current_numel

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex= True, sharey=True)
ax1.imshow(node_costs.argmin(0), cmap = 'tab20', interpolation='nearest')
ax1.set_title('Argmin node costs')
ax2.imshow(node_labels.argmax(0), cmap = 'tab20', interpolation='nearest')
ax2.set_title('Argmax node labels')
ax3.imshow(edge_costs_2d[0], cmap = 'gray', interpolation='nearest', vmax = 5.0, vmin = -5.0)
ax3.set_title('Row edges cost, dist=1')
ax4.imshow(edge_costs_2d[1], cmap = 'gray', interpolation='nearest', vmax = 5.0, vmin = -5.0)
ax4.set_title('Col edges cost, dist=1')
plt.show()