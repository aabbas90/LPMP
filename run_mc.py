import numpy as np
from lpmp_py import mc_solver, amc_solver
import argparse
import os
import pickle
import time
import matplotlib.pyplot as plt 
from matplotlib.widgets import MultiCursor

def get_edge_images(edge_array_1d):
    edge_distances = [1, 4, 16, 64, 128, 256]
    edge_sampling_intervals = [1, 1, 4, 4, 4, 4]
    edge_array_2d = []
    start_index_1d = 0
    for (i, (e_d, s_i)) in enumerate(zip(edge_distances, edge_sampling_intervals)):
        output_shape = [(num_rows - e_d) // s_i, num_cols // s_i]
        current_numel = np.prod(output_shape)
        end_index_1d = current_numel + start_index_1d
        edge_array_2d.append(edge_array_1d[start_index_1d:end_index_1d].reshape(output_shape))
        start_index_1d += current_numel

        output_shape = [num_rows // s_i, (num_cols - e_d) // s_i]
        current_numel = np.prod(output_shape)
        end_index_1d = current_numel + start_index_1d
        edge_array_2d.append(edge_array_1d[start_index_1d:end_index_1d].reshape(output_shape))
        start_index_1d += current_numel
    return edge_array_2d

def get_colored_edge_image(row_costs, col_costs):
    img = np.zeros((col_costs.shape[0], row_costs.shape[1], 3))
    img[:row_costs.shape[0], :row_costs.shape[1], 0] = row_costs
    img[:col_costs.shape[0], :col_costs.shape[1], 1] = col_costs
    img = 255.0 * (img - img.min()) / (img.max() - img.min())
    return img.astype(np.uint8)

parser = argparse.ArgumentParser(description='Multiway cut disk instance runner')
parser.add_argument('--path', metavar='path', type=str, help='Path to multiway cut instance file (.pkl)', 
                    default='/BS/ahmed_projects/work/data/multicut/non_partition_examples/fbszyplbof.pkl')
args = parser.parse_args()

input_path = args.path
print("Loading instance file")
instance = pickle.load(open(input_path, 'rb'))
print("Calling solvers")
node_costs = instance['node_costs']
edge_costs = instance['edge_costs']
start_time = time.time()

# Call multiway solver:
node_labels, edge_labels, solver_cost = mc_solver(node_costs, edge_costs)

end_time = time.time()
print(f" MWC Solver finished in: {end_time - start_time:.2f}secs. Solver cost: {solver_cost:.2f}")

start_time = time.time()
# Call AMC solver:
node_labels_amc, node_instance_ids_amc, edge_labels_amc, solver_cost_amc = amc_solver(node_costs, edge_costs)
end_time = time.time()
print(f"AMWC Solver finished in: {end_time - start_time:.2f}secs. Solver cost: {solver_cost_amc:.2f}")

# Transform node labels to 2D:
num_classes = node_costs.shape[1]
num_rows = 512
num_cols = 1024
node_costs = node_costs.transpose().reshape((num_classes, num_rows, num_cols))
node_labels = node_labels.transpose().reshape((num_classes, num_rows, num_cols))
node_labels_amc = node_labels_amc.transpose().reshape((num_classes, num_rows, num_cols))

# Transform edge labels to 2D: 
_, _, edge_only_costs = zip(*edge_costs)
edge_only_costs = np.array(list(edge_only_costs))
edge_costs_2d = get_edge_images(edge_only_costs)
edge_labels_amc_2d = get_edge_images(edge_labels_amc)

# Visualization:
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex= True, sharey=True)
ax1.imshow(node_costs.argmin(0), cmap = 'tab20', interpolation='nearest')
ax1.set_title('Argmin node costs')

ax2.imshow(node_labels_amc.argmax(0), cmap = 'tab20', interpolation='nearest')
ax2.set_title('Node labels AMC')

ax3.imshow(node_labels.argmax(0), cmap = 'tab20', interpolation='nearest')
ax3.set_title('Node labels MC')

ax4.imshow(get_colored_edge_image(-1.0 * edge_labels_amc_2d[0], -1.0 * edge_labels_amc_2d[1]), interpolation='nearest')
ax4.set_title('Edge labels AMC, dist=1 (zoom-in req.)')

ax5.imshow(edge_costs_2d[0], cmap = 'gray', interpolation='nearest', vmax = 10.0, vmin = -5.0)
ax5.set_title('Row edges cost, dist=1')
ax6.imshow(edge_costs_2d[1], cmap = 'gray', interpolation='nearest', vmax = 10.0, vmin = -5.0)
ax6.set_title('Col edges cost, dist=1')

multi = MultiCursor(fig.canvas, (ax1, ax2, ax3, ax4, ax5, ax6), color='r', lw=1, horizOn = True)

plt.show()