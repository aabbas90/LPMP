import torch
from ..raw_solvers import amc_solver
import numpy as np 
import torch.multiprocessing as mp

def get_edge_indices(image_shape, edge_distances, edge_sampling_intervals):
    indices = np.arange(np.prod(image_shape)).reshape(image_shape).astype(np.int32)
    edge_indices = {}
    current_si = 1
    for (i, e_d) in enumerate(edge_distances):
        if edge_sampling_intervals is not None:
            current_si = edge_sampling_intervals[i]

        left_offset = int(e_d / 2.0)
        right_offset = left_offset
        if e_d == 1:
            left_offset = 0
            right_offset = 1

        valid_left_offset = int(np.ceil(left_offset / current_si))
        valid_right_offset = int(np.ceil(right_offset / current_si))
        output_right_offset = -valid_right_offset if valid_right_offset > 0 else None
        e1_row = np.meshgrid(np.arange(-left_offset, image_shape[0] - left_offset, current_si)[valid_left_offset:output_right_offset], 
                            np.arange(0, image_shape[1], current_si), indexing='ij')
        e1_row = np.ravel_multi_index(e1_row, dims = image_shape)

        e2_row = np.meshgrid(np.arange(right_offset, image_shape[0] + right_offset, current_si)[valid_left_offset:output_right_offset], 
                            np.arange(0, image_shape[1], current_si), indexing='ij')

        e2_row = np.ravel_multi_index(e2_row, dims = image_shape)

        edge_indices[str(e_d) + 'row'] = {'e1': e1_row, 'e2': e2_row, 'valid_left_offset': valid_left_offset, 'valid_right_offset': valid_right_offset}

        e1_col = np.meshgrid(np.arange(0, image_shape[0], current_si),
                            np.arange(-left_offset, image_shape[1] - left_offset, current_si)[valid_left_offset:output_right_offset], indexing='ij')

        e1_col = np.ravel_multi_index(e1_col, dims = image_shape)

        e2_col = np.meshgrid(np.arange(0, image_shape[0], current_si),
                            np.arange(right_offset, image_shape[1] + right_offset, current_si)[valid_left_offset:output_right_offset], indexing='ij')
        e2_col = np.ravel_multi_index(e2_col, dims = image_shape)

        edge_indices[str(e_d) + 'col'] = {'e1': e1_col, 'e2': e2_col, 'valid_left_offset': valid_left_offset, 'valid_right_offset': valid_right_offset}

    return edge_indices

def solve_amc(batch_index, node_costs_cpu, edge_costs_cpu, edge_indices, edge_distances, edge_sampling_intervals, compute_pan_one_hot, thing_ids, return_dict):
    edge_costs_cpu_1d = []
    edge_indices_1 = []
    edge_indices_2 = []
    for (i, e_d) in enumerate(edge_distances):
        e1_row = edge_indices[str(e_d) + 'row']['e1']
        e2_row = edge_indices[str(e_d) + 'row']['e2']
        valid_left_offset = edge_indices[str(e_d) + 'row']['valid_left_offset']
        valid_right_offset = edge_indices[str(e_d) + 'row']['valid_right_offset']
        output_right_offset = -valid_right_offset if valid_right_offset > 0 else None

        current_row_costs = edge_costs_cpu[i][0, valid_left_offset:output_right_offset, :]
        edge_costs_cpu_1d.append(current_row_costs.flatten())
        edge_indices_1.append(e1_row.flatten())
        edge_indices_2.append(e2_row.flatten())

        e1_col = edge_indices[str(e_d) + 'col']['e1']
        e2_col = edge_indices[str(e_d) + 'col']['e2']
        valid_left_offset = edge_indices[str(e_d) + 'col']['valid_left_offset']
        valid_right_offset = edge_indices[str(e_d) + 'col']['valid_right_offset']
        output_right_offset = -valid_right_offset if valid_right_offset > 0 else None

        current_col_costs = edge_costs_cpu[i][1, :, valid_left_offset:output_right_offset]
        edge_costs_cpu_1d.append(current_col_costs.flatten())
        edge_indices_1.append(e1_col.flatten())
        edge_indices_2.append(e2_col.flatten())

    edge_costs_cpu_1d = np.concatenate(edge_costs_cpu_1d, 0)
    edge_indices_1 = np.concatenate(edge_indices_1, 0)
    edge_indices_2 = np.concatenate(edge_indices_2, 0)

    edge_list = list(zip(edge_indices_1, edge_indices_2, edge_costs_cpu_1d))
    num_classes = node_costs_cpu.shape[0]
    node_labels_img_shape = node_costs_cpu.shape[1:]
    node_labels_num_pixels = np.prod(node_labels_img_shape)
    node_costs_cpu = node_costs_cpu.transpose(1, 2, 0).reshape(node_labels_num_pixels, num_classes)
    node_labels, node_instance_ids, edge_labels_1d, solver_cost = amc_solver(node_costs_cpu, edge_list)
    node_instance_ids = node_instance_ids.reshape(node_labels_img_shape)
    node_labels = node_labels.transpose().reshape((num_classes, node_labels_img_shape[0], node_labels_img_shape[1]))
    edge_labels = []
    start_index = 0
    for (i, e_d) in enumerate(edge_distances):
        valid_left_offset = edge_indices[str(e_d) + 'row']['valid_left_offset']
        valid_right_offset = edge_indices[str(e_d) + 'row']['valid_right_offset']
        output_right_offset = -valid_right_offset if valid_right_offset > 0 else None

        current_costs_shape = edge_costs_cpu[i].shape
        full_edge_labels = np.zeros(current_costs_shape, dtype=np.uint8)
        valid_edge_labels = np.zeros((current_costs_shape[1] - valid_left_offset - valid_right_offset, current_costs_shape[2]), dtype=np.uint8)
        valid_shape = valid_edge_labels.shape
        valid_edge_labels = valid_edge_labels.flatten() 
        valid_edge_labels = edge_labels_1d[start_index:start_index + valid_edge_labels.size] 
        full_edge_labels[0, valid_left_offset:output_right_offset, :] = valid_edge_labels.reshape(valid_shape)

        start_index += valid_edge_labels.size

        valid_left_offset = edge_indices[str(e_d) + 'col']['valid_left_offset']
        valid_right_offset = edge_indices[str(e_d) + 'col']['valid_right_offset']
        output_right_offset = -valid_right_offset if valid_right_offset > 0 else None

        valid_edge_labels = np.zeros((current_costs_shape[1], current_costs_shape[2] - valid_left_offset - valid_right_offset), dtype=np.uint8)
        valid_shape = valid_edge_labels.shape
        valid_edge_labels = valid_edge_labels.flatten() 
        valid_edge_labels = edge_labels_1d[start_index:start_index + valid_edge_labels.size] 
        full_edge_labels[1, :, valid_left_offset:output_right_offset] = valid_edge_labels.reshape(valid_shape)
        start_index += valid_edge_labels.size
        edge_labels.append(full_edge_labels)

    panoptic_ids_one_hot = None
    if compute_pan_one_hot:
        mask_thing = np.zeros(node_labels_img_shape, dtype = np.uint8)
        for class_id in range(num_classes):
            if class_id in thing_ids:
                mask_thing = np.logical_or(mask_thing, node_labels[class_id, ::])

        only_thing_unique_instances = np.unique(node_instance_ids[mask_thing])
        num_stuff_classes = num_classes - len(thing_ids)
        panoptic_ids_one_hot = np.zeros((num_stuff_classes + len(only_thing_unique_instances), node_labels.shape[1], node_labels.shape[2]), dtype = np.uint8)
        out_index = 0
        for class_id in range(num_classes):
            if class_id not in thing_ids:
                panoptic_ids_one_hot[out_index, ::] = node_labels[class_id, ::]
                out_index += 1

        for instance_id in only_thing_unique_instances:
            current_mask = node_instance_ids == instance_id
            panoptic_ids_one_hot[out_index, ::] = current_mask
            out_index += 1

    return_dict[batch_index] = (node_labels, node_instance_ids, edge_labels, panoptic_ids_one_hot)

def solve_amc_batch(node_costs_batch, edge_costs_batch, edge_indices, edge_distances, edge_sampling_intervals, thing_ids = None, compute_pan_one_hot = False):
    panoptic_ids_one_hot = None
    if isinstance(node_costs_batch, torch.Tensor):
        node_costs_batch = torch.unbind(node_costs_batch, dim=0)

    device = node_costs_batch[0].device
    batch_size = len(node_costs_batch)
    node_costs_batch_cpu = [n.cpu().detach().numpy() for n in node_costs_batch]
    edge_costs_batch_cpu = []
    for b in range(batch_size):
        current_batch_edge_costs = []
        for i in range(len(edge_costs_batch)): # Iterate over different edge distances.
            current_batch_edge_costs.append(edge_costs_batch[i][b, ::].cpu().detach().numpy())
        edge_costs_batch_cpu.append(current_batch_edge_costs)

    if batch_size == 1:
        b = 0
        return_dict = {}
        solve_amc(b, node_costs_batch_cpu[b], edge_costs_batch_cpu[b], edge_indices, edge_distances, edge_sampling_intervals, compute_pan_one_hot, thing_ids, return_dict)
    else:
        ctx = mp.get_context('fork')
        manager = ctx.Manager()
        return_dict = manager.dict()
        workers = []
        for b in range(batch_size):
            worker = ctx.Process(target=solve_amc, args=(b, node_costs_batch_cpu[b], edge_costs_batch_cpu[b], edge_indices, edge_distances, edge_sampling_intervals, compute_pan_one_hot, thing_ids, return_dict))
            workers.append(worker)
        [w.start() for w in workers]  
        for worker in workers:
            worker.join()
            if worker.exitcode != 0:
                import sys
                print(f"ERROR: There was an error during multiprocessing with error code: {worker.exitcode}, in worker: {worker.name}. Possibly too many parallel tasks!")
                sys.exit(0)

    node_labels_batch = []
    node_instance_ids_batch = np.zeros((batch_size, 1, node_costs_batch_cpu[0].shape[1], node_costs_batch_cpu[0].shape[2]), dtype=np.int32)
    edge_labels_batch = []  
    panoptic_ids_one_hot_batch = []

    for b in sorted(return_dict.keys()):
        node_labels, node_instance_ids_batch[b, 0, ::], current_edge_labels, panoptic_ids_one_hot = return_dict[b]
        node_labels_batch.append(torch.from_numpy(node_labels))
        edge_labels_batch.append(current_edge_labels)
        assert(len(current_edge_labels) == len(edge_distances))
        if compute_pan_one_hot:
            panoptic_ids_one_hot_batch.append(torch.from_numpy(panoptic_ids_one_hot))

    edge_labels_batch_reorg = []
    for i in range(len(edge_labels_batch[0])): # Iterate over different edge distances.
        current_distance_edge_labels = []
        for b in range(batch_size):
            current_distance_edge_labels.append(edge_labels_batch[b][i])
        
        edge_labels_batch_reorg.append(torch.from_numpy(np.stack(current_distance_edge_labels, 0))) #.to(torch.float32).to(device))

    node_instance_ids_batch = torch.from_numpy(node_instance_ids_batch) #.to(torch.float32).to(device)
    return node_labels_batch, node_instance_ids_batch, tuple(edge_labels_batch_reorg), tuple(panoptic_ids_one_hot_batch) 

class AsymmetricMultiCutSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs): #ctx, edge_indices, params, node_costs, edge_costs):
        ctx.set_materialize_grads(False)
        edge_indices = inputs[0]
        params = inputs[1]
        node_costs = inputs[2]
        edge_costs = inputs[3:]

        node_labels, node_instance_ids, edge_labels, panoptic_ids_one_hot = solve_amc_batch(node_costs, edge_costs, edge_indices, params['edge_distances'], params['edge_sampling_intervals'], params['thing_ids'], params['instance_preturbation'])
        node_labels = torch.stack(node_labels, 0)
        ctx.params = params
        ctx.device = node_costs.device
        ctx.edge_indices = edge_indices
        out = tuple([])
        if params['instance_preturbation']:
            panoptic_ids_one_hot = tuple([p.to(torch.float16) for p in panoptic_ids_one_hot])
            ctx.mark_non_differentiable(panoptic_ids_one_hot)
            out += panoptic_ids_one_hot

        edge_labels = tuple([e.to(torch.float16) for e in edge_labels])
        ctx.mark_non_differentiable(node_instance_ids)
        out = (node_labels.to(torch.float16), ) + (node_instance_ids, ) + edge_labels + out
        return out

    @staticmethod
    def backward(*grad_inputs):
        """
        Backward pass computation.

        @param ctx: context from the forward pass
        @param grad_node_labels: "dL / d node_labels"
        @param grad_edge_labels: "dL / d edge_labels" 
        @param grad_node_instance_ids: Just a placeholder.
            does not contain meaningful value as node_instance_ids is non-differentiable. 
        @param grad_panoptic_ids_one_hot: Contains "dL / d panoptic_ids_one_hot" if params['instance_perturbation'] is True in forward.
            does not contain meaningful value as node_instance_ids is non-differentiable. 
        @return: gradient dL / node_costs, dL / edge_costs
        """
        ctx = grad_inputs[0]
        params = ctx.params
        num_edge_arrays = len(params['edge_distances'])
        # print(f"GRAD MEAN: {grad_node_costs.mean().item()}")
        grad_node_costs = grad_inputs[1].to(ctx.device)
        # print(f"GRAD MEAN: {grad_node_costs.mean().item()}")
        # print(f"mean: {grad_node_labels.mean().item()}, var: {grad_node_labels.var().item()}, max: {grad_node_labels.max().item()}, min: {grad_node_labels.min().item()}")
        grad_edge_costs = [g.to(ctx.device) for g in grad_inputs[3:3 + num_edge_arrays]]

        # if params['instance_preturbation']:
        #     grad_panoptic_ids_one_hot = grad_inputs[3 + num_edge_arrays:]
        #     print("AMC BACKWARD")
        #     print(grad_panoptic_ids_one_hot[0].shape)
        #     import matplotlib.pyplot as plt 
        #     for i in range(grad_panoptic_ids_one_hot[0].shape[0]):
        #         unique_values = np.unique(grad_panoptic_ids_one_hot[0][i, ::].squeeze().to(torch.float32).numpy())
        #         print(f"{i}: abs: {torch.abs(grad_panoptic_ids_one_hot[0][i, ::]).sum()}, unique vals: {unique_values}")
        #         print(grad_panoptic_ids_one_hot[0][i, ::].squeeze().dtype)
        #         # fig = plt.figure()
        #         # plt.imshow(grad_panoptic_ids_one_hot[0][i, ::].squeeze().to(torch.float32), cmap = 'gray', interpolation = 'nearest')
        #         # print()
        #         # plt.title(f"Number of unique values: {len(unique_values)}")
        #         # plt.colorbar()
        #         # plt.savefig('grad_' + str(i) +'.png')
        #         # plt.close()
        #TODOAA: Check -1
        # grad_node_costs = -1.0 * grad_node_labels
        # grad_edge_costs = [-1.0 * g for g in grad_edge_labels]

        # if params['finite_diff_order'] == 0: 
        #     # Straight-through estimator trick: Backward pass as identity.
        #     grad_node_costs = -1.0 * grad_node_labels
        #     grad_edge_costs = [-1.0 * g for g in grad_edge_labels]
        # else:
        #     saved_items = ctx.saved_tensors
        #     idx = 0
        #     node_costs = saved_items[idx]; idx += 1
        #     node_labels = saved_items[idx]; idx += 1
        #     edge_costs = saved_items[idx:idx + num_edge_arrays]; idx += num_edge_arrays
        #     edge_labels = saved_items[idx:idx + num_edge_arrays:]; idx += num_edge_arrays
        #     panoptic_ids_one_hot = None
        #     if params['instance_preturbation']:
        #         panoptic_ids_one_hot = saved_items[idx:]
        #     lambda_val = params["lambda_val"]
        #     epsilon_val = 1e-8
        #     assert grad_node_labels.shape == node_costs.shape
        #     assert grad_edge_labels is None or (grad_edge_labels[0].shape == edge_costs[0].shape and len(grad_edge_labels) == len(edge_costs))

        #     node_costs_forward = node_costs + lambda_val * grad_node_labels
        #     edge_costs_forward = [e_c + lambda_val * g_e_l for (e_c, g_e_l) in zip(edge_costs, grad_edge_labels)]
        #     edge_indices = ctx.edge_indices

        #     if params['finite_diff_order'] == 1: 
        #         node_labels_forward, _, edge_labels_forward, _ = solve_amc_batch(node_costs_forward, edge_costs_forward, edge_indices, params['edge_distances'], params['edge_sampling_intervals'])
        #         grad_node_costs = (node_labels_forward - node_labels) / (lambda_val + epsilon_val)
        #         grad_edge_costs = [(e_l_forward - e_l) / (lambda_val + epsilon_val) for (e_l_forward, e_l) in zip(edge_labels_forward, edge_labels)]

        #     elif params['finite_diff_order'] == 2:
        #         batch_size = node_costs.shape[0]
        #         node_costs_backward = node_costs - lambda_val * grad_node_labels
        #         edge_costs_backward = [e_c - lambda_val * g_e_l for (e_c, g_e_l) in zip(edge_costs, grad_edge_labels)]

        #         node_costs_combined = torch.cat((node_costs_forward, node_costs_backward), 0)
        #         edge_costs_combined = [torch.cat((ec_f, ec_b), 0) for (ec_f, ec_b) in zip(edge_costs_forward, edge_costs_backward)]

        #         node_labels_combined, _, edge_labels_combined, _ = solve_amc_batch(node_costs_combined, edge_costs_combined, edge_indices, params['edge_distances'], params['edge_sampling_intervals'])

        #         node_labels_forward = node_labels_combined[:batch_size, ::]
        #         node_labels_backward = node_labels_combined[batch_size:, ::]

        #         grad_node_costs = (node_labels_forward - node_labels_backward) / (2.0 * lambda_val + epsilon_val)

        #         # edge_labels_forward = edge_labels_combined[:edge_costs_forward.shape[0], ::]
        #         # edge_labels_backward = edge_labels_combined[edge_costs_forward.shape[0]:, ::]
        #         # grad_edge_costs = (edge_labels_forward - edge_labels_backward) / (2.0 * lambda_val + epsilon_val)
        #         grad_edge_costs = [(e_l_combined[:batch_size, ::] - e_l_combined[batch_size:, ::]) / (2.0 * lambda_val + epsilon_val) for e_l_combined in edge_labels_combined]

        #     else:
        #         assert False 
        # # Convert to tuple:
        # # print(f"nc: {grad_node_costs.abs().mean().item()}, ec: {grad_edge_costs[0].abs().mean().item()}, ")
        out = (None, ) + (None, ) + (grad_node_costs, ) + tuple(grad_edge_costs)
    
        return out # None, None, grad_node_costs, grad_edge_costs, 


class AsymmetricMulticutModule(torch.nn.Module):
    """
    Torch module for handling batches of Asymmetric Multicut Instances
    """
    def __init__(
        self,
        lambda_val,
        finite_diff_order,
        edge_distances,
        edge_sampling_intervals,
        thing_ids = None,
        instance_preturbation = False
    ):
        """
        @param lambda_val: lambda value for backpropagation
        @param finite_diff_order: Order of finite difference operation to compute gradients for backward,
            order = 1 is the approach of Vlastelica et. al, order = 2 would apply approach of Domke to compute
            two sided difference for better gradient approximation (however slower).
        @param edge_distances: List of distances between edges considered in creating the edge affinities.
            Same as affinityNet/panoptic_affinity/target_generator.py
        @param edge_sampling_intervals: List of sampling intervals at which edges are sampled in edge affinities.
            Same as affinityNet/panoptic_affinity/target_generator.py. None means all edges are considered (interval: 1).
        @param thing_ids: List of indices of segmentation targets which correspond to thing classes.
        @param instance_preturbation: True: Compute one-hot encoding of instance labels, save them and use them in backward pass.
        """
        super().__init__()
        self.solver = AsymmetricMultiCutSolver()
        self.image_size = None
        self.params = {"lambda_val": lambda_val, 
                        "finite_diff_order": finite_diff_order,  
                        "edge_distances": edge_distances, 
                        "edge_sampling_intervals": edge_sampling_intervals,
                        "instance_preturbation": instance_preturbation,
                        "thing_ids": thing_ids}

        self.edge_distances = edge_distances
        self.edge_sampling_intervals = edge_sampling_intervals

    def forward(self, node_costs_batch, edge_costs_batch):
        """
        """
        # Update edge indices if the image size is changed:
        if self.image_size is None or node_costs_batch[0, 0].shape != self.image_size:
            self.image_size = node_costs_batch[0, 0].shape
            self.edge_indices = get_edge_indices(self.image_size, self.edge_distances, self.edge_sampling_intervals)
            self.params['edge_indices'] = self.edge_indices
        model_input = (self.edge_indices, ) + (self.params, ) + (node_costs_batch, ) + tuple(edge_costs_batch)
        out = self.solver.apply(*model_input)
        return out # Tuple
