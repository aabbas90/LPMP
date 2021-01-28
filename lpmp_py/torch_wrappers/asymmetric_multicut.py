import torch
from ..raw_solvers import amc_solver
import numpy as np 
import multiprocessing as mp

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

        valid_left_offset = int(left_offset / current_si)
        valid_right_offset = int(right_offset / current_si)

        e1_row = np.meshgrid(np.arange(-left_offset, image_shape[0] - left_offset, current_si)[valid_left_offset:-valid_right_offset], 
                            np.arange(0, image_shape[1], current_si), indexing='ij')
        e1_row = np.ravel_multi_index(e1_row, dims = image_shape)

        e2_row = np.meshgrid(np.arange(right_offset, image_shape[0] + right_offset, current_si)[valid_left_offset:-valid_right_offset], 
                            np.arange(0, image_shape[1], current_si), indexing='ij')

        e2_row = np.ravel_multi_index(e2_row, dims = image_shape)

        edge_indices[str(e_d) + 'row'] = {'e1': e1_row, 'e2': e2_row, 'valid_left_offset': valid_left_offset, 'valid_right_offset': valid_right_offset}

        e1_col = np.meshgrid(np.arange(0, image_shape[0], current_si),
                            np.arange(-left_offset, image_shape[1] - left_offset, current_si)[valid_left_offset:-valid_right_offset], indexing='ij')

        e1_col = np.ravel_multi_index(e1_col, dims = image_shape)

        e2_col = np.meshgrid(np.arange(0, image_shape[0], current_si),
                            np.arange(right_offset, image_shape[1] + right_offset, current_si)[valid_left_offset:-valid_right_offset], indexing='ij')
        e2_col = np.ravel_multi_index(e2_col, dims = image_shape)

        edge_indices[str(e_d) + 'col'] = {'e1': e1_col, 'e2': e2_col, 'valid_left_offset': valid_left_offset, 'valid_right_offset': valid_right_offset}

    return edge_indices

class AsymmetricMultiCutSolver(torch.autograd.Function):
    @staticmethod
    def solve_amc(batch_index, node_costs_cpu, edge_costs_cpu, edge_indices, edge_distances, edge_sampling_intervals, return_dict):
        edge_costs_cpu_1d = []
        edge_indices_1 = []
        edge_indices_2 = []
        for (i, e_d) in enumerate(edge_distances):
            e1_row = edge_indices[str(e_d) + 'row']['e1']
            e2_row = edge_indices[str(e_d) + 'row']['e2']
            valid_left_offset = edge_indices[str(e_d) + 'row']['valid_left_offset']
            valid_right_offset = edge_indices[str(e_d) + 'row']['valid_right_offset']

            current_row_costs = edge_costs_cpu[i][0, valid_left_offset:-valid_right_offset, :]
            edge_costs_cpu_1d.append(current_row_costs.flatten())
            edge_indices_1.append(e1_row.flatten())
            edge_indices_2.append(e2_row.flatten())

            e1_col = edge_indices[str(e_d) + 'col']['e1']
            e2_col = edge_indices[str(e_d) + 'col']['e2']
            valid_left_offset = edge_indices[str(e_d) + 'col']['valid_left_offset']
            valid_right_offset = edge_indices[str(e_d) + 'col']['valid_right_offset']

            current_col_costs = edge_costs_cpu[i][1, :, valid_left_offset:-valid_right_offset]
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
            e1_row = edge_indices[str(e_d) + 'row']['e1']
            e2_row = edge_indices[str(e_d) + 'row']['e2']
            valid_left_offset = edge_indices[str(e_d) + 'row']['valid_left_offset']
            valid_right_offset = edge_indices[str(e_d) + 'row']['valid_right_offset']

            current_costs_shape = edge_costs_cpu[i].shape
            full_edge_labels = np.zeros(current_costs_shape)
            valid_edge_labels = np.zeros((current_costs_shape[1] - valid_left_offset - valid_right_offset, current_costs_shape[2]))
            valid_shape = valid_edge_labels.shape
            valid_edge_labels = valid_edge_labels.flatten() 
            valid_edge_labels = edge_labels_1d[start_index:start_index + valid_edge_labels.size] 
            full_edge_labels[0, valid_left_offset:-valid_right_offset, :] = valid_edge_labels.reshape(valid_shape)

            start_index += valid_edge_labels.size
            valid_edge_labels = np.zeros((current_costs_shape[1], current_costs_shape[2] - valid_left_offset - valid_right_offset))
            valid_shape = valid_edge_labels.shape
            valid_edge_labels = valid_edge_labels.flatten() 
            valid_edge_labels = edge_labels_1d[start_index:start_index + valid_edge_labels.size] 
            full_edge_labels[1, :, valid_left_offset:-valid_right_offset] = valid_edge_labels.reshape(valid_shape)

            start_index += valid_edge_labels.size
            edge_labels.append(full_edge_labels)

        return_dict[batch_index] = (node_labels, node_instance_ids, edge_labels)

    @staticmethod
    def solve_amc_batch(node_costs_batch, edge_costs_batch, edge_indices, edge_distances, edge_sampling_intervals):
        device = node_costs_batch.device
        batch_size = node_costs_batch.shape[0]
        node_costs_batch_cpu = node_costs_batch.cpu().detach().numpy()
        edge_costs_batch_cpu = []
        for b in range(batch_size):
            current_batch_edge_costs = []
            for i in range(len(edge_costs_batch)): # Iterate over different edge distances.
                current_batch_edge_costs.append(edge_costs_batch[i][b, ::].cpu().detach().numpy())
            edge_costs_batch_cpu.append(current_batch_edge_costs)

        workers = []
        manager = mp.Manager()
        return_dict = manager.dict()
        for b in range(batch_size):
        #    AsymmetricMultiCutSolver.solve_amc(b, node_costs_batch_cpu[b, ::], edge_costs_batch_cpu[b], edge_indices, edge_distances, edge_sampling_intervals, return_dict)
            worker = mp.Process(target=AsymmetricMultiCutSolver.solve_amc, args=(b, node_costs_batch_cpu[b, ::], edge_costs_batch_cpu[b], edge_indices, edge_distances, edge_sampling_intervals, return_dict))
            workers.append(worker)
        [w.start() for w in workers]  
        for worker in workers:
            worker.join()
        
        node_labels_batch = np.zeros_like(node_costs_batch_cpu)
        node_instance_ids_batch = np.zeros((node_costs_batch_cpu.shape[0], 1, node_costs_batch_cpu.shape[2], node_costs_batch_cpu.shape[3]))
        edge_labels_batch = []  

        for b in sorted(return_dict.keys()):
            node_labels_batch[b, ::], node_instance_ids_batch[b, 0, ::], current_edge_labels = return_dict[b]
            edge_labels_batch.append(current_edge_labels)
        
        edge_labels_batch_gpu = []
        for i in range(len(edge_labels_batch[0])): # Iterate over different edge distances.
            current_distance_edge_labels = []
            for b in range(batch_size):
                current_distance_edge_labels.append(edge_labels_batch[b][i])
            
            edge_labels_batch_gpu.append(torch.from_numpy(np.stack(current_distance_edge_labels, 0)).to(torch.float32).to(device))

        node_labels_batch = torch.from_numpy(node_labels_batch).to(torch.float32).to(device)
        node_instance_ids_batch = torch.from_numpy(node_instance_ids_batch).to(torch.float32).to(device)
        return node_labels_batch, node_instance_ids_batch, tuple(edge_labels_batch_gpu) 

    @staticmethod
    def forward(ctx, *inputs): #ctx, edge_indices, params, node_costs, edge_costs):
        edge_indices = inputs[0]
        params = inputs[1]
        node_costs = inputs[2]
        edge_costs = inputs[3:]

        device = node_costs.device

        node_labels, node_instance_ids, edge_labels = AsymmetricMultiCutSolver.solve_amc_batch(node_costs, edge_costs, edge_indices, params['edge_distances'], params['edge_sampling_intervals'])
        ctx.params = params
        ctx.edge_indices = edge_indices
        items_to_save = (node_costs, ) + (node_labels, ) + edge_costs + edge_labels
        ctx.save_for_backward(*items_to_save)
        ctx.mark_non_differentiable(node_instance_ids)
        out = (node_labels, ) + (node_instance_ids, ) + edge_labels
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
        @return: gradient dL / node_costs, dL / edge_costs
        """
        ctx = grad_inputs[0]
        grad_node_labels = grad_inputs[1]
        grad_edge_labels = grad_inputs[3:]
        params = ctx.params
        if params['finite_diff_order'] == 0: 
            # Straight-through estimator trick: Backward pass as identity.
            grad_node_costs = -1.0 * grad_node_labels
            grad_edge_costs = [-1.0 * g for g in grad_edge_labels]
        else:
            saved_items = ctx.saved_tensors
            node_costs = saved_items[0]
            node_labels = saved_items[1]
            num_edge_arrays = int((len(saved_items) - 2) / 2)
            edge_costs = saved_items[2:2 + num_edge_arrays]
            edge_labels = saved_items[2 + num_edge_arrays:]
            device = node_costs.device
            lambda_val = params["lambda_val"]
            epsilon_val = 1e-8
            assert grad_node_labels.shape == node_costs.shape and grad_edge_labels[0].shape == edge_costs[0].shape and len(grad_edge_labels) == len(edge_costs)

            node_costs_forward = node_costs + lambda_val * grad_node_labels
            edge_costs_forward = [e_c + lambda_val * g_e_l for (e_c, g_e_l) in zip(edge_costs, grad_edge_labels)]
            edge_indices = ctx.edge_indices

            if params['finite_diff_order'] == 1: 
                node_labels_forward, _, edge_labels_forward = AsymmetricMultiCutSolver.solve_amc_batch(node_costs_forward, edge_costs_forward, edge_indices, params['edge_distances'], params['edge_sampling_intervals'])
                grad_node_costs = (node_labels_forward - node_labels) / (lambda_val + epsilon_val)
                grad_edge_costs = [(e_l_forward - e_l) / (lambda_val + epsilon_val) for (e_l_forward, e_l) in zip(edge_labels_forward, edge_labels)]

            elif params['finite_diff_order'] == 2:
                batch_size = node_costs.shape[0]
                node_costs_backward = node_costs - lambda_val * grad_node_labels
                edge_costs_backward = [e_c - lambda_val * g_e_l for (e_c, g_e_l) in zip(edge_costs, grad_edge_labels)]

                node_costs_combined = torch.cat((node_costs_forward, node_costs_backward), 0)
                edge_costs_combined = [torch.cat((ec_f, ec_b), 0) for (ec_f, ec_b) in zip(edge_costs_forward, edge_costs_backward)]

                node_labels_combined, _, edge_labels_combined = AsymmetricMultiCutSolver.solve_amc_batch(node_costs_combined, edge_costs_combined, edge_indices, params['edge_distances'], params['edge_sampling_intervals'])

                node_labels_forward = node_labels_combined[:batch_size, ::]
                node_labels_backward = node_labels_combined[batch_size:, ::]

                grad_node_costs = (node_labels_forward - node_labels_backward) / (2.0 * lambda_val + epsilon_val)

                # edge_labels_forward = edge_labels_combined[:edge_costs_forward.shape[0], ::]
                # edge_labels_backward = edge_labels_combined[edge_costs_forward.shape[0]:, ::]
                # grad_edge_costs = (edge_labels_forward - edge_labels_backward) / (2.0 * lambda_val + epsilon_val)
                grad_edge_costs = [(e_l_combined[:batch_size, ::] - e_l_combined[batch_size:, ::]) / (2.0 * lambda_val + epsilon_val) for e_l_combined in edge_labels_combined]

            # elif params['finite_diff_order'] == 4:
            #     node_costs_double_forward = node_costs + 2 * lambda_val * grad_node_labels
            #     node_costs_backward = node_costs - lambda_val * grad_node_labels
            #     node_costs_double_backward = node_costs - 2 * lambda_val * grad_node_labels

            #     node_costs_combined = torch.cat((node_costs_double_forward, node_costs_forward, node_costs_backward, node_costs_double_backward), 0)

            #     edge_costs_double_forward = edge_costs + 2 * lambda_val * grad_edge_labels
            #     edge_costs_backward = edge_costs - lambda_val * grad_edge_labels
            #     edge_costs_double_backward = edge_costs - 2 * lambda_val * grad_edge_labels

            #     edge_costs_combined = torch.cat((edge_costs_double_forward, edge_costs_forward, edge_costs_backward, edge_costs_double_backward), 0)
            #     node_labels_combined, _, edge_labels_combined = AsymmetricMultiCutSolver.solve_amc_batch(node_costs_combined, edge_costs_combined, edge_indices, params['edge_distances'], params['edge_sampling_intervals'])

            #     shape_each_eval = node_costs_double_forward.shape[0]
            #     node_labels_double_forward = node_labels_combined[:shape_each_eval, ::]
            #     node_labels_forward = node_labels_combined[shape_each_eval:2 * shape_each_eval, ::]
            #     node_labels_backward = node_labels_combined[2 * shape_each_eval:3 * shape_each_eval, ::]
            #     node_labels_double_backward = node_labels_combined[3 * shape_each_eval:, ::]

            #     grad_node_costs = (-node_labels_double_forward +
            #                         8 * node_labels_forward - 
            #                         8 * node_labels_backward +
            #                         node_labels_double_backward) / (12.0 * lambda_val + epsilon_val)

            #     edge_labels_double_forward = edge_labels_combined[:shape_each_eval, ::]
            #     edge_labels_forward = edge_labels_combined[shape_each_eval:2 * shape_each_eval, ::]
            #     edge_labels_backward = edge_labels_combined[2 * shape_each_eval:3 * shape_each_eval, ::]
            #     edge_labels_double_backward = edge_labels_combined[3 * shape_each_eval:, ::]

            #     grad_edge_costs = (-edge_labels_double_forward +
            #                         8 * edge_labels_forward - 
            #                         8 * edge_labels_backward +
            #                         edge_labels_double_backward) / (12.0 * lambda_val + epsilon_val)
                # import pdb; pdb.set_trace()
                # import matplotlib.pyplot as plt 
                # plt.imshow(grad_edge_labels[0, 0, :, :].squeeze().detach().cpu())
                # plt.colorbar()
                # plt.savefig('/home/ahabbas/tmp.png')
                # import pdb; pdb.set_trace()

            else:
                assert False 
        # Convert to tuple:
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
        edge_sampling_intervals
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
        """
        super().__init__()
        self.solver = AsymmetricMultiCutSolver()
        self.image_size = None
        self.params = {"lambda_val": lambda_val, "finite_diff_order": finite_diff_order,  "edge_distances": edge_distances, "edge_sampling_intervals": edge_sampling_intervals}
        self.edge_distances = edge_distances
        self.edge_sampling_intervals = edge_sampling_intervals

    def forward(self, node_costs_batch, edge_costs_batch):
        """
        """
        # Update edge indices if the image size is changed:
        if self.image_size is None or node_costs_batch[0, 0].shape != self.image_size:
            self.image_size = node_costs_batch[0, 0].shape
            self.edge_indices = get_edge_indices(self.image_size, self.edge_distances, self.edge_sampling_intervals)
        model_input = (self.edge_indices, ) + (self.params, ) + (node_costs_batch, ) + tuple(edge_costs_batch)
        out = self.solver.apply(*model_input)
        return out # Tuple
