import torch
from ..raw_solvers import amc_solver
import numpy as np 
import multiprocessing as mp

def get_edge_indices(image_shape, edge_distances):
    indices = np.arange(np.prod(image_shape)).reshape(image_shape).astype(np.int32)

    edge_indices = []
    for e_d in edge_distances:
        e1_row = indices[:-e_d, :].ravel()
        e2_row = indices[e_d:, :].ravel()
        row_edges = np.stack((e1_row, e2_row), 1)
        edge_indices.append(row_edges)

        e1_col = indices[:, :-e_d].ravel()
        e2_col = indices[:, e_d:].ravel()
        col_edges = np.stack((e1_col, e2_col), 1)
        edge_indices.append(col_edges)

    edge_indices = np.concatenate(edge_indices, 0)
    return edge_indices

    # edge_costs = np.concatenate((edge_costs_cpu[0, ::].flatten(), edge_costs_cpu[1, ::].flatten()), 0)
    # edge_list = list(zip(edge_indices[:, 0], edge_indices[:, 1], edge_costs))
    # return edge_list

class AsymmetricMultiCutSolver(torch.autograd.Function):
    @staticmethod
    def solve_amc(batch_index, node_costs_cpu, edge_costs_cpu, edge_indices, edge_distances, return_dict):
        edge_costs_cpu_1d = []
        for (i, e_d) in enumerate(edge_distances):
            row_aff = edge_costs_cpu[2 * i, :-e_d, :]
            edge_costs_cpu_1d.append(row_aff.flatten())
            col_aff = edge_costs_cpu[2 * i + 1, :, :-e_d]
            edge_costs_cpu_1d.append(col_aff.flatten())

        edge_costs_cpu_1d = np.concatenate(edge_costs_cpu_1d, 0)
        edge_list = list(zip(edge_indices[:, 0], edge_indices[:, 1], edge_costs_cpu_1d))
        num_classes = node_costs_cpu.shape[0]
        node_labels_img_shape = node_costs_cpu.shape[1:]
        node_labels_num_pixels = np.prod(node_labels_img_shape)
        node_costs_cpu = node_costs_cpu.transpose(1, 2, 0).reshape(node_labels_num_pixels, num_classes)
        node_labels, node_instance_ids, edge_labels_1d, solver_cost = amc_solver(node_costs_cpu, edge_list)
        node_instance_ids = node_instance_ids.reshape(node_labels_img_shape)
        node_labels = node_labels.transpose().reshape((num_classes, node_labels_img_shape[0], node_labels_img_shape[1]))
        edge_labels = np.zeros(edge_costs_cpu.shape)
        start_index_1d = 0
        for (i, e_d) in enumerate(edge_distances):
            output_shape = edge_labels[2 * i, :-e_d, :].shape
            current_numel = np.prod(output_shape)
            end_index_1d = current_numel + start_index_1d
            edge_labels[2 * i, :-e_d, :] = edge_labels_1d[start_index_1d:end_index_1d].reshape(output_shape)
            start_index_1d += current_numel

            output_shape = edge_labels[2 * i + 1, :, :-e_d].shape
            current_numel = np.prod(output_shape)
            end_index_1d = current_numel + start_index_1d
            edge_labels[2 * i + 1, :, :-e_d] = edge_labels_1d[start_index_1d:end_index_1d].reshape(output_shape)
            start_index_1d += current_numel
        
        return_dict[batch_index] = (node_labels, node_instance_ids, edge_labels)

    @staticmethod
    def solve_amc_batch(node_costs_batch, edge_costs_batch, edge_indices, edge_distances):
        device = edge_costs_batch.device
        node_costs_batch_cpu = node_costs_batch.cpu().detach().numpy()
        edge_costs_batch_cpu = edge_costs_batch.cpu().detach().numpy()
        node_labels_batch = np.zeros_like(node_costs_batch_cpu)
        node_instance_ids_batch = np.zeros((node_costs_batch_cpu.shape[0], 1, node_costs_batch_cpu.shape[2], node_costs_batch_cpu.shape[3]))
        edge_labels_batch = np.zeros_like(edge_costs_batch_cpu)

        batch_size = node_costs_batch.shape[0]

        workers = []
        manager = mp.Manager()
        return_dict = manager.dict()
        for b in range(batch_size):
        #    AsymmetricMultiCutSolver.solve_amc(b, node_costs_batch_cpu[b, ::], edge_costs_batch_cpu[b, ::], edge_indices, edge_distances, return_dict)
            worker = mp.Process(target=AsymmetricMultiCutSolver.solve_amc, args=(b, node_costs_batch_cpu[b, ::], edge_costs_batch_cpu[b, ::], edge_indices, edge_distances, return_dict))
            workers.append(worker)
        [w.start() for w in workers]  
        for worker in workers:
            worker.join()
        
        for b in sorted(return_dict.keys()):
            node_labels_batch[b, ::], node_instance_ids_batch[b, 0, ::], edge_labels_batch[b, ::] = return_dict[b]
        node_labels_batch = torch.from_numpy(node_labels_batch).to(torch.float32).to(device)
        edge_labels_batch = torch.from_numpy(edge_labels_batch).to(torch.float32).to(device)
        node_instance_ids_batch = torch.from_numpy(node_instance_ids_batch).to(torch.float32).to(device)
        return node_labels_batch, node_instance_ids_batch, edge_labels_batch 

    @staticmethod
    def forward(ctx, node_costs, edge_costs, edge_indices, params):
        """
        @param ctx: context for backpropagation
        """
        device = node_costs.device
        node_labels, node_instance_ids, edge_labels = AsymmetricMultiCutSolver.solve_amc_batch(node_costs, edge_costs, edge_indices, params['edge_distances'])
        ctx.params = params
        ctx.edge_indices = edge_indices
        ctx.save_for_backward(node_costs, node_labels, edge_costs, edge_labels)
        ctx.mark_non_differentiable(node_instance_ids)
        return node_labels, edge_labels, node_instance_ids

    @staticmethod
    def backward(ctx, grad_node_labels, grad_edge_labels, grad_node_instance_ids):
        """
        Backward pass computation.

        @param ctx: context from the forward pass
        @param grad_node_labels: "dL / d node_labels"
        @param grad_edge_labels: "dL / d edge_labels" 
        @param grad_node_instance_ids: Just a placeholder.
            does not contain meaningful value as node_instance_ids is non-differentiable. 
        @return: gradient dL / node_costs, dL / edge_costs
        """
        params = ctx.params
        if params['finite_diff_order'] == 0: 
            # Straight-through estimator trick: Backward pass as identity.
            grad_node_costs = -1 * grad_node_labels
            grad_edge_costs = -1 * grad_edge_labels
        else:
            node_costs, node_labels, edge_costs, edge_labels = ctx.saved_tensors
            device = node_costs.device
            lambda_val = params["lambda_val"]
            epsilon_val = 1e-8
            assert grad_node_labels.shape == node_costs.shape and grad_edge_labels.shape == edge_costs.shape

            node_costs_forward = node_costs + lambda_val * grad_node_labels
            edge_costs_forward = edge_costs + lambda_val * grad_edge_labels
            edge_indices = ctx.edge_indices

            if params['finite_diff_order'] == 1: 
                node_labels_forward, _, edge_labels_forward = AsymmetricMultiCutSolver.solve_amc_batch(node_costs_forward, edge_costs_forward, edge_indices, params['edge_distances'])
                grad_node_costs = (node_labels_forward - node_labels) / (lambda_val + epsilon_val)
                grad_edge_costs = (edge_labels_forward - edge_labels) / (lambda_val + epsilon_val)
            
            elif params['finite_diff_order'] == 2:
                node_costs_backward = node_costs - lambda_val * grad_node_labels
                edge_costs_backward = edge_costs - lambda_val * grad_edge_labels

                node_costs_combined = torch.cat((node_costs_forward, node_costs_backward), 0)
                edge_costs_combined = torch.cat((edge_costs_forward, edge_costs_backward), 0)

                node_labels_combined, _, edge_labels_combined = AsymmetricMultiCutSolver.solve_amc_batch(node_costs_combined, edge_costs_combined, edge_indices, params['edge_distances'])

                node_labels_forward = node_labels_combined[:node_costs_forward.shape[0], ::]
                node_labels_backward = node_labels_combined[node_costs_forward.shape[0]:, ::]

                grad_node_costs = (node_labels_forward - node_labels_backward) / (2.0 * lambda_val + epsilon_val)

                edge_labels_forward = edge_labels_combined[:edge_costs_forward.shape[0], ::]
                edge_labels_backward = edge_labels_combined[edge_costs_forward.shape[0]:, ::]
                grad_edge_costs = (edge_labels_forward - edge_labels_backward) / (2.0 * lambda_val + epsilon_val)
                # import pdb; pdb.set_trace()
                # import matplotlib.pyplot as plt 
                # plt.imshow(grad_edge_labels[0, 0, :, :].squeeze().detach().cpu())
                # plt.colorbar()
                # plt.savefig('/home/ahabbas/tmp.png')
                # import pdb; pdb.set_trace()

            else:
                assert False 

        return grad_node_costs, grad_edge_costs, None, None


class AsymmetricMulticutModule(torch.nn.Module):
    """
    Torch module for handling batches of Asymmetric Multicut Instances
    """
    def __init__(
        self,
        lambda_val,
        finite_diff_order,
        edge_distances
    ):
        """
        @param lambda_val: lambda value for backpropagation
        @param finite_diff_order: Order of finite difference operation to compute gradients for backward,
            order = 1 is the approach of Vlastelica et. al, order = 2 would apply approach of Domke to compute
            two sided difference for better gradient approximation (however slower).
        @param edge_distances: List of distances between edges considered in creating the edge affinities.
            Same as affinityNet/panoptic_affinity/target_generator.py
        """
        super().__init__()
        self.solver = AsymmetricMultiCutSolver()
        self.image_size = None
        self.params = {"lambda_val": lambda_val, "finite_diff_order": finite_diff_order,  "edge_distances": edge_distances}
        self.edge_distances = edge_distances

    def forward(self, node_costs_batch, edge_costs_batch):
        """
        """
        # Update edge indices if the image size is changed:
        if self.image_size is None or edge_costs_batch[0, 0].shape != self.image_size:
            self.image_size = edge_costs_batch[0, 0].shape
            self.edge_indices = get_edge_indices(self.image_size, self.edge_distances)
        node_labels_batch, edge_labels_batch, node_instance_ids_batch = self.solver.apply(node_costs_batch, edge_costs_batch, self.edge_indices, self.params)
        return node_labels_batch, edge_labels_batch, node_instance_ids_batch
