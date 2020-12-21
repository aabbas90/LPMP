import torch
from ..raw_solvers import amc_solver
import numpy as np 
import torch.multiprocessing as mp
import time

def get_edge_indices(image_shape):
    indices = np.arange(np.prod(image_shape)).reshape(image_shape).astype(np.int32)

    e1_row = indices[:-1, :].ravel()
    e2_row = indices[1:, :].ravel()
    row_edges = np.stack((e1_row, e2_row), 1)

    e1_col = indices[:, :-1].ravel()
    e2_col = indices[:, 1:].ravel()
    col_edges = np.stack((e1_col, e2_col), 1)

    edge_indices = np.concatenate((row_edges, col_edges), 0)
    return edge_indices
    # edge_costs = np.concatenate((edge_costs_cpu[0, ::].flatten(), edge_costs_cpu[1, ::].flatten()), 0)
    # edge_list = list(zip(edge_indices[:, 0], edge_indices[:, 1], edge_costs))
    # return edge_list

class AsymmetricMultiCutSolver(torch.autograd.Function):
    @staticmethod
    def solve_amc(node_costs, edge_costs, edge_indices):
        device = edge_costs.device
        edge_costs_cpu = edge_costs.cpu().detach().numpy()
        row_aff = edge_costs_cpu[0, :-1, :]
        col_aff = edge_costs_cpu[1, :, :-1]
        edge_costs_cpu = np.concatenate((row_aff.flatten(), col_aff.flatten()), 0)
        edge_list = list(zip(edge_indices[:, 0], edge_indices[:, 1], edge_costs_cpu))
        node_labels, edge_labels_1d, solver_cost = amc_solver(node_costs, edge_list)
        edge_labels = np.zeros(edge_costs.shape)
        edge_labels[0, :-1, :] = edge_labels_1d[:row_aff.size].reshape(row_aff.shape)
        edge_labels[1, :, :-1] = edge_labels_1d[row_aff.size:].reshape(col_aff.shape)
        node_labels = torch.from_numpy(node_labels).to(torch.float32).to(device)
        edge_labels = torch.from_numpy(edge_labels).to(torch.float32).to(device)
        return node_labels, edge_labels 

    @staticmethod
    def forward(ctx, node_costs, edge_costs, edge_indices, params):
        """
        @param ctx: context for backpropagation
        """
        device = node_costs.device
        node_labels, edge_labels = AsymmetricMultiCutSolver.solve_amc(node_costs, edge_costs, edge_indices)
        ctx.params = params
        ctx.edge_indices = edge_indices
        ctx.save_for_backward(node_costs, node_labels, edge_costs, edge_labels)
        return node_labels, edge_labels

    @staticmethod
    def backward(ctx, grad_node_labels, grad_edge_labels):
        """
        Backward pass computation.

        @param ctx: context from the forward pass
        @param grad_node_labels: "dL / d node_labels"
        @param grad_edge_labels: "dL / d edge_labels" 
        @return: gradient dL / node_costs, dL / edge_costs
        """
        node_costs, node_labels, edge_costs, edge_labels = ctx.saved_tensors
        device = node_costs.device
        lambda_val = ctx.params["lambda_val"]
        epsilon_val = 1e-8
        assert grad_node_labels.shape == node_costs.shape and grad_edge_labels.shape == edge_costs.shape

        node_costs_prime = node_costs + lambda_val * grad_node_labels
        edge_costs_prime = edge_costs + lambda_val * grad_edge_labels
        edge_indices = ctx.edge_indices

        node_labels_prime, edge_labels_prime = AsymmetricMultiCutSolver.solve_amc(node_costs_prime, edge_costs_prime, edge_indices)

        node_labels_prime = torch.from_numpy(node_labels_prime).to(torch.float32).to(device)
        edge_labels_prime = torch.from_numpy(edge_labels_prime).to(torch.float32).to(device)

        grad_node_costs = -(node_labels - node_labels_prime) / (lambda_val + epsilon_val)
        grad_edge_costs = -(edge_labels - edge_labels_prime) / (lambda_val + epsilon_val)

        return grad_node_costs, grad_edge_costs, None, None


class AsymmetricMulticutModule(torch.nn.Module):
    """
    Torch module for handling batches of Asymmetric Multicut Instances
    """
    def __init__(
        self,
        image_size,
        lambda_val,
    ):
        """
        @param lambda_val: lambda value for backpropagation
        """
        super().__init__()
        self.solver = AsymmetricMultiCutSolver()
        self.edge_indices = get_edge_indices(image_size)
        self.params = {"lambda_val": lambda_val}

    def forward(self, node_costs_batch, edge_costs_batch):
        """
        """
        node_labels_batch = torch.zeros_like(node_costs_batch)
        edge_labels_batch = torch.zeros_like(edge_costs_batch)
        batch_size = node_costs_batch.shape[0]
        # for b in range(batch_size):
        #     node_labels_batch[b, ::], edge_labels_batch[b, ::] = self.solver.apply(node_costs_batch[b, ::], edge_costs_batch[b, ::], self.edge_indices, self.params)
        start = time.time()
        workers = []
        for b in range(batch_size):
            worker = mp.Process(target=self.solver.apply, args=(node_costs_batch[b, ::], edge_costs_batch[b, ::], self.edge_indices, self.params, ))
            workers.append(worker)
        [w.start() for w in workers]  
        for worker in workers:
           worker.join()

        print(time.time() - start)  
        import pdb; pdb.set_trace()
        return node_labels_batch, edge_labels_batch
