import logging
import os

import dgl
import numpy as np
import scipy.sparse as sp
from dgl import graph
from dgl.data import DGLDataset, download
import pickle
import scipy.io as sio
from scipy.sparse import coo_matrix
import torch

import world


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

class SocialDataset:
    def __init__(self,
                 raw_dir
                 ):
        super(SocialDataset, self).__init__()
        self.raw_dir = raw_dir
        self.D_adj_D = None
        self._init_social_graph()
        self._init_inter_graph()
        self.social_D_adj_D = None
        self.inter_D_adj_D = None


    def _init_social_graph(self):
        # process raw data to graphs, labels, splitting masks
        file_path = f'Data/{self.raw_dir}/trustnetwork.mat'
        data = sio.loadmat(file_path)
        a = data['trustnetwork']
        a_int = a.astype(int)
        sr = a_int[:, 1]
        de = a_int[:, 0]

        self.social_graph = dgl.graph((sr,de))
        self.n_users = self.social_graph.num_nodes()

        """
                    Split edge set for training and testing
                """
        eids = np.arange(self.social_graph.num_edges())
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * 0.2)
        train_size = self.social_graph.num_edges() - test_size
        test_pos_u, test_pos_v = sr[eids[:test_size]], de[eids[:test_size]]
        train_pos_u, train_pos_v = sr[eids[test_size:]], de[eids[test_size:]]

        sp_mat = coo_matrix(([1]*train_size,(train_pos_u, train_pos_v)),shape=(self.n_users,self.n_users))
        self.social_adj_matrix = sp_mat.toarray()

        self.train_size = train_size
        self.test_size = test_size
        self.allPos = self.getPosUsers(list(range(self.n_users)))
        self.test_pos_u = test_pos_u
        self.test_pos_v = test_pos_v

        self.testDict = self.__build_test()


    def _init_inter_graph(self):
        file_path = f'Data/{self.raw_dir}/rating.mat'
        data = sio.loadmat(file_path)
        a = data['rating']
        src = a[:, 0]
        dst = a[:, 1]
        e_weight = a[:, 3]

        sp_mat = coo_matrix(([1]*len(src), (src, dst)))
        g = dgl.bipartite_from_scipy(sp_mat, vtype='_V', etype='_E', utype='_U')
        self.inter_adj = coo_matrix(([1]*len(src), (src, dst)), shape=(g.num_nodes(ntype='_U'), g.num_nodes(ntype='_V')),
                              dtype=np.int16)
        self.inter_adj_matrix = self.inter_adj.toarray()
        self.inter_adj_weight = coo_matrix((e_weight / 15, (src, dst)),
                                           shape=(g.num_nodes(ntype='_U'), g.num_nodes(ntype='_V')),
                                           dtype=np.float32)

        self.inter_graph = g
        self.n_items = self.inter_graph.num_nodes(ntype='_V')
        self.n_users_U = self.inter_graph.num_nodes(ntype='_U')
        self.n_ratings = list(set(e_weight))

        self._U = src
        self._V = dst
        self._R = e_weight
        self.user_items, self.rating = self.__build_inter()

        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items),
                                dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R_w = self.inter_adj_weight.tolil()
        adj_mat[:self.n_users, self.n_users:] = R_w
        adj_mat[self.n_users:, :self.n_users] = R_w.T
        adj_mat = adj_mat.todok()
        self.adj_weight = adj_mat


    def get_social_D_adj_D(self):
        if self.social_D_adj_D is None:
            try:
                norm_adj = sp.load_npz(f'./Data/{self.raw_dir}/social_adj_mat.npz')
                logging.debug("successfully loaded normalized social adjacency matrix")
            except IOError:
                logging.debug("generating adjacency matrix")

                """
                    compute degree matrix D and  D^(-1/2)
                """

                rowsum = np.array(self.social_adj_matrix.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                d_mat = d_mat.astype(np.float32)
                """
                     compute D^(-1/2)AD^(-1/2)
                """
                adj_matrix_gpu = torch.FloatTensor(self.social_adj_matrix).to(world.device)
                d_mat_gpu = torch.FloatTensor(d_mat.toarray()).to(world.device)
                norm_adj = torch.mm(d_mat_gpu,adj_matrix_gpu).to(world.device)
                norm_adj = torch.mm(norm_adj,d_mat_gpu).to(world.device)
                norm_adj = norm_adj.cpu().numpy()
                norm_adj = sp.csr_matrix(norm_adj)
                sp.save_npz(f'./Data/{self.raw_dir}/social_adj_mat.npz', norm_adj)

            self.social_D_adj_D = _convert_sp_mat_to_sp_tensor(norm_adj)
            self.social_D_adj_D = self.social_D_adj_D.coalesce().to(world.device)
            return self.social_D_adj_D

    def get_inter_D_adj_D(self):
        if self.inter_D_adj_D is None:
            try:
                norm_adj = sp.load_npz(f'./Data/{self.raw_dir}/interaction_adj_mat_w.npz')
                logging.debug("successfully loaded normalized interaction adjacency matrix")
            except IOError:
                logging.debug("generating adjacency matrix")

                """  
                    build a graph in torch.sparse.IntTensor.
                    A = 
                        |I,   R|
                        |R^T, I|
                """
                adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items),
                                        dtype=np.int16)
                adj_mat = adj_mat.tolil()
                R = self.inter_adj.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                print(adj_mat)
                """
                    compute degree matrix D and  D^(-1/2)
                """
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                """
                    compute D^(-1/2)AD^(-1/2)
                """
                norm_adj = d_mat.dot(self.adj_weight)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                sp.save_npz(f'./Data/{self.raw_dir}/interaction_adj_mat_w.npz', norm_adj)

            self.inter_D_adj_D = _convert_sp_mat_to_sp_tensor(norm_adj)
            self.inter_D_adj_D = self.inter_D_adj_D.coalesce().to(world.device)
            return self.inter_D_adj_D



    def getPosUsers(self, users):
        """
        Method of get user all positive users
        Returns
        -------
        [ndarray0,...,ndarray_users]
        """
        posUsers = []
        for user in users:
            posUsers.append(self.social_adj_matrix[user].nonzero()[0])

        return posUsers

    def __build_test(self):
        """
        Method of build test dictionary
        Returns
        -------
            dict: {user: [users]}
        """
        test_data = {}
        for i in range(self.test_size):
            user_u = self.test_pos_u[i]
            user_v = self.test_pos_v[i]
            if test_data.get(user_u):
                test_data[user_u].append(user_v)
            else:
                test_data[user_u] = [user_v]
        return test_data

    def __build_inter(self):
        """
        Method of build user_items dictionary and rating dictionary
        Returns
        -------
            dict: {user: [items]}
                  {user: [rating]}
        """
        user_items = {}
        rating = {}
        for i in range(len(self._U)):
            user_u = self._U[i]
            user_v = self._V[i]
            r = self._R[i]
            if user_items.get(user_u):
                user_items[user_u].append(user_v)
                rating[user_u].append(r)
            else:
                user_items[user_u] = [user_v]
                rating[user_u] = [r]
        return user_items,rating

