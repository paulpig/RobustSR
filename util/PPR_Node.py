from sklearn.preprocessing import normalize, StandardScaler
import numpy as np
from cytoolz import curry
from scipy import sparse as sp
import os
import multiprocessing as mp


class PPR:
    #Node-wise personalized pagerank
    def __init__(self, adj_mat, maxsize=200, n_order=2, alpha=0.85):
        self.n_order = n_order
        self.maxsize = maxsize
        self.adj_mat = adj_mat
        self.P = normalize(adj_mat, norm='l1', axis=0) #(user_num, item_num)
        self.d = np.array(adj_mat.sum(1)).squeeze()
    
    @curry
    def search(self, seed, alpha=0.85):
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1])
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9) #(user_num, 1)
        
        # idx = scores.argsort()[::-1][:self.maxsize]
        # neighbor = np.array(x.indices[idx])
        
        # seed_idx = np.where(neighbor == seed)[0]
        # if seed_idx.size == 0:
        #     neighbor = np.append(np.array([seed]), neighbor)
        # else :
        #     seed_idx = seed_idx[0]
        #     neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]
            
        # assert np.where(neighbor == seed)[0].size == 1
        # assert np.where(neighbor == seed)[0][0] == 0
        
        # return neighbor
        return scores
    
    # @curry
    # def process(self, seed):
    #     ppr_path = os.path.join(path, 'ppr{}'.format(seed))
    #     if not os.path.isfile(ppr_path) or os.stat(ppr_path).st_size == 0:
    #         print ('Processing node {}.'.format(seed))
    #         neighbor = self.search(seed)
    #         # torch.save(neighbor, ppr_path)
    #     else :
    #         print ('File of node {} exists.'.format(seed))
    
    def search_all(self, node_num, path):
        neighbor  = {}
        # if os.path.isfile(path+'_neighbor') and os.stat(path+'_neighbor').st_size != 0:
        #     print ("Exists neighbor file")
            # neighbor = torch.load(path+'_neighbor')
        # else :
            # print ("Extracting subgraphs")
            # os.system('mkdir {}'.format(path))
        with mp.Pool() as pool:
            # list(pool.imap_unordered(self.process(path), list(range(node_num)), chunksize=1000))
            list(pool.imap_unordered(self.search(), list(range(node_num)), chunksize=1000))
            
        print ("Finish Extracting")
            # for i in range(node_num):
                # neighbor[i] = torch.load(os.path.join(path, 'ppr{}'.format(i)))
            # torch.save(neighbor, path+'_neighbor')
            # os.system('rm -r {}'.format(path))
            # print ("Finish Writing")
        return neighbor