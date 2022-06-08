from cProfile import label
from re import T
from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix, eye
import scipy.sparse as sp
import numpy as np
import os
from util import config
from util.loss import bpr_loss
import random
import pdb
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#Suggested Maxium epoch LastFM: 120, Douban-Book: 30, Yelp: 30.
#Read the paper for the values of other parameters. tmp
'''
We have transplated QRec from py2 to py3. But we found that, with py3, SEPT achieves higher NDCG
but lower (slightly) Prec and Recall compared with the results reported in the paper.
'''
class SEPTV2_add_social_rep(SocialRecommender, GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(SEPTV2_add_social_rep, self).readConfiguration()
        args = config.OptionConf(self.config['SEPTV2_add_social_rep'])
        self.n_layers = int(args['-n_layer'])
        self.ss_rate = float(args['-ss_rate'])
        self.drop_rate = float(args['-drop_rate'])
        self.instance_cnt = int(args['-ins_cnt'])
        self.ppr_rate = float(args['-ppr_rate'])
        self.s_cl_rate = float(args['-s_cl_rate'])
        self.social_ppr_cluster_w = float(args['-social_ppr_cluster_w'])
        self.rec_loss_aug_w = float(args['-rec_loss_aug_w'])
        self.interact_ppr_w = float(args['-interact_ppr_w'])
        self.rec_ppr_aug_w = float(args['-rec_ppr_aug_w'])
        self.inter_ppr_w = float(args['-inter_ppr_w'])
        self.graph_label_w = float(args['-graph_label_w'])
        self.cluster_type = int(args['-cluster_type'])
        self.inter_cl_w = float(args['-inter_cl_w'])
        # pdb.set_trace()


    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_items), dtype=np.float32)
        return ratingMatrix

    def get_birectional_social_matrix(self):
        row_idx = [self.data.user[pair[0]] for pair in self.social.relation]
        col_idx = [self.data.user[pair[1]] for pair in self.social.relation]
        follower_np = np.array(row_idx)
        followee_np = np.array(col_idx)
        relations = np.ones_like(follower_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((relations, (follower_np, followee_np)), shape=(self.num_users, self.num_users))
        # pdb.set_trace()
        adj_mat = tmp_adj.multiply(tmp_adj)
        return adj_mat
    
    def get_birectional_social_matrix_v2(self):
        def normalization(M):
            rowsum = np.array(M.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(M)
            return norm_adj_tmp.dot(d_mat_inv)
        row_idx = [self.data.user[pair[0]] for pair in self.social.relation]
        col_idx = [self.data.user[pair[1]] for pair in self.social.relation]

        # follower_np = np.array(row_idx + col_idx) #bidirection.
        # followee_np = np.array(col_idx + row_idx)
        follower_np = np.array(row_idx)
        followee_np = np.array(col_idx)
        relations = np.ones_like(follower_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((relations, (follower_np, followee_np)), shape=(self.num_users, self.num_users))
        # pdb.set_trace()
        # adj_mat = tmp_adj.multiply(tmp_adj)
        adj_mat = normalization(tmp_adj)
        return adj_mat

    def get_social_related_views(self, social_mat, rating_mat):
        def normalization(M):
            rowsum = np.array(M.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(M)
            return norm_adj_tmp.dot(d_mat_inv)
        social_matrix = social_mat.dot(social_mat)
        social_matrix =  social_matrix.multiply(social_mat) + eye(self.num_users)
        sharing_matrix = rating_mat.dot(rating_mat.T)
        sharing_matrix = sharing_matrix.multiply(social_mat) + eye(self.num_users)
        social_matrix = normalization(social_matrix)
        sharing_matrix = normalization(sharing_matrix)
        return [social_matrix, sharing_matrix]

    def get_local_global_user_rep(self, socail_ppr_mat, social_local_mat):
        """
        socail_ppr_mat: sp_mat
        social_local_mat: sp_mat
        """
        socail_ppr_mat = self._convert_sp_mat_to_sp_tensor(socail_ppr_mat)
        social_local_mat = self._convert_sp_mat_to_sp_tensor(social_local_mat)
        # add self-gating
        user_embeedings_v1 = self.user_embeddings * tf.nn.sigmoid(tf.layers.dense(inputs=self.user_embeddings, units=self.emb_size, activation=None))
        user_embeedings_v2 = self.user_embeddings * tf.nn.sigmoid(tf.layers.dense(inputs=self.user_embeddings, units=self.emb_size, activation=None))
        friend_view_embeddings_global, friend_view_embeddings_local = user_embeedings_v1, user_embeedings_v2
        all_social_embeddings_local = [user_embeedings_v1]
        all_social_embeddings_global = [user_embeedings_v2]
        
        # all_social_embeddings_local = []
        # all_social_embeddings_global = []
        # friend view, global
        friend_view_embeddings_global = tf.sparse_tensor_dense_matmul(socail_ppr_mat, friend_view_embeddings_global)
        norm_embeddings = tf.math.l2_normalize(friend_view_embeddings_global, axis=1)
        all_social_embeddings_global += [norm_embeddings]
        
        for k in range(self.n_layers):
            # friend view, local
            friend_view_embeddings_local = tf.sparse_tensor_dense_matmul(social_local_mat, friend_view_embeddings_local)
            norm_embeddings = tf.math.l2_normalize(friend_view_embeddings_local, axis=1)
            all_social_embeddings_local += [norm_embeddings]
        
        self.global_social_embeddings = tf.reduce_sum(all_social_embeddings_global, axis=0)
        self.local_social_embeddings = tf.reduce_sum(all_social_embeddings_local, axis=0)
        return self.global_social_embeddings, self.local_social_embeddings

    def get_global_user_rep(self, socail_ppr_mat):
        """
        socail_ppr_mat: sp_mat
        social_local_mat: sp_mat
        """
        socail_ppr_mat = self._convert_sp_mat_to_sp_tensor(socail_ppr_mat)
        edge_embeddings_local = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_social_embeddings_local = [edge_embeddings_local]
        
        for k in range(self.n_layers):
            # friend view, local
            edge_embeddings_local = tf.sparse_tensor_dense_matmul(socail_ppr_mat, edge_embeddings_local)
            norm_embeddings = tf.math.l2_normalize(edge_embeddings_local, axis=1)
            all_social_embeddings_local += [norm_embeddings]
        

        local_social_embeddings = tf.reduce_sum(all_social_embeddings_local, axis=0)
        return local_social_embeddings
    
    def _create_variable(self):
        self.sub_mat = {}
        self.sub_mat['adj_values_sub'] = tf.placeholder(tf.float32)
        self.sub_mat['adj_indices_sub'] = tf.placeholder(tf.int64)
        self.sub_mat['adj_shape_sub'] = tf.placeholder(tf.int64)
        self.sub_mat['sub_mat'] = tf.SparseTensor(
            self.sub_mat['adj_indices_sub'],
            self.sub_mat['adj_values_sub'],
            self.sub_mat['adj_shape_sub'])

    def adj_normalize(self, mx):
        """
        Row-normalize sparse matrix
        """
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def cal_ppr_social_mat(self, weight=0.15, type_enc="ppr", bidirectional=False):
        def normalization(M):
            rowsum = np.array(M.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(M)
            return norm_adj_tmp.dot(d_mat_inv)
        
        from numpy.linalg import inv
        n_nodes = self.num_users
        s_row_idx = [self.data.user[pair[0]] for pair in self.social.relation]
        s_col_idx = [self.data.user[pair[1]] for pair in self.social.relation]
        # relations = np.ones_like(s_row_idx + s_col_idx, dtype=np.float32)
        relations = np.ones_like(s_row_idx, dtype=np.float32)
        social_mat = sp.csr_matrix((relations, (s_row_idx, s_col_idx)), shape=(n_nodes, n_nodes)) #social 对称的.
        # social_mat = sp.csr_matrix((relations, (s_row_idx + s_col_idx, s_col_idx + s_row_idx)), shape=(n_nodes, n_nodes))

        #bidirection
        # if bidirectional == True:
        #     social_mat_v1 = social_mat.multiply(social_mat.T)
        #     pdb.set_trace()
        adj = social_mat.tocoo()
        # c = 0.15
        if type_enc == "ppr":
            eigen_adj = weight * inv((sp.eye(adj.shape[0]) - (1 - weight) * self.adj_normalize(adj)).toarray()) #array format
        elif type_enc == "hk":
            rowsum = np.array(adj.sum(1))
            r_inv = np.power(rowsum, -1.0).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = adj.dot(r_mat_inv).toarray()
            eigen_adj = np.exp(weight * mx - weight)
            # adj_normalize
        elif type_enc == 'origin':
            eigen_adj = adj.toarray()
        
        social_mat = normalization(social_mat)  
        return eigen_adj, social_mat

    
    def cal_ppr_common(self, social_mat, weight=0.15):
        from numpy.linalg import inv
        def normalization(M):
            rowsum = np.array(M.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(M)
            return norm_adj_tmp.dot(d_mat_inv)
        adj = social_mat.tocoo()
        # c = 0.15
        eigen_adj = weight * inv((sp.eye(adj.shape[0]) - (1 - weight) * self.adj_normalize(adj)).toarray()) #array format
        social_mat = normalization(social_mat)
        return eigen_adj, social_mat

    def cal_ppr_interact_mat(self, weight=0.15):
        def normalization(M):
            rowsum = np.array(M.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(M)
            return norm_adj_tmp.dot(d_mat_inv)
        
        from numpy.linalg import inv
        n_nodes = self.num_users + self.num_items
        row_idx = [self.data.user[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.item[pair[1]] for pair in self.data.trainingData]
        # s_row_idx = [self.data.user[pair[0]] for pair in self.social.relation]
        # s_col_idx = [self.data.user[pair[1]] for pair in self.social.relation]
        s_row_idx = row_idx + col_idx
        s_col_idx = col_idx + row_idx
        relations = np.ones_like(s_row_idx, dtype=np.float32)
        social_mat = sp.csr_matrix((relations, (s_row_idx, s_col_idx)), shape=(n_nodes, n_nodes))
        adj = social_mat.tocoo()
        # c = 0.15
        eigen_adj = weight * inv((sp.eye(adj.shape[0]) - (1 - weight) * self.adj_normalize(adj)).toarray()) #array format
        social_mat = normalization(social_mat)
        return eigen_adj, social_mat

    def get_interaction_uu_ii(self, uu_weight=8, ii_weight=5):
        n_nodes = self.num_users + self.num_items
        row_idx = [self.data.user[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.item[pair[1]] for pair in self.data.trainingData]
        # s_row_idx = [self.data.user[pair[0]] for pair in self.social.relation]
        # s_col_idx = [self.data.user[pair[1]] for pair in self.social.relation]
        # if is_subgraph and self.drop_rate > 0:
        # keep_idx = random.sample(list(range(self.data.elemCount())), int(self.data.elemCount() * (1 - self.drop_rate)))
        # user_np = np.array(row_idx)[keep_idx]
        # item_np = np.array(col_idx)[keep_idx]
        ratings = np.ones_like(row_idx, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (row_idx, col_idx)), shape=(self.num_users, self.num_items))
        uu_adj = tmp_adj @ tmp_adj.T #(user_num, user_num)
        ii_adj = tmp_adj.T @ tmp_adj   #(item_num, item_num)

        uu_coo = uu_adj.tocoo().astype(np.float32)
        assert len(uu_coo.row) == len(uu_coo.col) and len(uu_coo.col) == len(uu_coo.data)
        uu_index = np.nonzero(uu_coo.data > uu_weight)[0]
        uu_ratings = np.ones_like(uu_index, dtype=np.float32)
        uu_adj_crop = sp.csr_matrix((uu_ratings, (uu_coo.row[uu_index], uu_coo.col[uu_index])), shape=uu_coo.shape)
        # indices = [uu_coo.row, uu_coo.col, coo.data]
        # return tf.SparseTensor(indices, coo.data, coo.shape)

        ii_coo = ii_adj.tocoo().astype(np.float32)
        assert len(ii_coo.row) == len(ii_coo.col) and len(ii_coo.col) == len(ii_coo.data)
        ii_index = np.nonzero(ii_coo.data > ii_weight)[0]
        ii_ratings = np.ones_like(ii_index, dtype=np.float32)
        ii_adj_crop = sp.csr_matrix((ii_ratings, (ii_coo.row[ii_index], ii_coo.col[ii_index])), shape=ii_coo.shape)


        # test, social uu
        # from numpy.linalg import inv
        # n_nodes = self.num_users
        # s_row_idx = [self.data.user[pair[0]] for pair in self.social.relation]
        # s_col_idx = [self.data.user[pair[1]] for pair in self.social.relation]
        # # relations = np.ones_like(s_row_idx + s_col_idx, dtype=np.float32)
        # relations = np.ones_like(s_row_idx, dtype=np.float32)
        # social_mat = sp.csr_matrix((relations, (s_row_idx, s_col_idx)), shape=(n_nodes, n_nodes))

        # pdb.set_trace()
        return uu_adj_crop, ii_adj_crop
    
    def get_adj_mat(self, is_subgraph=False):
        n_nodes = self.num_users + self.num_items
        row_idx = [self.data.user[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.item[pair[1]] for pair in self.data.trainingData]
        s_row_idx = [self.data.user[pair[0]] for pair in self.social.relation]
        s_col_idx = [self.data.user[pair[1]] for pair in self.social.relation]
        if is_subgraph and self.drop_rate > 0:
            keep_idx = random.sample(list(range(self.data.elemCount())), int(self.data.elemCount() * (1 - self.drop_rate)))
            user_np = np.array(row_idx)[keep_idx]
            item_np = np.array(col_idx)[keep_idx]
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, self.num_users + item_np)), shape=(n_nodes, n_nodes))
            adj_mat = tmp_adj + tmp_adj.T

            #add edge masked social net
            skeep_idx = random.sample(list(range(len(s_row_idx))), int(len(s_row_idx) * (1 - self.drop_rate)))
            follower_np = np.array(s_row_idx)[skeep_idx]
            followee_np = np.array(s_col_idx)[skeep_idx]
            relations = np.ones_like(follower_np, dtype=np.float32)
            social_mat = sp.csr_matrix((relations, (follower_np, followee_np)), shape=(n_nodes, n_nodes))
            social_mat = social_mat.multiply(social_mat)
            #adj_mat = adj_mat+social_mat # remove social-related edegs.
        else:
            user_np = np.array(row_idx)
            item_np = np.array(col_idx)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
            adj_mat = tmp_adj + tmp_adj.T

            follower_np = np.array(s_row_idx)
            followee_np = np.array(s_col_idx)
            relations = np.ones_like(follower_np, dtype=np.float32)
            social_mat = sp.csr_matrix((relations, (follower_np, followee_np)), shape=(n_nodes, n_nodes))
            social_mat = social_mat.multiply(social_mat)
            adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape


    def sampleTopkUsers(self, userEmbedding, top_k=10):
        """
        return 
            (user_num, dim)
        """
        # mask eye
        # user_embeddings = tf.nn.embedding_lookup(userEmbedding, tf.unique(self.u_idx)[0])
        # user_embeddings = tf.nn.l2_normalize(user_embeddings, axis=1) #(user_num, dim)

        # sample_prob = self.social_ppr_mat + -1e5 * tf.eye(self.num_users)
        sample_prob = self.social_ppr_mat
        # sample_prob = self.social_ppr_mat
        sample_prob = tf.nn.softmax(sample_prob)
        # topKUsers = t.multinomial(sample_prob, top_k) #(user_num, topk)
        indices = tf.math.top_k(sample_prob, top_k)[1]
        # pdb.set_trace()
        topKuserEmbs = tf.nn.embedding_lookup(userEmbedding, indices)
        # topKuserEmbs = userEmbedding[topKUsers] #(user_num, topk, dim)
        return tf.reduce_sum(topKuserEmbs, axis=1) #(user_num, dim)
    

    def sampleTopkUsersKeepOri(self, userEmbedding, top_k=10, social_ppr_mat=None, mask_ori=False, add_norm=True, add_self=True, social_layer_num=1):
        """
        self.bs_matrix: csr mat
        social_ppr_mat: array
        return 
            (user_num, dim)
        """
        # mask eye
        # user_embeddings = tf.nn.embedding_lookup(userEmbedding, tf.unique(self.u_idx)[0])
        # user_embeddings = tf.nn.l2_normalize(user_embeddings, axis=1) #(user_num, dim)
        # social_mat = self._convert_sp_mat_to_sp_tensor(self.bs_matrix)
        # social_mat = tf.sparse.to_dense(social_mat) 
        # pdb.set_trace()
        # sample_prob = tf.multiply(self.social_ppr_mat, tf.sparse.to_dense(social_mat))
        
        bs_matrix = self.bs_matrix.toarray()
        # extract from the ppr mat
        # indices = tf.math.top_k(social_ppr_mat , top_k)[1]

        # ii, _ = tf.meshgrid(tf.range(self.num_users), tf.range(top_k), indexing='ij')
        # full_indices = tf.reshape(tf.stack([ii, indices], axis=-1), [-1, len(self.social_ppr_mat.shape)])

        # tensor = tf.zeros_like(social_mat, dtype=tf.float32)
        # updates = tf.ones(full_indices.shape[0])
        # ppr_mask_mat = tf.tensor_scatter_update(tensor, full_indices, updates)

        # extract from the ppr mat
        # social_ppr_mat = social_ppr_mat.toarray()
        indices = np.argsort(social_ppr_mat, axis=-1)[:,-top_k:] #(user_num, top k)

        user_index = np.arange(self.num_users).reshape(self.num_users, 1)
        user_index = np.tile(user_index, top_k).reshape(1, -1)

        # pdb.set_trace()
        # b = np.array([[0, 3], [0, 9], [0, 5]]).reshape(1, -1)
        indices = indices.reshape(1, -1)
        indices = np.concatenate([user_index, indices], axis=0).tolist()
        # a = [[1,2,3], [4,5,6]]
        ppr_mask_mat = np.zeros_like(social_ppr_mat)
        ppr_mask_mat[indices] = 1

        if mask_ori:
            # true neighbors, mask origin social graph.
            ppr_mask_mat = ppr_mask_mat * bs_matrix 

        if add_norm:
            # calculat norm weights
            rowsum = np.array(ppr_mask_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(ppr_mask_mat)
        else:
            norm_adj_tmp = np.array(ppr_mask_mat)
        # sample_prob = self.social_ppr_mat
        # sample_prob = tf.nn.softmax(sample_prob)
        # # topKUsers = t.multinomial(sample_prob, top_k) #(user_num, topk)
        # indices = tf.math.top_k(sample_prob, top_k)[1]
        # pdb.set_trace()

        # gcn, 多层
        # pdb.set_trace()
        norm_adj_tmp = sp.csr_matrix(norm_adj_tmp)
        socail_ppr_mat = self._convert_sp_mat_to_sp_tensor(norm_adj_tmp)

        # userEmbedding = userEmbedding * tf.nn.sigmoid(tf.layers.dense(inputs=userEmbedding, units=self.emb_size, activation=None))
        userEmbedding = userEmbedding

        if add_self==True:
            all_social_embeddings_global = [userEmbedding]
        else:
            all_social_embeddings_global = []

        # for k in range(self.n_layers):
        for k in range(social_layer_num):
            # friend view, local
            friend_view_embeddings_local = tf.sparse_tensor_dense_matmul(socail_ppr_mat, userEmbedding)
            norm_embeddings = tf.math.l2_normalize(friend_view_embeddings_local, axis=1)
            all_social_embeddings_global += [norm_embeddings]
        
        subgraph_emb = tf.reduce_sum(all_social_embeddings_global, axis=0)
        # self.local_social_embeddings = tf.reduce_sum(all_social_embeddings_local, axis=0)
        # subgraph_emb = tf.sparse_tensor_dense_matmul(socail_ppr_mat, userEmbedding)
        # norm_adj_tmp
        # topKuserEmbs = tf.nn.embedding_lookup(userEmbedding, indices)
        # topKuserEmbs = userEmbedding[topKUsers] #(user_num, topk, dim)
        
        # return tf.reduce_sum(topKuserEmbs, axis=1) #(user_num, dim)
        # pdb.set_trace()
        return subgraph_emb


    def sampleTopkUsersKeepOriFromInteraction(self, userEmbedding, top_k=10, social_ppr_mat=None, mask_ori=False, add_norm=True, add_self=True, social_layer_num=1):
        """
        self.bs_matrix: csr mat
        social_ppr_mat: array
        return 
            (user_num, dim)
        """
        
        social_ppr_mat = tf.matmul(userEmbedding, tf.transpose(userEmbedding, perm=[1, 0])) #(num_user, num_user)

        social_mat = self._convert_sp_mat_to_sp_tensor(self.bs_matrix)
        social_mat = tf.sparse.to_dense(social_mat) 

        # extract from the ppr mat
        indices = tf.math.top_k(social_ppr_mat , top_k)[1]

        ii, _ = tf.meshgrid(tf.range(self.num_users), tf.range(top_k), indexing='ij')
        full_indices = tf.reshape(tf.stack([ii, indices], axis=-1), [-1, len(self.social_ppr_mat.shape)])

        tensor = tf.zeros_like(social_mat, dtype=tf.float32)
        updates = tf.ones(full_indices.shape[0])
        ppr_mask_mat = tf.tensor_scatter_update(tensor, full_indices, updates)

        if mask_ori:
            # true neighbors, mask origin social graph.
            ppr_mask_mat = tf.multiply(ppr_mask_mat, social_mat)
        
        if add_norm:
            # calculat norm weights
            rowsum = tf.reduce_sum(ppr_mask_mat, axis=1)
            rowsum = tf.math.pow(rowsum, tf.constant(-0.5, dtype=tf.float32))
            
            d_inv = tf.where(tf.math.is_nan(rowsum), tf.zeros_like(rowsum), rowsum) # exclude nan
            d_inv = tf.where(tf.math.is_inf(d_inv), tf.zeros_like(d_inv), d_inv)  # exclude inf

            d_inv = tf.reshape(d_inv, [-1])
            d_mat_inv = tf.linalg.diag(d_inv)
            norm_adj_tmp = tf.matmul(d_mat_inv, ppr_mask_mat)

            # rowsum = np.array(ppr_mask_mat.sum(1))
            # d_inv = np.power(rowsum, -0.5).flatten()
            # d_inv[np.isinf(d_inv)] = 0.
            # d_mat_inv = sp.diags(d_inv)
            # norm_adj_tmp = d_mat_inv.dot(ppr_mask_mat)
        else:
            norm_adj_tmp = ppr_mask_mat

        # convert socail_ppr_mat to sparse tensor.
        socail_ppr_mat = tf.sparse.from_dense(norm_adj_tmp)

        userEmbedding = userEmbedding

        if add_self==True:
            all_social_embeddings_global = [userEmbedding]
        else:
            all_social_embeddings_global = []

        # for k in range(self.n_layers):
        for k in range(social_layer_num):
            # friend view, local
            friend_view_embeddings_local = tf.sparse_tensor_dense_matmul(socail_ppr_mat, userEmbedding)
            norm_embeddings = tf.math.l2_normalize(friend_view_embeddings_local, axis=1)
            all_social_embeddings_global += [norm_embeddings]
        
        subgraph_emb = tf.reduce_sum(all_social_embeddings_global, axis=0)

        return subgraph_emb
    

    def sampleTopkUsersKeepOriFromInteractionUUMat(self, userEmbedding, top_k=10, social_ppr_mat=None, mask_ori=False, add_norm=True, add_self=True, social_layer_num=1):
        """
        self.bs_matrix: csr mat
        social_ppr_mat: array
        return 
            (user_num, dim)
        """
        
        bs_matrix = self.bs_matrix.toarray()

        indices = np.argsort(social_ppr_mat, axis=-1)[:,-top_k:] #(user_num, top k)

        user_index = np.arange(self.num_users).reshape(self.num_users, 1)
        user_index = np.tile(user_index, top_k).reshape(1, -1)

        # pdb.set_trace()
        # b = np.array([[0, 3], [0, 9], [0, 5]]).reshape(1, -1)
        indices = indices.reshape(1, -1)
        indices = np.concatenate([user_index, indices], axis=0).tolist()
        # a = [[1,2,3], [4,5,6]]
        ppr_mask_mat = np.zeros_like(social_ppr_mat)
        ppr_mask_mat[indices] = 1

        if mask_ori:
            # true neighbors, mask origin social graph.
            ppr_mask_mat = ppr_mask_mat * bs_matrix 

        if add_norm:
            # calculat norm weights
            rowsum = np.array(ppr_mask_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(ppr_mask_mat)
        else:
            norm_adj_tmp = np.array(ppr_mask_mat)
        # sample_prob = self.social_ppr_mat
        # sample_prob = tf.nn.softmax(sample_prob)
        # # topKUsers = t.multinomial(sample_prob, top_k) #(user_num, topk)
        # indices = tf.math.top_k(sample_prob, top_k)[1]
        # pdb.set_trace()

        # gcn, 多层
        # pdb.set_trace()
        norm_adj_tmp = sp.csr_matrix(norm_adj_tmp)
        socail_ppr_mat = self._convert_sp_mat_to_sp_tensor(norm_adj_tmp)

        # userEmbedding = userEmbedding * tf.nn.sigmoid(tf.layers.dense(inputs=userEmbedding, units=self.emb_size, activation=None))
        userEmbedding = userEmbedding

        if add_self==True:
            all_social_embeddings_global = [userEmbedding]
        else:
            all_social_embeddings_global = []

        # for k in range(self.n_layers):
        for k in range(social_layer_num):
            # friend view, local
            friend_view_embeddings_local = tf.sparse_tensor_dense_matmul(socail_ppr_mat, userEmbedding)
            norm_embeddings = tf.math.l2_normalize(friend_view_embeddings_local, axis=1)
            all_social_embeddings_global += [norm_embeddings]
        
        subgraph_emb = tf.reduce_sum(all_social_embeddings_global, axis=0)
        # self.local_social_embeddings = tf.reduce_sum(all_social_embeddings_local, axis=0)
        # subgraph_emb = tf.sparse_tensor_dense_matmul(socail_ppr_mat, userEmbedding)
        # norm_adj_tmp
        # topKuserEmbs = tf.nn.embedding_lookup(userEmbedding, indices)
        # topKuserEmbs = userEmbedding[topKUsers] #(user_num, topk, dim)
        
        # return tf.reduce_sum(topKuserEmbs, axis=1) #(user_num, dim)
        # pdb.set_trace()
        return subgraph_emb

    
    def gcn_on_social_net(self, social_local_mat, input_embeddings):
        """
        socail_ppr_mat: sp_mat
        social_local_mat: sp_mat
        """
        # socail_ppr_mat = self._convert_sp_mat_to_sp_tensor(socail_ppr_mat)
        social_local_mat = self._convert_sp_mat_to_sp_tensor(social_local_mat)
        # add self-gating
        # user_embeedings_v1 = self.user_embeddings * tf.nn.sigmoid(tf.layers.dense(inputs=input_embeddings, units=self.emb_size, activation=None))
        user_embeedings_v1 = input_embeddings
        # user_embeedings_v2 = self.user_embeddings * tf.nn.sigmoid(tf.layers.dense(inputs=self.user_embeddings, units=self.emb_size, activation=None))
        # friend_view_embeddings_global, friend_view_embeddings_local = user_embeedings_v1, user_embeedings_v2
        friend_view_embeddings_local = user_embeedings_v1
        all_social_embeddings_local = [user_embeedings_v1]
        # all_social_embeddings_local = []
        # all_social_embeddings_global = [user_embeedings_v2]
        
        # all_social_embeddings_local = []
        # all_social_embeddings_global = []
        # friend view, global
        # friend_view_embeddings_global = tf.sparse_tensor_dense_matmul(socail_ppr_mat, friend_view_embeddings_global)
        # norm_embeddings = tf.math.l2_normalize(friend_view_embeddings_global, axis=1)
        # all_social_embeddings_global += [norm_embeddings]
        
        for k in range(self.n_layers):
            # friend view, local
            friend_view_embeddings_local = tf.sparse_tensor_dense_matmul(social_local_mat, friend_view_embeddings_local)
            norm_embeddings = tf.math.l2_normalize(friend_view_embeddings_local, axis=1)
            all_social_embeddings_local += [norm_embeddings]
        
        # self.global_social_embeddings = tf.reduce_sum(all_social_embeddings_global, axis=0)
        local_social_embeddings = tf.reduce_sum(all_social_embeddings_local, axis=0)
        return local_social_embeddings

        # return 

    
    def graph_partition(self, adjacency_mat, n=10):
        """
        adjacency_mat: list of numpy
        graph partitioning
        """

        import numpy as np
        import pymetis
        # convert numpy mat to indice
        adjacency_list = []
        # pdb.set_trace()
        for adj_row in adjacency_mat:
            adjacency_list.append(np.nonzero(adj_row)[0])
        n_cuts, ss_labels = pymetis.part_graph(n, adjacency=adjacency_list)
        return ss_labels

    def initModel(self):
        super(SEPTV2_add_social_rep, self).initModel()
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self._create_variable()
        self.bs_matrix = self.get_birectional_social_matrix()
        self.rating_mat = self.buildSparseRatingMatrix()
        social_mat, sharing_mat = self.get_social_related_views(self.bs_matrix, self.rating_mat)
        social_mat = self._convert_sp_mat_to_sp_tensor(social_mat)
        sharing_mat = self._convert_sp_mat_to_sp_tensor(sharing_mat)
        self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.emb_size], stddev=0.005), name='U') / 2
        self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.emb_size], stddev=0.005), name='V') / 2
        # initialize adjacency matrices
        ui_mat = self.create_joint_sparse_adj_tensor()
        friend_view_embeddings = self.user_embeddings
        sharing_view_embeddings = self.user_embeddings
        all_social_embeddings = [friend_view_embeddings]
        all_sharing_embeddings = [sharing_view_embeddings]
        ego_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = [ego_embeddings]
        aug_embeddings = ego_embeddings
        all_aug_embeddings = [ego_embeddings]
        # pdb.set_trace()
        #multi-view convolution: LightGCN structure
        for k in range(self.n_layers):
            # friend view
            friend_view_embeddings = tf.sparse_tensor_dense_matmul(social_mat,friend_view_embeddings)
            norm_embeddings = tf.math.l2_normalize(friend_view_embeddings, axis=1)
            all_social_embeddings += [norm_embeddings]
            # sharing view
            sharing_view_embeddings = tf.sparse_tensor_dense_matmul(sharing_mat,sharing_view_embeddings)
            norm_embeddings = tf.math.l2_normalize(sharing_view_embeddings, axis=1)
            all_sharing_embeddings += [norm_embeddings]
            # preference view
            ego_embeddings = tf.sparse_tensor_dense_matmul(ui_mat, ego_embeddings)
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
            # unlabeled sample view
            aug_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat'], aug_embeddings)
            norm_embeddings = tf.math.l2_normalize(aug_embeddings, axis=1)
            all_aug_embeddings += [norm_embeddings]

        # multi-view convolution: NGCF structure
        # initializer = tf.contrib.layers.xavier_initializer()
        # self.weights = dict()
        # for k in range(self.n_layers):
        #     for view in range(4):
        #         self.weights['W_%d_1_%d' %(k,view)] = tf.Variable(
        #             initializer([self.emb_size,self.emb_size]), name='W_%d_1_%d' %(k,view))
        #         self.weights['W_%d_2_%d' %(k,view)] = tf.Variable(
        #             initializer([self.emb_size,self.emb_size]), name='W_%d_2_%d' %(k,view))
        #
        # for k in range(self.n_layers):
        #     #friend view
        #     side_embeddings = tf.sparse_tensor_dense_matmul(social_mat,friend_view_embeddings)
        #     sum_embeddings = tf.matmul(side_embeddings+friend_view_embeddings, self.weights['W_%d_1_0' % k])
        #     bi_embeddings = tf.multiply(friend_view_embeddings, side_embeddings)
        #     bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_%d_2_0' % k])
        #     friend_view_embeddings = tf.nn.leaky_relu(sum_embeddings+bi_embeddings)
        #     norm_embeddings = tf.math.l2_normalize(friend_view_embeddings, axis=1)
        #     all_social_embeddings += [norm_embeddings]
        #     #sharing view
        #     side_embeddings = tf.sparse_tensor_dense_matmul(sharing_mat,sharing_view_embeddings)
        #     sum_embeddings = tf.matmul(side_embeddings+sharing_view_embeddings, self.weights['W_%d_1_1' % k])
        #     bi_embeddings = tf.multiply(sharing_view_embeddings, side_embeddings)
        #     bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_%d_2_1' % k])
        #     sharing_view_embeddings = tf.nn.leaky_relu(sum_embeddings+bi_embeddings)
        #     norm_embeddings = tf.math.l2_normalize(sharing_view_embeddings, axis=1)
        #     all_sharing_embeddings += [norm_embeddings]
        #     #preference view
        #     side_embeddings = tf.sparse_tensor_dense_matmul(ui_mat, ego_embeddings)
        #     sum_embeddings = tf.matmul(side_embeddings+ego_embeddings, self.weights['W_%d_1_2' % k])
        #     bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
        #     bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_%d_2_2' % k])
        #     ego_embeddings = tf.nn.leaky_relu(sum_embeddings+bi_embeddings)
        #     norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
        #     all_embeddings += [norm_embeddings]
        #     # unlabeled sample view
        #     side_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat'], aug_embeddings)
        #     sum_embeddings = tf.matmul(side_embeddings+aug_embeddings, self.weights['W_%d_1_3' % k])
        #     bi_embeddings = tf.multiply(aug_embeddings, side_embeddings)
        #     bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_%d_2_3' % k])
        #     aug_embeddings = tf.nn.leaky_relu(sum_embeddings+bi_embeddings)
        #     norm_embeddings = tf.math.l2_normalize(aug_embeddings, axis=1)
        #     all_aug_embeddings += [norm_embeddings]

        # averaging the view-specific embeddings
        self.friend_view_embeddings = tf.reduce_sum(all_social_embeddings, axis=0)
        self.sharing_view_embeddings = tf.reduce_sum(all_sharing_embeddings, axis=0)
        all_embeddings = tf.reduce_sum(all_embeddings, axis=0)
        self.all_embeddings = all_embeddings
        self.rec_user_embeddings, self.rec_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        aug_embeddings = tf.reduce_sum(all_aug_embeddings, axis=0)
        self.aug_user_embeddings, self.aug_item_embeddings = tf.split(aug_embeddings, [self.num_users, self.num_items], 0)
        

        # if self.ppr_rate != 0.0:
        # add social-based ppr mat
        social_ppr_mat, social_ppr_sp_mat = self.cal_ppr_social_mat()

        # social_ppr_mat, social_ppr_sp_mat = self.cal_ppr_social_mat(type='hk', weight=0.7)
        # social_ppr_mat, social_ppr_sp_mat = self.cal_ppr_social_mat(type='origin')
        # social_mat = self._convert_sp_mat_to_sp_tensor(social_ppr_mat)
        # social_mat = tf.sparse.to_dense(social_mat)
        self.social_ppr_mat = tf.convert_to_tensor(social_ppr_mat, dtype=tf.float32) #(userNum, userNum)
        self.user_user_sim = tf.matmul(self.rec_user_embeddings, tf.transpose(self.rec_user_embeddings, perm=[1, 0])) #(userNum, userNum)
        # pdb.set_trace()

        if self.s_cl_rate != 0.0:
            # add social based CL loss, performance is not good.
            self.bi_social_matrix = self.get_birectional_social_matrix_v2()
            self.global_social_user_emb, self.local_social_user_emb = self.get_local_global_user_rep(social_ppr_sp_mat, self.bi_social_matrix)

        # if self.social_ppr_cluster_w != 0.0:
        # add top k merge emb
        # self.social_ppr_cluster_emb = self.sampleTopkUsers(self.user_embeddings)
        # self.cluster_type = 1
        if self.cluster_type == 1: #top_10 is better than top_5 and top_20.
            self.social_ppr_cluster_emb = self.sampleTopkUsers(self.rec_user_embeddings, top_k=10) # context embedding. not good. 模型是有效的, 说明聚合操作可以深挖; how to cluster users? 通过graph partition得到每个节点的标签, 根据标签得到聚合表征;
        elif self.cluster_type == 2: # todo
            self.social_ppr_cluster_emb = tf.tile(tf.expand_dims(tf.reduce_mean(self.rec_user_embeddings, axis=0), 0), [self.num_users,1]) # global embedding. not good.
            self.item_ppr_cluster_emb = tf.tile(tf.expand_dims(tf.reduce_mean(self.rec_item_embeddings, axis=0), 0), [self.num_items,1]) # global embedding. not good.
            self.edgo_ppr_cluster_emb = tf.concat([self.social_ppr_cluster_emb, self.item_ppr_cluster_emb], axis=0) #(num_user + num_item, dim)
        elif self.cluster_type == 3: # same as cluster_type: 1
            # self.social_ppr_cluster_emb = self.sampleTopkUsersKeepOri(self.rec_user_embeddings, top_k=5, mask_ori=True, social_ppr_mat=social_ppr_mat)
            # self.social_ppr_cluster_emb = self.sampleTopkUsersKeepOri(self.rec_user_embeddings, top_k=10, mask_ori=True, social_ppr_mat=social_ppr_mat)
            # self.social_ppr_cluster_emb = self.sampleTopkUsersKeepOri(self.rec_user_embeddings, top_k=10, mask_ori=True, social_ppr_mat=social_ppr_mat, add_norm=False)
            # self.social_ppr_cluster_emb = self.sampleTopkUsersKeepOri(self.rec_user_embeddings, top_k=10, mask_ori=True, social_ppr_mat=social_ppr_mat, add_norm=True) # best result
            # self.social_ppr_cluster_emb = self.sampleTopkUsersKeepOri(self.rec_user_embeddings, top_k=10, social_ppr_mat=social_ppr_mat, mask_ori=True,  add_norm=True, add_self=False)
            self.social_ppr_cluster_emb = self.sampleTopkUsersKeepOri(self.rec_user_embeddings, top_k=10, social_ppr_mat=social_ppr_mat, mask_ori=True, add_norm=True, add_self=True, social_layer_num=2) #目前的sota.
            self.inter_flag = True
            if self.inter_flag == True:
                self.interaction_cluster_emb = self.sampleTopkUsersKeepOriFromInteraction(self.rec_user_embeddings, top_k=10, social_ppr_mat=None, mask_ori=True, add_norm=True, add_self=True, social_layer_num=2)
                # self.interaction_cluster_emb = self.sampleTopkUsersKeepOriFromInteraction(self.rec_user_embeddings, top_k=10, social_ppr_mat=None, mask_ori=True, add_norm=False, add_self=True, social_layer_num=2)
        elif self.cluster_type == 4: # worse than cluster_type_1
            self.inter_flag = False
            nor_social_mat = self.get_birectional_social_matrix_v2()
            self.social_ppr_cluster_emb = self.gcn_on_social_net(nor_social_mat, self.rec_user_embeddings)
        elif self.cluster_type == 5: #worse than cluster_type_1
            self.social_ppr_cluster_emb = self.sampleTopkUsers(self.rec_user_embeddings, top_k=10) # context embedding. not good. 模型是有效的, 说明聚合操作可以深挖; how to cluster users? 通过graph partition得到每个节点的标签, 根据标签得到聚合表征;
            self.social_global_cluster_emb = tf.tile(tf.expand_dims(tf.reduce_sum(self.social_ppr_cluster_emb, axis=0), 0), [self.num_users,1]) # [num_user, dim]
        elif self.cluster_type == 6:
            self.social_ppr_cluster_emb = self.sampleTopkUsersKeepOri(self.rec_user_embeddings, top_k=10, social_ppr_mat=social_ppr_mat, mask_ori=True, add_norm=True, add_self=True, social_layer_num=2) #目前的sota.
            # add interactioin uu cluster emb
            self.inter_flag = False
            # todo
            # uu_inter_mat, ii_inter_mat = self.get_interaction_uu_ii(uu_weight=0, ii_weight=0) # csr mat; 调整下不同的参数的效果;
            # uu_inter_ppr_mat, uu_inter_ppr_sp_mat =  self.cal_ppr_common(uu_inter_mat) # good results.
            # uu_inter_ppr_mat = uu_inter_ppr_mat.toarray()
            # self.interaction_cluster_emb = self.sampleTopkUsersKeepOriFromInteractionUUMat(self.rec_user_embeddings, top_k=10, social_ppr_mat=uu_inter_ppr_mat, mask_ori=True, add_norm=True, add_self=True, social_layer_num=2)
                
        if self.rec_loss_aug_w != 0.0:
            # add dropout edged grpahs
            self.batch_user_aug_emb = tf.nn.embedding_lookup(self.aug_user_embeddings, self.u_idx)
            self.batch_pos_item_aug_emb = tf.nn.embedding_lookup(self.aug_item_embeddings, self.v_idx)
            self.batch_neg_item_aug_emb = tf.nn.embedding_lookup(self.aug_item_embeddings, self.neg_idx)

        # interaction-based loss
        if self.interact_ppr_w != 0.0:
            interact_ppr_mat, interact_ppr_sp_mat = self.cal_ppr_interact_mat()
            self.interact_ppr_mat = tf.convert_to_tensor(interact_ppr_mat, dtype=tf.float32) #(userNum, userNum)
            self.edgo_edgo_sim = tf.matmul(all_embeddings, tf.transpose(all_embeddings, perm=[1, 0])) #(userNum, userNum)
        # self.edgo_aug_v2_emb = self.get_global_user_rep(interact_ppr_sp_mat)

        if self.inter_ppr_w != 0.0:
            # add uu ppr and ii pprdd
            self.user_user_aug_sim = tf.matmul(self.aug_user_embeddings, tf.transpose(self.aug_user_embeddings, perm=[1, 0])) #(userNum, userNum)
            uu_inter_mat, ii_inter_mat = self.get_interaction_uu_ii(uu_weight=0, ii_weight=0)
            uu_inter_ppr_mat, uu_inter_ppr_sp_mat =  self.cal_ppr_common(uu_inter_mat) # good results.
            # ii_inter_ppr_mat, ii_inter_ppr_sp_mat =  self.cal_ppr_common(ii_inter_mat)
            # uu_inter_ppr_mat = uu_inter_mat.toarray() * uu_inter_ppr_mat # origin matric, bad results.
            # uu_inter_ppr_mat = uu_inter_mat.toarray() # origin matric, bad results.
            # ii_inter_ppr_mat = ii_inter_mat.toarray() # origin matric
            self.uu_inter_ppr_mat = tf.convert_to_tensor(uu_inter_ppr_mat, dtype=tf.float32) #(userNum, userNum)
            # self.ii_inter_ppr_mat = tf.convert_to_tensor(ii_inter_ppr_mat, dtype=tf.float32) #(userNum, userNum)
            # self.item_item_sim = tf.matmul(self.rec_item_embeddings, tf.transpose(self.rec_item_embeddings, perm=[1, 0])) #(userNum, userNum)

        if self.graph_label_w != 0.0:
            self.graph_label_num = self.num_users // 50
            graph_label = self.graph_partition(self.bs_matrix.toarray(), self.graph_label_num)
            self.graph_label = tf.Variable(graph_label, trainable=False)
            self.graph_batch_label = tf.nn.embedding_lookup(self.graph_label, self.u_idx)

            # pdb.set_trace()

        # if self.inter_cl_w != 0.0:
        self.ppr_user_emb_fi = tf.matmul(self.social_ppr_mat, self.rec_user_embeddings)
        # self.rec_user_embeddings = self.social_ppr_cluster_emb

        # add type 
        # self.rec_user_embeddings = self.social_ppr_cluster_emb + (self.rec_user_embeddings + self.ppr_user_emb_fi) / 2.0
        # mean type
        # self.rec_user_embeddings = (self.social_ppr_cluster_emb + self.rec_user_embeddings + self.ppr_user_emb_fi) / 3.0
        # concat type
        initializer = tf.contrib.layers.xavier_initializer()
        self.weights_v1 = tf.Variable(initializer([self.emb_size *3, self.emb_size]), name='output_weight_v1')
        self.rec_user_embeddings = tf.matmul(tf.concat([self.social_ppr_cluster_emb, (self.rec_user_embeddings + self.ppr_user_emb_fi)/2.0], axis=1), self.weights_v1)

        # embedding look-up
        self.batch_user_emb = tf.nn.embedding_lookup(self.rec_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.rec_item_embeddings, self.v_idx)
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.rec_item_embeddings, self.neg_idx)
      
    def ssl_layer_loss(self, userEmb, userEmbAug, ssl_temp=0.1):
        """
        cl loss in one batch, not in all users.
        one side.
        """
        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), axis=1)
        # current_user_embeddings = userEmb[user]
        user_embeddings = tf.nn.embedding_lookup(userEmb, tf.unique(self.u_idx)[0])
        aug_user_embeddings = tf.nn.embedding_lookup(userEmbAug, tf.unique(self.u_idx)[0])
        user_embeddings = tf.nn.l2_normalize(user_embeddings, axis=1)
        aug_user_embeddings = tf.nn.l2_normalize(aug_user_embeddings, axis=1)
        
        pos = score(user_embeddings, aug_user_embeddings) #(user_num)
        ttl_score = tf.matmul(user_embeddings, aug_user_embeddings, transpose_a=False, transpose_b=True) #(user_num, user_num)
        pos_score = tf.exp(pos / ssl_temp)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / ssl_temp), axis=1)
        ssl_loss = - tf.reduce_sum(tf.log(pos_score / ttl_score))

        return ssl_loss

    def ssl_layer_both_item_loss(self, userEmb, userEmbAug, ssl_temp=0.1):
        """
        cl loss in one batch, not in all users.
        one side.
        """
        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), axis=1)
        # current_user_embeddings = userEmb[user]
        user_embeddings = tf.nn.embedding_lookup(userEmb, tf.unique(self.u_idx)[0])
        aug_user_embeddings = tf.nn.embedding_lookup(userEmbAug, tf.unique(self.u_idx)[0])
        user_embeddings = tf.nn.l2_normalize(user_embeddings, axis=1)
        aug_user_embeddings = tf.nn.l2_normalize(aug_user_embeddings, axis=1)
        
        pos = score(user_embeddings, aug_user_embeddings) #(user_num)
        ttl_score = tf.matmul(user_embeddings, aug_user_embeddings, transpose_a=False, transpose_b=True) #(user_num, user_num)
        pos_score = tf.exp(pos / ssl_temp)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / ssl_temp), axis=1)
        ssl_loss = - tf.reduce_sum(tf.log(pos_score / ttl_score))

        return ssl_loss
    
    def hinge_cl_loss(self, hidden1, summary1):
        r"""Computes the margin objective."""
        def maginLoss(scores_pos, scores_neg, w=1.0):
            loss_matrix = tf.maximum(0., w - scores_pos + scores_neg)  # we could also use tf.nn.relu here
            loss = tf.reduce_sum(loss_matrix)
            return loss
        shuf_index = tf.random.shuffle(tf.range(summary1.shape[0]))

        hidden2 = tf.nn.embedding_lookup(hidden1, shuf_index)
        summary2 = tf.nn.embedding_lookup(summary1, shuf_index)
        
        logits_aa =  tf.math.sigmoid(tf.reduce_sum(hidden1 * summary1, axis = -1))
        logits_bb =  tf.math.sigmoid(tf.reduce_sum(hidden2 * summary2, axis = -1))
        logits_ab =  tf.math.sigmoid(tf.reduce_sum(hidden1 * summary2, axis = -1))
        logits_ba =  tf.math.sigmoid(tf.reduce_sum(hidden2 * summary1, axis = -1))
        
        TotalLoss = 0.0
        TotalLoss += maginLoss(logits_aa, logits_ba)
        TotalLoss += maginLoss(logits_bb, logits_ab)
        
        return TotalLoss
    
    def label_prediction(self, emb):
        """
        user ids in a batch.
        return:
            (userNumInBatch, userNumInBatch)
        """
        emb = tf.nn.embedding_lookup(emb, tf.unique(self.u_idx)[0])
        emb = tf.nn.l2_normalize(emb, axis=1)
        aug_emb = tf.nn.embedding_lookup(self.aug_user_embeddings, tf.unique(self.u_idx)[0])
        aug_emb = tf.nn.l2_normalize(aug_emb, axis=1)
        prob = tf.matmul(emb, aug_emb, transpose_b=True)
        # avoid self-sampling
        # diag = tf.diag_part(prob)
        # prob = tf.matrix_diag(-diag)+prob
        prob = tf.nn.softmax(prob)
        return prob

    def sampling(self, logits):
        return tf.math.top_k(logits, self.instance_cnt)[1]

    def generate_pesudo_labels(self, prob1, prob2):
        positive = (prob1 + prob2) / 2
        pos_examples = self.sampling(positive)
        return pos_examples

    def neighbor_discrimination(self, positive, emb):
        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), axis=2)
        emb = tf.nn.embedding_lookup(emb, tf.unique(self.u_idx)[0])
        emb = tf.nn.l2_normalize(emb, axis=1)
        aug_emb = tf.nn.embedding_lookup(self.aug_user_embeddings, tf.unique(self.u_idx)[0])
        aug_emb = tf.nn.l2_normalize(aug_emb, axis=1)
        pos_emb = tf.nn.embedding_lookup(aug_emb, positive)
        emb2 = tf.reshape(emb, [-1, 1, self.emb_size])
        emb2 = tf.tile(emb2, [1, self.instance_cnt, 1])
        pos = score(emb2, pos_emb)
        ttl_score = tf.matmul(emb, aug_emb, transpose_a=False, transpose_b=True)
        pos_score = tf.reduce_sum(tf.exp(pos / 0.1), axis=1)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / 0.1), axis=1)
        ssl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        return ssl_loss

    def trainModel(self):
        # training the recommendation model, sept source code;
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        # rec_loss += self.regU * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        rec_loss += self.regU * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb) + tf.nn.l2_loss(
                self.batch_neg_item_emb)) #与Lightgcn对齐.

        reg_loss = self.regU * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb) + tf.nn.l2_loss(
                self.batch_neg_item_emb)) #与Lightgcn对齐.
        
        # rec loss
        loss = rec_loss

        if self.ss_rate != 0.0:
            # self-supervision prediction
            social_prediction = self.label_prediction(self.friend_view_embeddings)
            sharing_prediction = self.label_prediction(self.sharing_view_embeddings)
            rec_prediction = self.label_prediction(self.rec_user_embeddings)
            # find informative positive examples for each encoder
            self.f_pos = self.generate_pesudo_labels(sharing_prediction, rec_prediction)
            self.sh_pos = self.generate_pesudo_labels(social_prediction, rec_prediction)
            self.r_pos = self.generate_pesudo_labels(social_prediction, sharing_prediction)
            # neighbor-discrimination based contrastive learning
            self.neighbor_dis_loss = self.neighbor_discrimination(self.f_pos, self.friend_view_embeddings)
            self.neighbor_dis_loss += self.neighbor_discrimination(self.sh_pos, self.sharing_view_embeddings)
            self.neighbor_dis_loss += self.neighbor_discrimination(self.r_pos, self.rec_user_embeddings)
            self.neighbor_dis_loss = self.ss_rate * self.neighbor_dis_loss
            #only rec-based loss
            # neighbor_w = 1.0
            # self.neighbor_dis_loss = self.ss_rate * self.neighbor_discrimination(self.r_pos, self.rec_user_embeddings)
            # self.neighbor_dis_loss = self.ss_rate * self.neighbor_discrimination(self.sh_pos, self.sharing_view_embeddings)
            # self.neighbor_dis_loss = self.ss_rate * self.neighbor_discrimination(self.f_pos, self.friend_view_embeddings)
            loss += self.neighbor_dis_loss

        if self.ppr_rate != 0.0:
            # 1. add social-based ppr loss
            self.social_ppr_loss = self.ppr_rate * tf.losses.mean_squared_error(self.social_ppr_mat, self.user_user_sim)
            loss += self.social_ppr_loss # ppr is good;

        if self.s_cl_rate != 0.0:
            # 2. add social-based loss
            # self.socail_cl_loss_v1 = self.ssl_layer_loss(self.global_social_user_emb, self.local_social_user_emb)
            # self.socail_cl_loss_v2 = self.ssl_layer_loss(self.local_social_user_emb, self.global_social_user_emb)
            self.socail_cl_loss_v1 = self.ssl_layer_loss(self.rec_user_embeddings, self.local_social_user_emb) #v2
            self.socail_cl_loss_v2 = self.ssl_layer_loss(self.rec_user_embeddings, self.global_social_user_emb)
            # self.socail_cl_loss_v1 = self.hinge_cl_loss(self.rec_user_embeddings, self.local_social_user_emb)
            # self.socail_cl_loss_v2 = self.hinge_cl_loss(self.rec_user_embeddings, self.global_social_user_emb)
            self.social_cl_loss_final = self.s_cl_rate * (self.socail_cl_loss_v1 + self.socail_cl_loss_v2)/2
            loss += self.social_cl_loss_final # not good;

        # if self.social_ppr_cluster_w != 0.0:
        #     # 3. add top k user embedding.
        #     if self.cluster_type == 1 or self.cluster_type == 3 or self.cluster_type == 4 or self.cluster_type == 6:
        #         self.social_ppr_cluster_loss = self.social_ppr_cluster_w * self.ssl_layer_loss(self.rec_user_embeddings, self.social_ppr_cluster_emb)
        #         # self.social_ppr_cluster_loss += self.social_ppr_cluster_w * self.ssl_layer_loss(self.rec_item_embeddings, self.item_ppr_cluster_emb)
        #         # self.socail_ppr_cluster_hinge_loss = self.social_ppr_cluster_w * self.hinge_cl_loss(self.user_embeddings, self.social_ppr_cluster_emb)
        #         # loss += self.social_ppr_cluster_w * self.social_ppr_cluster_loss
        #         if self.inter_flag == True:
        #             self.social_ppr_cluster_loss += self.social_ppr_cluster_w * self.ssl_layer_loss(self.rec_user_embeddings, self.interaction_cluster_emb)
            
        #     elif self.cluster_type == 2:
        #         self.social_ppr_cluster_loss = self.social_ppr_cluster_w * self.ssl_layer_both_item_loss(self.edgo_ppr_cluster_emb, self.social_ppr_cluster_emb)
            
        #     elif self.cluster_type == 5:
        #         self.social_ppr_cluster_loss = self.social_ppr_cluster_w * (self.ssl_layer_both_item_loss(self.rec_user_embeddings, self.social_ppr_cluster_emb) + self.ssl_layer_both_item_loss(self.social_ppr_cluster_emb, self.social_global_cluster_emb))
                
        #     loss += self.social_ppr_cluster_loss # not good;


        if self.rec_loss_aug_w != 0.0:
            # 4. recover from the masked graph
            # self.rec_loss_aug = bpr_loss(self.batch_user_aug_emb, self.batch_pos_item_aug_emb, self.batch_neg_item_aug_emb) # v1
            self.rec_loss_aug = self.rec_loss_aug_w * (self.ssl_layer_loss(self.rec_user_embeddings, self.aug_user_embeddings) + self.ssl_layer_loss(self.rec_item_embeddings, self.aug_item_embeddings))
            # self.rec_loss_aug = self.rec_loss_aug_w * (self.ssl_layer_loss(self.rec_user_embeddings, self.aug_user_embeddings))
            loss += self.rec_loss_aug # good;
        
        if self.inter_ppr_w != 0.0:
            # 5. add interaction ppr loss
            # self.uu_inter_ppr_loss = tf.losses.mean_squared_error(self.uu_inter_ppr_mat, self.user_user_sim) # v1
            self.uu_inter_ppr_loss = self.inter_ppr_w * tf.losses.mean_squared_error(self.uu_inter_ppr_mat, self.user_user_aug_sim) # v2, add regular on the argumented graph
            # self.uu_inter_ppr_loss = self.inter_ppr_w * (tf.losses.mean_squared_error(self.uu_inter_ppr_mat, self.user_user_aug_sim) + 0.5 * tf.losses.mean_squared_error(self.ii_inter_ppr_mat, self.item_item_aug_sim))
            # self.ii_inter_ppr_loss = tf.losses.mean_squared_error(self.ii_inter_ppr_mat, self.item_item_sim)
            loss += self.uu_inter_ppr_loss # 

        if self.graph_label_w != 0.0:
            logit_user = tf.layers.dense(inputs=tf.reshape(self.batch_user_emb, [-1, self.emb_size]), units=self.graph_label_num, activation=None)
            self.graph_label_loss = self.graph_label_w * tf.losses.sparse_softmax_cross_entropy(self.graph_batch_label, logit_user)
            # self.graph_label_loss = self.graph_label_w * tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_user, labels=self.graph_batch_label)
            loss += self.graph_label_loss
        # optimizer setting
        # loss += self.inter_ppr_w * (self.uu_inter_ppr_loss + self.ii_inter_ppr_loss)   

        if self.interact_ppr_w != 0.0:
            # add interaction intimacy scores
            self.interact_ppr_loss = tf.losses.mean_squared_error(self.interact_ppr_mat, self.edgo_edgo_sim) #0.08403892
            # self.rec_ppr_aug_loss = self.ssl_layer_loss(self.all_embeddings, self.edgo_aug_v2_emb) #4681.483
            loss += self.interact_ppr_w * self.interact_ppr_loss
            # loss += self.rec_ppr_aug_w * self.rec_ppr_aug_loss
        else:
            self.interact_ppr_loss = rec_loss
        
        # pdb.set_trace()
        v1_opt = tf.train.AdamOptimizer(self.lRate)
        v1_op = v1_opt.minimize(rec_loss)
        v2_opt = tf.train.AdamOptimizer(self.lRate)
        v2_op = v2_opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            #joint learning
            if epoch > self.maxEpoch / 3: #need ?
            # if epoch > -1: #need ?
            # if epoch > -1:
                #pdb.set_trace()
                sub_mat = {}
                sub_mat['adj_indices_sub'], sub_mat['adj_values_sub'], sub_mat[
                    'adj_shape_sub'] = self._convert_csr_to_sparse_tensor_inputs(
                    self.get_adj_mat(is_subgraph=True))
                # sub_mat['adj_indices_sub'], sub_mat['adj_values_sub'], sub_mat[
                #     'adj_shape_sub'] = self._convert_csr_to_sparse_tensor_inputs(
                #     self.get_adj_mat(is_subgraph=False))
                 
                for n, batch in enumerate(self.next_batch_pairwise()):
                    user_idx, i_idx, j_idx = batch
                    feed_dict = {self.u_idx: user_idx,
                                 self.v_idx: i_idx,
                                 self.neg_idx: j_idx}
                    feed_dict.update({
                        self.sub_mat['adj_values_sub']: sub_mat['adj_values_sub'],
                        self.sub_mat['adj_indices_sub']: sub_mat['adj_indices_sub'],
                        self.sub_mat['adj_shape_sub']: sub_mat['adj_shape_sub'],
                    })
                    # _, opt_l, l1, l2, l3 = self.sess.run([v2_op, loss, rec_loss, self.neighbor_dis_loss, self.social_ppr_loss],feed_dict=feed_dict)
                    # _, opt_l, l1, l2 = self.sess.run([v2_op, loss, rec_loss, self.graph_label_loss],feed_dict=feed_dict)
                    _, opt_l, l1 = self.sess.run([v2_op, loss, rec_loss],feed_dict=feed_dict)
                    # pdb.set_trace()
                    # print(self.foldInfo, 'training:', epoch + 1, 'batch', n, 'total loss:', opt_l, 'rec loss:', l1, 'graph_label_loss:', l2)
                    print(self.foldInfo, 'training:', epoch + 1, 'batch', n, 'total loss:', opt_l, 'rec loss:', l1)
            else:
                #initialization with only recommendation task
                for n, batch in enumerate(self.next_batch_pairwise()):
                    user_idx, i_idx, j_idx = batch
                    feed_dict = {self.u_idx: user_idx,
                                 self.v_idx: i_idx,
                                 self.neg_idx: j_idx}
                    # _, l1, user_embeddings = self.sess.run([v1_op, rec_loss, self.rec_user_embeddings],
                    #                       feed_dict=feed_dict)
                    _, l1, reg_l = self.sess.run([v1_op, rec_loss, reg_loss],
                                          feed_dict=feed_dict)
                    print(self.foldInfo, 'training:', epoch + 1, 'batch', n, 'rec loss:', l1)
            self.U, self.V = self.sess.run([self.rec_user_embeddings, self.rec_item_embeddings])
            if epoch % 10 == 0 or epoch == (self.maxEpoch - 1):
                self.ranking_performance(epoch) #model performance;
        self.U,self.V = self.bestU,self.bestV

        # save best user and item embs, user and item dict,  id2user, id2item
        # self.data.id2user, self.data.id2item: dict
        # self.U, self.V: numpy
        # save
        # exp = 'add_inter_social_rep'
        # exp = 'mean_inter_social_rep'
        # exp = 'mlp_inter_social_rep'
        # np.save('./exp/lastfm/{}/user_emb'.format(exp), self.U)
        # np.save('./exp/lastfm/{}/item_emb'.format(exp), self.V)
        # with open('./exp/lastfm/{}/id2user.pickle'.format(exp), 'wb') as handle:
        #     pickle.dump(self.data.id2user, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('./exp/lastfm/{}/id2item.pickle'.format(exp), 'wb') as handle:
        #     pickle.dump(self.data.id2item, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.rec_user_embeddings, self.rec_item_embeddings])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items
