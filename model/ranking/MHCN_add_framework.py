from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix
import numpy as np
from util.loss import bpr_loss
import os
from util import config
from math import sqrt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pickle
import scipy.sparse as sp
import pdb

# Recommended Maximum Epoch Setting: LastFM 120 Douban 30 Yelp 30
# A slight performance drop is observed when we transplanted the model from python2 to python3. The cause is unclear.

class MHCN_add_framework(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(MHCN_add_framework, self).readConfiguration()
        args = config.OptionConf(self.config['MHCN_add_framework'])
        self.n_layers = int(args['-n_layer'])
        self.ss_rate = float(args['-ss_rate'])
        self.ppr_rate = float(args['-ppr_rate'])
        self.social_ppr_cluster_w = float(args['-social_ppr_cluster_w'])
        self.top_k = int(args['-topk'])
        self.cluster_type = int(args['-cluster_type']) 

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return ratingMatrix

    def buildJointAdjacency(self):
        indices = [[self.data.user[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
        values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
                  for item in self.data.trainingData]
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_users, self.num_items])
        return norm_adj

    def buildMotifInducedAdjacencyMatrix(self):
        S = self.buildSparseRelationMatrix()
        Y = self.buildSparseRatingMatrix()
        self.userAdjacency = Y.tocsr()
        self.itemAdjacency = Y.T.tocsr()
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9+A9.T
        A10  = Y.dot(Y.T)-A8-A9
        #addition and row-normalization
        H_s = sum([A1,A2,A3,A4,A5,A6,A7])
        H_s = H_s.multiply(1.0/H_s.sum(axis=1).reshape(-1, 1))
        H_j = sum([A8,A9])
        H_j = H_j.multiply(1.0/H_j.sum(axis=1).reshape(-1, 1))
        H_p = A10
        H_p = H_p.multiply(H_p>1)
        H_p = H_p.multiply(1.0/H_p.sum(axis=1).reshape(-1, 1))

        return [H_s,H_j,H_p]

    def adj_to_sparse_tensor(self,adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj
    
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

    def _split_A_hat(self, X):
        A_fold_hat = []
        self.n_fold = 2000
        fold_len = (self.num_users) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.num_users
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(tf.convert_to_tensor(X[start:end, :], dtype=tf.float32))
            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            # print("start: {}, end: {}".format(start, end))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def sampleTopkUsersKeepOri(self, userEmbedding, top_k=10, social_ppr_mat=None, mask_ori=False, add_norm=True, add_self=True, social_layer_num=1):
        """
        self.bs_matrix: csr mat
        social_ppr_mat: array
        return 
            (user_num, dim)
        """
        
        bs_matrix = self.bs_matrix.toarray()
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
        return subgraph_emb

    
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

    def initModel(self):
        super(MHCN_add_framework, self).initModel()
        M_matrices = self.buildMotifInducedAdjacencyMatrix()
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        self.n_channel = 4
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        #define learnable paramters
        for i in range(self.n_channel):
            self.weights['gating%d' % (i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='g_W_%d_1' % (i + 1))
            self.weights['gating_bias%d' %(i+1)] = tf.Variable(initializer([1, self.emb_size]), name='g_W_b_%d_1' % (i + 1))
            self.weights['sgating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='sg_W_%d_1' % (i + 1))
            self.weights['sgating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]), name='sg_W_b_%d_1' % (i + 1))
        self.weights['attention'] = tf.Variable(initializer([1, self.emb_size]), name='at')
        self.weights['attention_mat'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm')
        #define inline functions
        def self_gating(em,channel):
            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em,self.weights['gating%d' % channel])+self.weights['gating_bias%d' %channel]))
        def self_supervised_gating(em, channel):
            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em, self.weights['sgating%d' % channel])+self.weights['sgating_bias%d' % channel]))
        def channel_attention(*channel_embeddings):
            weights = []
            for embedding in channel_embeddings:
                weights.append(tf.reduce_sum(tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])),1))
            score = tf.nn.softmax(tf.transpose(weights))
            mixed_embeddings = 0
            for i in range(len(weights)):
                mixed_embeddings += tf.transpose(tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i])))
            return mixed_embeddings,score
        #initialize adjacency matrices
        H_s = M_matrices[0]
        H_s = self.adj_to_sparse_tensor(H_s)
        H_j = M_matrices[1]
        H_j = self.adj_to_sparse_tensor(H_j)
        H_p = M_matrices[2]
        H_p = self.adj_to_sparse_tensor(H_p)
        R = self.buildJointAdjacency()
        #self-gating
        user_embeddings_c1 = self_gating(self.user_embeddings,1)
        user_embeddings_c2 = self_gating(self.user_embeddings, 2)
        user_embeddings_c3 = self_gating(self.user_embeddings, 3)
        simple_user_embeddings = self_gating(self.user_embeddings,4)
        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        item_embeddings = self.item_embeddings
        all_embeddings_i = [item_embeddings]

        self.ss_loss = 0
        #multi-channel convolution
        for k in range(self.n_layers):
            mixed_embedding = channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)[0] + simple_user_embeddings / 2
            #Channel S
            user_embeddings_c1 = tf.sparse_tensor_dense_matmul(H_s,user_embeddings_c1)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c1, axis=1)
            all_embeddings_c1 += [norm_embeddings]
            #Channel J
            user_embeddings_c2 = tf.sparse_tensor_dense_matmul(H_j, user_embeddings_c2)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c2, axis=1)
            all_embeddings_c2 += [norm_embeddings]
            #Channel P
            user_embeddings_c3 = tf.sparse_tensor_dense_matmul(H_p, user_embeddings_c3)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c3, axis=1)
            all_embeddings_c3 += [norm_embeddings]
            # item convolution
            new_item_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(R), mixed_embedding)
            norm_embeddings = tf.math.l2_normalize(new_item_embeddings, axis=1)
            all_embeddings_i += [norm_embeddings]
            simple_user_embeddings = tf.sparse_tensor_dense_matmul(R, item_embeddings)
            all_embeddings_simple += [tf.math.l2_normalize(simple_user_embeddings, axis=1)]
            item_embeddings = new_item_embeddings
        #averaging the channel-specific embeddings
        user_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
        user_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
        user_embeddings_c3 = tf.reduce_sum(all_embeddings_c3, axis=0)
        simple_user_embeddings = tf.reduce_sum(all_embeddings_simple, axis=0)
        item_embeddings = tf.reduce_sum(all_embeddings_i, axis=0)
        #aggregating channel-specific embeddings
        self.final_item_embeddings = item_embeddings
        self.final_user_embeddings,self.attention_score = channel_attention(user_embeddings_c1,user_embeddings_c2,user_embeddings_c3)
        self.final_user_embeddings += simple_user_embeddings/2
        #create self-supervised loss
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,1), H_s)
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,2), H_j)
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings,3), H_p)
        #embedding look-up
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)

        # add our framework
        self.rec_user_embeddings = self_supervised_gating(self.final_user_embeddings, 4)
        self.bs_matrix = self.get_birectional_social_matrix()
        # add framework
        if self.ppr_rate != 0.0:
            # add social-based ppr mat
            social_ppr_mat, social_ppr_sp_mat = self.cal_ppr_social_mat()

            # social_ppr_mat, social_ppr_sp_mat = self.cal_ppr_social_mat(type='hk', weight=0.7)
            # social_ppr_mat, social_ppr_sp_mat = self.cal_ppr_social_mat(type='origin')
            # social_mat = self._convert_sp_mat_to_sp_tensor(social_ppr_mat)
            # social_mat = tf.sparse.to_dense(social_mat)
            # self.social_ppr_mat = tf.convert_to_tensor(social_ppr_mat, dtype=tf.float32) #(userNum, userNum), OOM, wrong;
            # self.user_user_sim = tf.matmul(self.rec_user_embeddings, tf.transpose(self.rec_user_embeddings, perm=[1, 0])) #(userNum, userNum)
            self.social_ppr_sub_tensor_list = self._split_A_hat(social_ppr_mat)
            rec_user_embeddings_split = self._split_A_hat(self.rec_user_embeddings)
            self.rec_user_user_sim_list = []
            for sub_rec_user_embs in rec_user_embeddings_split:
                self.rec_user_user_sim_list.append(tf.matmul(sub_rec_user_embs, tf.transpose(self.rec_user_embeddings, perm=[1, 0])))
        
        if self.social_ppr_cluster_w != 0.0:
            if self.cluster_type == 6:
                self.social_ppr_cluster_emb = self.sampleTopkUsersKeepOri(self.rec_user_embeddings, top_k=self.top_k, social_ppr_mat=social_ppr_mat, mask_ori=True, add_norm=True, add_self=True, social_layer_num=2) #目前的sota.

    def hierarchical_self_supervision(self,em,adj):
        def row_shuffle(embedding):
            return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))
        def row_column_shuffle(embedding):
            corrupted_embedding = tf.transpose(tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
            corrupted_embedding = tf.gather(corrupted_embedding, tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
            return corrupted_embedding
        def score(x1,x2):
            return tf.reduce_sum(tf.multiply(x1,x2),1)
        user_embeddings = em
        # user_embeddings = tf.math.l2_normalize(em,1) #For Douban, normalization is needed.
        edge_embeddings = tf.sparse_tensor_dense_matmul(adj,user_embeddings)
        #Local MIM
        pos = score(user_embeddings,edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings),edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings),user_embeddings)
        local_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos-neg1))-tf.log(tf.sigmoid(neg1-neg2)))
        #Global MIM
        graph = tf.reduce_mean(edge_embeddings,0)
        pos = score(edge_embeddings,graph)
        neg1 = score(row_column_shuffle(edge_embeddings),graph)
        global_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos-neg1)))
        return global_loss+local_loss

    def trainModel(self):
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        reg_loss = 0
        for key in self.weights:
            reg_loss += 0.001*tf.nn.l2_loss(self.weights[key])
        reg_loss += self.regU * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        total_loss = rec_loss+reg_loss + self.ss_rate*self.ss_loss

        # add framework
        if self.ppr_rate != 0.0:
            # 1. add social-based ppr loss
            # self.social_ppr_loss = self.ppr_rate * tf.losses.mean_squared_error(self.social_ppr_mat, self.user_user_sim)
            loss_list = []
            for ppr_score, user_sim in zip(self.social_ppr_sub_tensor_list, self.rec_user_user_sim_list):
                loss_list.append(tf.reduce_sum(tf.losses.mean_squared_error(ppr_score, user_sim, reduction=tf.losses.Reduction.NONE), axis=-1))
            # pdb.set_trace()
            self.social_ppr_loss = self.ppr_rate * tf.reduce_sum(tf.concat(loss_list, axis=0))/(self.num_users * self.num_users)
            # self.social_ppr_loss = self.ppr_rate * tf.losses.mean_squared_error(self.social_ppr_mat, self.user_user_sim)
            total_loss += self.social_ppr_loss # ppr is good;

        if self.social_ppr_cluster_w != 0.0:
            # 3. add top k user embedding.
            if self.cluster_type == 6:
                self.social_ppr_cluster_loss = self.social_ppr_cluster_w * self.ssl_layer_loss(self.rec_user_embeddings, self.social_ppr_cluster_emb)
            
            total_loss += self.social_ppr_cluster_loss # not good;


        opt = tf.train.AdamOptimizer(self.lRate)
        train_op = opt.minimize(total_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # Suggested Maximum epoch Setting: LastFM 120 Douban 30 Yelp 30
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l1, l2, l3, l4 = self.sess.run([train_op, rec_loss, self.ss_loss, self.social_ppr_loss, self.social_ppr_cluster_loss],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'rec loss:', l1)#,'ss_loss',l2
                # pdb.set_trace()
            self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])
            if epoch % 1 == 0 or epoch == (self.maxEpoch - 1):
                self.ranking_performance(epoch)
    #self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
        self.U,self.V = self.bestU,self.bestV

        # model_name = "MHCN"
        # np.save('./exp/lastfm/{}/user_emb'.format(model_name), self.U)
        # np.save('./exp/lastfm/{}/item_emb'.format(model_name), self.V)
        # with open('./exp/lastfm/{}/id2user.pickle'.format(model_name), 'wb') as handle:
        #     pickle.dump(self.data.id2user, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('./exp/lastfm/{}/id2item.pickle'.format(model_name), 'wb') as handle:
        #     pickle.dump(self.data.id2item, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # print("path:  " + './exp/lastfm/{}/user_emb'.format(model_name))

    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items