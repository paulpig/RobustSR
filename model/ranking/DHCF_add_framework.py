#coding:utf8
from base.deepRecommender import DeepRecommender
from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
from scipy.sparse import coo_matrix,hstack
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from util import config

#The original implementation is not opensourced. I reproduced this model by emailing with the author.
#However, I think this model is not effective. There are a lot of problems in the original paper.
#Build 2-hop hyperedge would lead to a very dense adjacency matrix. I think it would result in over-smoothing.
#So, I just use the 1-hop hyperedge.

class DHCF_add_framework(SocialRecommender, GraphRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None, relation=None, fold='[1]'):
        # super(DHCF_add_framework, self).__init__(conf,trainingSet,testSet,fold)
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(DHCF_add_framework, self).readConfiguration()
        args = config.OptionConf(self.config['DHCF_add_framework'])
        self.ppr_rate = float(args['-ppr_rate'])
        self.social_ppr_cluster_w = float(args['-social_ppr_cluster_w'])

    def buildAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return u_i_adj

    
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
        indices = np.argsort(social_ppr_mat, axis=-1)[:,-top_k:] #(user_num, top k)

        user_index = np.arange(self.num_users).reshape(self.num_users, 1)
        user_index = np.tile(user_index, top_k).reshape(1, -1)
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
    
    
    def initModel(self):
        super(DHCF_add_framework, self).initModel()
        #Build adjacency matrix
        A = self.buildAdjacencyMatrix()

        self.bs_matrix = self.get_birectional_social_matrix()
        #Build incidence matrix
        #H_u = hstack([A,A.dot(A.transpose().dot(A))])
        H_u = A
        D_u_v = H_u.sum(axis=1).reshape(1,-1)
        D_u_e = H_u.sum(axis=0).reshape(1,-1)
        temp1 = (H_u.transpose().multiply(np.sqrt(1.0/D_u_v))).transpose()
        temp2 = temp1.transpose()
        A_u = temp1.multiply(1.0/D_u_e).dot(temp2)
        A_u = A_u.tocoo()
        indices = np.mat([A_u.row, A_u.col]).transpose()
        H_u = tf.SparseTensor(indices, A_u.data.astype(np.float32), A_u.shape)
        H_i = A.transpose()
        D_i_v = H_i.sum(axis=1).reshape(1,-1)
        D_i_e = H_i.sum(axis=0).reshape(1,-1)
        temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()
        temp2 = temp1.transpose()
        A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)
        A_i = A_i.tocoo()
        indices = np.mat([A_i.row, A_i.col]).transpose()
        H_i = tf.SparseTensor(indices, A_i.data.astype(np.float32), A_i.shape)

        print('Runing on GPU...')
        #Build network
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        initializer = tf.contrib.layers.xavier_initializer()
        self.n_layer = 2
        self.weights={}
        for i in range(self.n_layer):
            self.weights['layer_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_%d' % (i + 1))

        user_embeddings = self.user_embeddings
        item_embeddings = self.item_embeddings
        all_user_embeddings = [user_embeddings]
        all_item_embeddings = [item_embeddings]

        # message dropout.
        def without_dropout(embedding):
            return embedding

        def dropout(embedding):
            return tf.nn.dropout(embedding, rate=0.1)

        # index = 0
        for i in range(self.n_layer):
            with tf.device("/gpu:{}".format(i*2)):
                new_user_embeddings = tf.sparse_tensor_dense_matmul(H_u,self.user_embeddings)
                user_embeddings = tf.nn.leaky_relu(tf.matmul(new_user_embeddings,self.weights['layer_%d' %(i+1)])+ user_embeddings)
                
                user_embeddings = tf.cond(self.isTraining, lambda: dropout(user_embeddings),
                                        lambda: without_dropout(user_embeddings))
                user_embeddings = tf.math.l2_normalize(user_embeddings,axis=1)

            with tf.device("/gpu:{}".format(i*2 + 1)):
                new_item_embeddings = tf.sparse_tensor_dense_matmul(H_i,self.item_embeddings)
                item_embeddings = tf.nn.leaky_relu(tf.matmul(new_item_embeddings,self.weights['layer_%d' %(i+1)])+ item_embeddings)
                item_embeddings = tf.cond(self.isTraining, lambda: dropout(item_embeddings),
                                        lambda: without_dropout(item_embeddings))
                item_embeddings = tf.math.l2_normalize(item_embeddings,axis=1)
            
            all_item_embeddings.append(item_embeddings)
            all_user_embeddings.append(user_embeddings)

        # user_embeddings = tf.reduce_sum(all_user_embeddings,axis=0)/(1+self.n_layer)
        # item_embeddings = tf.reduce_sum(all_item_embeddings, axis=0) / (1 + self.n_layer)
        
        user_embeddings = tf.reduce_sum(all_user_embeddings,axis=0)
        item_embeddings = tf.reduce_sum(all_item_embeddings, axis=0)
        # user_embeddings = tf.concat(all_user_embeddings,axis=1)
        # item_embeddings = tf.concat(all_item_embeddings, axis=1)

        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding,item_embeddings),1)
        

        self.rec_user_embeddings = user_embeddings
        self.rec_item_embeddings = item_embeddings
        if self.ppr_rate != 0.0:
            # add social-based ppr mat
            social_ppr_mat, social_ppr_sp_mat = self.cal_ppr_social_mat()
            self.social_ppr_mat = tf.convert_to_tensor(social_ppr_mat, dtype=tf.float32) #(userNum, userNum)
            self.user_user_sim = tf.matmul(self.rec_user_embeddings, tf.transpose(self.rec_user_embeddings, perm=[1, 0])) #(userNum, userNum)

        
        if self.social_ppr_cluster_w != 0.0:
            self.social_ppr_cluster_emb = self.sampleTopkUsersKeepOri(self.rec_user_embeddings, top_k=10, social_ppr_mat=social_ppr_mat, mask_ori=True, add_norm=True, add_self=True, social_layer_num=2) #目前的sota.

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

    def trainModel(self):
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        reg_loss = self.regU * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
                                                                    tf.nn.l2_loss(self.neg_item_embedding))
        for i in range(self.n_layer):
            reg_loss+= self.regU*tf.nn.l2_loss(self.weights['layer_%d' %(i+1)])
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + reg_loss

        if self.ppr_rate != 0.0:
            # 1. add social-based ppr loss
            self.social_ppr_loss = self.ppr_rate * tf.losses.mean_squared_error(self.social_ppr_mat, self.user_user_sim)
            loss += self.social_ppr_loss # ppr is good;

        if self.social_ppr_cluster_w != 0.0:
            self.social_ppr_cluster_loss = self.social_ppr_cluster_w * self.ssl_layer_loss(self.rec_user_embeddings, self.social_ppr_cluster_emb)
            loss += self.social_ppr_cluster_loss
        
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss, colocate_gradients_with_ops=True)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.isTraining:1})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.sess.run(self.test,feed_dict={self.u_idx:u,self.isTraining:0})
        else:
            return [self.data.globalMean] * self.num_items