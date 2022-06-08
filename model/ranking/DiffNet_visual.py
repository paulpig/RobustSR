from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix
import numpy as np
import os
from util import config
import pickle
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#For general comparison. We do not include the user/item features extracted from text/images

from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix
import numpy as np
import os
from util import config
import scipy.sparse as sp
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#For general comparison. We do not include the user/item features extracted from text/images

class DiffNet_visual(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(DiffNet_visual, self).readConfiguration()
        args = config.OptionConf(self.config['DiffNet_visual'])
        self.n_layers = int(args['-n_layer']) #the number of layers of the recommendation module (discriminator)

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0/len(self.social.followees[pair[0]])]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix

    def get_birectional_social_matrix(self):
        row_idx = [self.data.user[pair[0]] for pair in self.social.relation]
        col_idx = [self.data.user[pair[1]] for pair in self.social.relation]
        follower_np = np.array(row_idx)
        followee_np = np.array(col_idx)
        relations = np.ones_like(follower_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((relations, (follower_np, followee_np)), shape=(self.num_users, self.num_users))
        # pdb.set_trace()
        adj_mat = tmp_adj.multiply(tmp_adj)
        return adj_mat.tocoo()
    

    def initModel(self):
        super(DiffNet_visual, self).initModel()
        S = self.buildSparseRelationMatrix()
        # S = self.get_birectional_social_matrix()
        
        indices = np.mat([S.row, S.col]).transpose()
        self.S = tf.SparseTensor(indices, S.data.astype(np.float32), S.shape)
        self.A = self.create_sparse_adj_tensor()

    def trainModel(self):
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        for k in range(self.n_layers):
            self.weights['weights%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, self.emb_size]), name='weights%d' % k)

        # user_embeddings = self.user_embeddings
        # # all_social_embeddings = [user_embeddings]
        # for k in range(self.n_layers):
        #     new_user_embeddings = tf.sparse_tensor_dense_matmul(self.S, user_embeddings)
        #     new_user_embeddings = tf.math.l2_normalize(new_user_embeddings, axis=1)
        #     user_embeddings = tf.matmul(tf.concat([new_user_embeddings,user_embeddings],1),self.weights['weights%d' % k])
        #     # user_embeddings = tf.math.l2_normalize(user_embeddings, axis=1)
        #     # all_social_embeddings += [user_embeddings]
        #     # user_embeddings = tf.nn.relu(user_embeddings)
        #     # user_embeddings = tf.math.l2_normalize(user_embeddings,axis=1)

        # # user_embeddings = tf.reduce_sum(all_social_embeddings, axis=0)

        # # final_user_embeddings = user_embeddings + tf.sparse_tensor_dense_matmul(self.A, self.item_embeddings)
        # final_user_embeddings = user_embeddings + tf.math.l2_normalize(tf.sparse_tensor_dense_matmul(self.A, self.item_embeddings))

        user_embeddings = self.user_embeddings
        for k in range(self.n_layers):
            new_user_embeddings = tf.sparse_tensor_dense_matmul(self.S,user_embeddings)
            user_embeddings = tf.matmul(tf.concat([new_user_embeddings,user_embeddings],1),self.weights['weights%d' % k])
            # user_embeddings = tf.nn.relu(user_embeddings)
            user_embeddings = tf.math.l2_normalize(user_embeddings,axis=1)

        # final_user_embeddings = user_embeddings+tf.sparse_tensor_dense_matmul(self.A,self.item_embeddings)
        final_user_embeddings = user_embeddings + tf.math.l2_normalize(tf.sparse_tensor_dense_matmul(self.A, self.item_embeddings))

        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(final_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1)

        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (
                    tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
                    tf.nn.l2_loss(self.neg_item_embedding))
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)
            
            if epoch % 10 == 0 or epoch == (self.maxEpoch - 1):
                self.ranking_performance(epoch) #model performance;
                self.U, self.V = self.sess.run([final_user_embeddings, self.item_embeddings],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
        # self.U,self.V = self.bestU,self.bestV

        model_name = "diffNet_v5_true"
        np.save('./exp/lastfm/{}/user_emb'.format(model_name), self.U)
        np.save('./exp/lastfm/{}/item_emb'.format(model_name), self.V)
        with open('./exp/lastfm/{}/id2user.pickle'.format(model_name), 'wb') as handle:
            pickle.dump(self.data.id2user, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./exp/lastfm/{}/id2item.pickle'.format(model_name), 'wb') as handle:
            pickle.dump(self.data.id2item, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("path:  " + './exp/lastfm/{}/user_emb'.format(model_name))
    # def saveModel(self):
    #     self.bestU, self.bestV = self.sess.run([self.u_embedding, self.v_embedding])

    
    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.sess.run(self.test,feed_dict={self.u_idx:u})
        else:
            return [self.data.globalMean] * self.num_items
    # def predictForRanking(self, u):
    #     'invoked to rank all the items for the user'
    #     if self.data.containsUser(u):
    #         u = self.data.getUserId(u)
    #         # return self.sess.run(self.test,feed_dict={self.u_idx:u})
    #         return self.V.dot(self.U[u])
    #     else:
    #         return [self.data.globalMean] * self.num_items