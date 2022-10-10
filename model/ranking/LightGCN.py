from base.graphRecommender import GraphRecommender
import tensorflow as tf
from util.loss import bpr_loss
from util.config import OptionConf
import pickle
import numpy as np
import pdb
"""
LightGCN model
"""
class LightGCN(GraphRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(LightGCN, self).__init__(conf,trainingSet,testSet,fold)
        args = OptionConf(self.config['LightGCN'])
        self.n_layers = int(args['-n_layer'])
        # pdb.set_trace()

    def initModel(self):
        super(LightGCN, self).initModel()
        #SEPT对齐, 修正初始化.
        # self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.emb_size], stddev=0.005), name='U') / 2
        # self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.emb_size], stddev=0.005), name='V') / 2
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)
        norm_adj = self.create_joint_sparse_adj_tensor()
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj, ego_embeddings)
            #add normalize, 对齐sept.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
            # all_embeddings += [ego_embeddings]
        # all_embeddings = tf.reduce_mean(all_embeddings, axis=0)
        all_embeddings = tf.reduce_sum(all_embeddings, axis=0) #从mean修改为sum...
        # all_embeddings = tf.reduce_mean(all_embeddings, axis=0) #从mean修改为sum...
        self.multi_user_embeddings, self.multi_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.multi_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.multi_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.multi_item_embeddings, self.v_idx)
        # self.test = tf.reduce_sum(tf.multiply(self.batch_user_emb, self.multi_item_embeddings), 1)

    def trainModel(self):
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        rec_loss += self.regU * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb) + tf.nn.l2_loss(
                self.batch_neg_item_emb))
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(rec_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                # user_emb_batch几乎为0.
                # _, l, user_embeddings = self.sess.run([train, rec_loss, self.multi_user_embeddings],
                #                 feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                _, l = self.sess.run([train, rec_loss],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                # pdb.set_trace()
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'loss:', l)
            self.U, self.V = self.sess.run([self.multi_user_embeddings, self.multi_item_embeddings])
            self.ranking_performance(epoch)
        self.U,self.V = self.bestU,self.bestV

        exp = 'LightGCN_v1'
        np.save('./exp/douban_book/{}/user_emb'.format(exp), self.U)
        np.save('./exp/douban_book/{}/item_emb'.format(exp), self.V)
        with open('./exp/douban_book/{}/id2user.pickle'.format(exp), 'wb') as handle:
            pickle.dump(self.data.id2user, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./exp/douban_book/{}/id2item.pickle'.format(exp), 'wb') as handle:
            pickle.dump(self.data.id2item, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.multi_user_embeddings, self.multi_item_embeddings])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items