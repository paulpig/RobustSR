import numpy as np
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from re import compile,findall,split
import pdb
import matplotlib.pyplot as plt

#model_name = 'only_ppr'
#model_name = '0_1_score' # wrong
#model_name = 'LightGCN' # 1.5175
#model_name = 'only_output_subgraph' # 1.8492
#model_name = 'add_inter_social_rep' # 2.9293
#model_name = 'only_cl' # 1.34416, 1.3418
#model_name = 'diffNet' # 3.3931
model_name = 'diffNet_v2' # 1.3980
#model_name = 'MHCN' # 1.3485, 1.3600
#model_name = 'SEPT' # 1.4492, 1.4350

user_emb = np.load('./{}/user_emb.npy'.format(model_name))
item_emb = np.load('./{}/item_emb.npy'.format(model_name))

print(model_name)

with open('./{}/id2user.pickle'.format(model_name), 'rb') as handle:
    id2user = pickle.load(handle)

with open('./{}/id2item.pickle'.format(model_name), 'rb') as handle:
    id2item = pickle.load(handle)

user2id = dict([(value, key) for key, value in id2user.items()])
item2id = dict([(value, key) for key, value in id2item.items()])

#pdb.set_trace()
#print(user2id)
#print(user_emb[:2])
#print(id2user)

# read dataset
dataset_name = 'lastfm'
#dataset_path = '/pub/data/kyyx/wbc/QRec/dataset/{}/'.format(dataset_name)
#dataset_path = '/pub/data/kyyx/wbc/QRec/dataset/{}/'.format(dataset_name)
dataset_path = './'

rating = dataset_path + 'ratings.txt'
social = dataset_path + 'trusts.txt'

# 随机选择一个用户, 检索用户的社交邻居和交互的items;
delim = ' |,|\t'

with open(rating) as f:
    ratings = f.readlines()

user2items = {}
for lineno, line in enumerate(ratings):
    order = split(delim,line.strip())
    if order[0] not in user2id or order[1] not in item2id:
        continue
    userid = user2id[order[0]]
    itemid = item2id[order[1]]
    if userid not in user2items:
        user2items[userid] = [itemid]
    else:
        user2items[userid].append(itemid)

with open(social) as f:
    social_raw = f.readlines()

user2users = {}
for lineno, line in enumerate(social_raw):
    order = split(delim,line.strip())
    if order[0] not in user2id or order[1] not in user2id:
        continue
    userid = user2id[order[0]]
    userid2 = user2id[order[1]]
    if userid not in user2users:
        user2users[userid] = [userid2]
    else:
        user2users[userid].append(userid2)

def calculate_distance(user_id_str):
# 随机选一个用户
#print(user2users)
    #user_id = 190
    user_id = user2id[user_id_str]
    #print(user2users[user_id])
    #print(user2items[user_id])

    center_user_emb = user_emb[user_id]
    neighbor_users = user_emb[user2users[user_id]]
    neighbor_items = item_emb[user2items[user_id]]
    neighbor_users_ids = user2users[user_id]

    tmp_array = neighbor_users_ids.copy()
# only 2-hop
#2_hop_neighbor_users = []
    hop2_neighbor_users = []
    hop3_neighbor_users = []
    for index, user_tmp_id in enumerate(tmp_array):
        #if len(user2users[user_tmp_id]) > 50:
        #    continue
        #print(tmp_array)
        #neighbor_users_ids.extend(user2users[user_tmp_id])
        hop2_neighbor_users.extend(user2users[user_tmp_id]) # 2-hop
        tmp_3_array = user2users[user_tmp_id].copy()
        for index, user_tmp_id_3 in enumerate(tmp_3_array):
            hop3_neighbor_users.extend(user2users[user_tmp_id_3]) # 3-hop

#neighbor_users = user_emb[neighbor_users_ids]
    hop2_neighbor_users_emb = user_emb[hop2_neighbor_users]
    hop3_neighbor_users_emb = user_emb[hop3_neighbor_users]

# distance
    centor_user_dis = np.mean(np.reshape(center_user_emb, (1, -1)) - neighbor_users, axis=0)
    centor_item_dis = np.mean(np.reshape(center_user_emb, (1, -1)) - neighbor_items, axis=0)

# l2 distance
    distance_user_item = np.linalg.norm(centor_user_dis-centor_item_dis)
    #print("distance: ", distance_user_item)
    return distance_user_item

sim_total = []
for index in range(len(user2id)):
    index_data = "{}".format(index)
    if index_data not in user2id:
        continue

    #if len(user2users[user2id[index_data]]) < 2 or len(user2users[user2id[index_data]]) > 10:
    #    continue
    #if index > 1000:
    #    break
    sim = calculate_distance(index_data)
    sim_total.append(sim)

print("final: ", np.mean(sim_total))
print("length: ", len(sim_total))
#pdb.set_trace()

exit(0)
total_array = np.concatenate([neighbor_users, neighbor_items, np.reshape(center_user_emb, (1, -1))])

#total_array = np.concatenate([hop3_neighbor_users_emb, neighbor_users, neighbor_items, np.reshape(center_user_emb, (1, -1))])
#total_array = np.append(total_array, center_user_emb)
#pdb.set_trace()
#tsne = TSNE(n_components=2)
#results = tsne.fit_transform(total_array)
#print(results)
#label = [0]* len(hop3_neighbor_users_emb) + [1]*len(neighbor_users) + [2] * len(neighbor_items) + [3]
label = [1]*len(neighbor_users) + [2] * len(neighbor_items) + [3]

pca = PCA(n_components=2)
results = pca.fit_transform(total_array)

print("neighbor_user: {}, neighbor_item: {}".format(len(neighbor_users), len(neighbor_items)))
#print(results)
plt.title(model_name)

plt.scatter(results[:, 0], results[:, 1], c=label)
#plt.show()

plt.savefig('{}_distance.png'.format(model_name))
