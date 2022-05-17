import numpy as np
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from re import compile,findall,split
import pdb
import matplotlib.pyplot as plt
import matplotlib

model_name = 'only_ppr'
#model_name = '0_1_score' # wrong
#model_name = 'LightGCN'
#model_name = 'only_output_subgraph'
#model_name = 'add_inter_social_rep'
#model_name = 'only_cl'
#model_name = 'diffNet'
#model_name = 'MHCN'
#model_name = 'sept'
user_emb = np.load('./{}/user_emb.npy'.format(model_name))
item_emb = np.load('./{}/item_emb.npy'.format(model_name))

print(model_name)

# 统一一下字典;
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


def calcualte_weight(user_id_str):
    # 随机选一个用户
    #print(user2users)
    #user_id = 190
    #user_id = 220
    #user_id = 240
    #user_id = 600
    #user_id_str = '300'
    user_id = user2id[user_id_str]
    #print(user2users[user_id])
    #print(user2items[user_id])

    center_user_emb = user_emb[user_id]
    neighbor_users = user_emb[user2users[user_id]] # 1-hop
    neighbor_items = item_emb[user2items[user_id]]
    hop1_neighbor_users = user2users[user_id]

    neighbor_users_ids = user2users[user_id]

    # using neighbor items as the centor user
    center_user_emb = np.mean(neighbor_items, axis=0)

# add random users
    neg_user_ids = []
    neg_num = 100
    while neg_num >0:
        user_id = np.random.randint(len(user2id), size=1)[0]
        #print(user_id)
        #print('test',neighbor_users_ids)
        while user_id in neighbor_users_ids and user_id in neg_user_ids:
            user_id = np.random.randint(len(user2id), size=1)[0]
            #print(user_id, neighbor_users_ids)
        neg_user_ids.append(user_id)
        neg_num -= 1

    neg_users_emb = user_emb[neg_user_ids]
#neg_sim_users = np.sum(np.reshape(center_user_emb, (1,-1)) * neg_users_emb, axis=-1)
    neg_sim_users = neg_users_emb.dot(center_user_emb)/ (np.linalg.norm(neg_users_emb, axis=1) * np.linalg.norm(center_user_emb))

# add 2-hop neighbors
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

    # 1-hop
    #hop2_neighbor_users_emb = neighbor_users
    #print("neighbors number: ", len(hop2_neighbor_users))
    #if len(hop2_neighbor_users) > 100:
    #    return 0.0
    hop3_neighbor_users_emb = user_emb[hop3_neighbor_users]

    # 设置邻居节点的hops
    final_neighbor_users_emb = hop3_neighbor_users_emb
    #final_neighbor_users_emb = hop2_neighbor_users_emb
    #final_neighbor_users_emb = neighbor_users

    #if len(hop3_neighbor_users) < 20 or len(hop3_neighbor_users) > 999:
    #    return 0.0

    print("hop3: ", len(hop3_neighbor_users))
# calculate sim
    #sim_users = final_neighbor_users_emb.dot(center_user_emb)
    #sim_users = np.sum(np.reshape(center_user_emb, (1,-1)) * final_neighbor_users_emb, axis=-1)
    sim_users = final_neighbor_users_emb.dot(center_user_emb)/ (np.linalg.norm(final_neighbor_users_emb, axis=1) * np.linalg.norm(center_user_emb))
    sim_users = np.reshape(sim_users, (1, -1))


#sim_users_total = np.sum(np.reshape(center_user_emb, (1,-1)) * user_emb, axis=-1)
    user_emb_rm_pos = [index for index in range(len(user_emb)) if index not in hop2_neighbor_users and index not in hop1_neighbor_users and index not in hop3_neighbor_users]

    user_emb_rm_pos_emb = user_emb[user_emb_rm_pos]

    #sim_users_total = user_emb.dot(center_user_emb)/ (np.linalg.norm(user_emb, axis=1) * np.linalg.norm(center_user_emb))
    sim_users_total = user_emb_rm_pos_emb.dot(center_user_emb)/ (np.linalg.norm(user_emb_rm_pos_emb, axis=1) * np.linalg.norm(center_user_emb))
    #sim_users_total = user_emb.dot(center_user_emb)
    #sim_users_total = user_emb.dot(center_user_emb)/ (np.linalg.norm(user_emb, axis=1) * np.linalg.norm(center_user_emb))

    #print("mean social sim:", np.mean(sim_users))
    #print("mean social sim subtract:", np.mean(sim_users) - np.mean(neg_sim_users))
    #print("mean social total sim:", np.mean(sim_users_total))
    #print("mean social total subtract total:", np.mean(sim_users) - np.mean(sim_users_total))
    return np.mean(sim_users) - np.mean(sim_users_total)
    #return np.mean(sim_users)

sim_total = []
for index in range(len(user2id)):
    index_data = "{}".format(index)
    if index_data not in user2id:
        continue

    #if len(user2users[user2id[index_data]]) < 2 or len(user2users[user2id[index_data]]) > 10:
    #    continue
    if index > 100:
        break
    sim = calcualte_weight(index_data)
    sim_total.append(sim)

print("final: ", np.mean(sim_total))
print("length: ", len(sim_total))


sim_v1 = calcualte_weight('300')
print("case result: ", sim_v1)
sim_v1 = calcualte_weight('200')
print("case result: ", sim_v1)

sim_v1 = calcualte_weight('100')
print("case result: ", sim_v1)
sim_v1 = calcualte_weight('400')
print("case result: ", sim_v1)
## label
#label_v1 = [1]*len(sim_users)
#vegetables = [1]
#
##print(sim_users)
#fig, ax = plt.subplots()
#im = ax.imshow(sim_users)
#
## Show all ticks and label them with the respective list entries
##ax.set_xticks(np.arange(len(farmers)), labels=farmers)
##ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)
#
## Rotate the tick labels and set their alignment.
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#         rotation_mode="anchor")
#
## Loop over data dimensions and create text annotations.
###for i in range(len(vegetables)):
##    for j in range(len(label_v1)):
##        text = ax.text(j, i, sim_users[i, j],
##                       ha="center", va="center", color="w")
#
#ax.set_title("Harvest of local farmers (in tons/year)")
#fig.tight_layout()
##plt.show()
#
#plt.savefig('{}_hot.png'.format(model_name))
