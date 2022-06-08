import numpy as np
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from re import compile,findall,split
import pdb
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

plt.rcParams['figure.figsize'] = (7,6)

#model_name = 'only_ppr'
#model_name = '0_1_score' # wrong
#model_name = 'LightGCN'
#model_name = 'only_output_subgraph'
#model_name = 'add_inter_social_rep'
#model_name = 'mean_inter_social_rep'
#model_name = 'add_inter_social_rep'
#model_name = 'mlp_inter_social_rep'
model_name = 'only_cl'
#model_name = 'danser_v2'
#model_name = 'diffNet'
#model_name = 'diffNet_v2'
#model_name = 'only_output_subgraph' # 1.8492
#model_name = 'MHCN'
#model_name = 'ESRF'
#model_name = 'SEPT'
#model_name = 'SCIL_v3'
#model_name = 'SCIL_v1'
#model_name = 'SCIL_v2'
#model_name = 'SCIL'
#model_name = 'SEPT_v1'
case_id = [505, 380]
#case_id = [508, 470]

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
dataset_path = './lastfm/'

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

# 随机选一个用户
#print(user2users)
user_id = 190
#user_id_str = '800'
#user_id = user2id[user_id_str]
#user_id = 25
print(user2users[user_id])
print(user2items[user_id])

def calculate_emb(user_id):
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


#pdb.set_trace()
    total_array = np.concatenate([neighbor_users, neighbor_items, np.reshape(center_user_emb, (1, -1))])
#total_array = np.concatenate([hop3_neighbor_users_emb, neighbor_users, neighbor_items, np.reshape(center_user_emb, (1, -1))])
#total_array = np.append(total_array, center_user_emb)
#pdb.set_trace()
#tsne = TSNE(n_components=2)
#results = tsne.fit_transform(total_array)
#print(results)
#label = [0]* len(hop3_neighbor_users_emb) + [1]*len(neighbor_users) + [2] * len(neighbor_items) + [3]
    #label = [1]*len(neighbor_users) + [2] * len(neighbor_items) + [3]
    #marker = ['^'] * len(neighbor_users) + ['o'] * len(neighbor_items) + ['*']
    return total_array, neighbor_users, neighbor_items
# pca visual
#pca = PCA(n_components=2)
#results = pca.fit_transform(total_array)

#total_array, neighbor_users, neighbor_items = calculate_emb(190)
total_array, neighbor_users, neighbor_items = calculate_emb(case_id[0])
#label = ['#ff7f00']*len(neighbor_users) + ['#7570b3'] * len(neighbor_items) + ['#e41a1c']
#label = ['#292C6D']*len(neighbor_users) + ['#FAEDF0'] * len(neighbor_items) + ['#e41a1c']
#label = ['#C000C0']*len(neighbor_users) + ['#1B00FF'] * len(neighbor_items) + ['#40C2C2']
label = ['#707b7c']*len(neighbor_users) + ['#884e9f'] * len(neighbor_items) + ['#d68910']
marker = ['^'] * len(neighbor_users) + ['o'] * len(neighbor_items) + ['*']
marker = ['^'] * len(neighbor_users) + ['o'] * len(neighbor_items) + ['*']

perplexity = 50
n_components = 2
tsne = manifold.TSNE(
    n_components=n_components,
    init="random",
    random_state=0,
    perplexity=perplexity,
    learning_rate="auto",
    n_iter=1000,
)
#results_v2 = tsne.fit_transform(item_emb)
results = tsne.fit_transform(total_array)

legends= []
print("neighbor_user: {}, neighbor_item: {}".format(len(neighbor_users), len(neighbor_items)))
#print(results)
#plt.title(model_name)
for index in range(len(label)):
    if index+1 == len(label):
        plt_obj = plt.scatter(results[index, 0], results[index, 1], c=label[index], marker=marker[index], s=300)
        legends.append(plt_obj)
    else:
        plt.scatter(results[index, 0], results[index, 1], c=label[index], marker=marker[index], s=100)

#total_array, neighbor_users, neighbor_items = calculate_emb(210)
#total_array, neighbor_users, neighbor_items = calculate_emb(280)
#total_array, neighbor_users, neighbor_items = calculate_emb(340)
#total_array, neighbor_users, neighbor_items = calculate_emb(120)
total_array, neighbor_users, neighbor_items = calculate_emb(case_id[1])
#label = ['#A0522D']*len(neighbor_users) + ['#4daf4a'] * len(neighbor_items) + ['#00FFFF']
#label = ['#1f618d']*len(neighbor_users) + ['#127a65'] * len(neighbor_items) + ['#a04000']
label = ['#2571a3']*len(neighbor_users) + ['#239954'] * len(neighbor_items) + ['#ba4a01']
marker = ['^'] * len(neighbor_users) + ['o'] * len(neighbor_items) + ['*']

results = tsne.fit_transform(total_array)
for index in range(len(label)):
    if index+1 == len(label):
        plt_obj = plt.scatter(results[index, 0], results[index, 1], c=label[index], marker=marker[index], s=300)
        legends.append(plt_obj)
    else:
        plt.scatter(results[index, 0], results[index, 1], c=label[index], marker=marker[index], s=99)
#for index in range(len(label)):
#    plt.scatter(results[index, 0], results[index, 1], c=label[index], marker=marker[index])
#plt.scatter(results[:, 0], results[:, 1], c=label, marker=marker)
print("neighbor_user: {}, neighbor_item: {}".format(len(neighbor_users), len(neighbor_items)))
#plt.vlines(1.0, -3, 1, linestyles='dashed', colors='red') #竖线
plt.xticks([])
plt.yticks([])
#plt.figure(figsize=(8, 8))
#plt.figure(figsize=())
plt.legend(handles=legends, labels=[case_id[0], case_id[1]], labelspacing=2.0, loc='upper left')

plt.show()

#plt.savefig('{}.pdf'.format(model_name), bbox_inches='tight')

exit(0)

# user embedding visuation

pca_v2 = PCA(n_components=2)
results_v2 = pca_v2.fit_transform(user_emb)
#perplexity = 200
#n_components = 2
#tsne = manifold.TSNE(
#    n_components=n_components,
#    init="random",
#    random_state=0,
#    perplexity=perplexity,
#    learning_rate="auto",
#    n_iter=300,
#)
#results_v2 = tsne.fit_transform(item_emb)
#results_v2 = tsne.fit_transform(user_emb)

results_v2 = results_v2/np.reshape(np.sqrt(np.sum(np.power(results_v2, 2), -1)),(-1, 1))

def kde1(x, y, ax):
    from scipy.stats import gaussian_kde

    # Calculate the point density
    xy = np.vstack([x,y])
    kernel = gaussian_kde(xy, bw_method='silverman')

    xmin = x.min()-0.5
    xmax = x.max()+0.5
    ymin = y.min()-0.5
    ymax = y.max()+0.5

    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = np.reshape(kernel(positions).T, X.shape)
    print(Z.shape)
    ax.imshow(np.rot90(Z), cmap=plt.cm.viridis,
              extent=[xmin, xmax, ymin, ymax])
    # ax.imshow(np.rot90(Z), cmap='Blues',
    #           extent=[xmin, xmax, ymin, ymax])

# fig, axarr = plt.subplots(1, 1)
kde1(results_v2[:, 0], results_v2[:, 1], plt)

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


plt.xlim((-1.2, 1.2))
plt.ylim((-1.2, 1.2))
size = 20
plt.scatter(np.array(results_v1[:,0])[user_id], np.array(results_v2[:,1])[user_id], c='red', s =size, marker='^') # 12
#index_hop2 = [0, 1, 4] # best
#index_hop2 = [0, 4] # best
index_hop2 = range(0, len(neighbor_users_ids))
plt.scatter(np.array(results_v2[:,0])[np.array(neighbor_users_ids)[index_hop2]], np.array(results_v2[:,1])[np.array(neighbor_users_ids)[index_hop2]], c='blue', s=size) # 12h
index = range(80, 100)
index = range( 90, 120)
index_final = []
for i in index:
    if results_v2[:,0][i] - results_v2[:,0][user_id] <= 1e-2 and results_v2[:,1][i] - results_v2[:,1][user_id] <= 1e-2:
        continue
    else:
        index_final.append(i)
index = index_final
# index = range(140, 160)
plt.scatter(np.array(results_v2[:,0])[index], np.array(results_v2[:,1])[index], c='black', s=size, marker='*') # 48
#axarr[0].scatter(np.array(results_v2[:,0])[hop3_neighbor_users], np.array(results_v2[:,1])[hop3_neighbor_users], c='red') # 12
plt.tight_layout()
plt.savefig('only_cl.pdf')
plt.show()
