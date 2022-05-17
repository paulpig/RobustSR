import numpy as np
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from re import compile,findall,split
import pdb
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KernelDensity
import random
import matplotlib as mpl



model_name = 'only_ppr'
#model_name = 'ppr_v2'
#model_name = '0_1_score' # wrong
#model_name = 'only_output_subgraph'
#model_name = 'add_inter_social_rep'
#model_name = 'only_cl'
#model_name = 'diffNet'
#model_name = 'LightGCN'
#model_name = 'diffNet_v3'
#model_name = 'diffNet_v2'
#model_name = 'ESRF'
#model_name = 'MHCN'
#model_name = 'SEPT'
#model_name = 'SCIL_v1'
#model_name = 'SCIL'
#model_name = 'SCIL_v2'

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

# 随机选一个用户
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
    #centor_user_dis = np.mean(np.reshape(center_user_emb, (1, -1)) - neighbor_users, axis=0)
    #centor_item_dis = np.mean(np.reshape(center_user_emb, (1, -1)) - neighbor_items, axis=0)

    # l2 distance, alignment
    #distance_user_item = np.linalg.norm(centor_user_dis-centor_item_dis)
    #print("distance: ", distance_user_item)

    #centor_user_dis =np.power( np.mean(np.linalg.norm(np.reshape(center_user_emb, (1, -1)) - neighbor_users)), 2)
    #return centor_user_dis

    # uniform
    #random_user_id = random.randint(0, len(user_emb))
    #while random_user_id == user_id or random_user_id not in id2user.keys():
    #    random_user_id = random.randint(0, len(user2id))

    #random_user_emb = user_emb[random_user_id]
    #return np.exp(-2*np.power(np.linalg.norm(center_user_emb - random_user_emb), 2))

    # cos sim
    #cos_sim = cosine_similarity(center_user_emb.reshape(1, -1), np.mean(hop2_neighbor_users_emb, axis=0).reshape(1, -1))
    #return cos_sim

    # return 1-hop neighbor emb
    #return np.mean(hop2_neighbor_users_emb, axis=0)
    return np.mean(neighbor_users, axis=0)
    #return np.mean(neighbor_items, axis=0)


# get 1-hop
#sim_total = []
#for index in range(len(user2id)):
#    index_data = "{}".format(index)
#    if index_data not in user2id:
#        continue
#
#    #if len(user2users[user2id[index_data]]) < 2 or len(user2users[user2id[index_data]]) > 10:
#    #    continue
#    #if index > 1000:
#    #    break
#    sim = calculate_distance(index_data)
#    sim_total.append(sim)
#
##print("user item match score: ", np.mean(sim_total))
#print("user item match score: ",np.log( np.mean(sim_total)))
## case study




##print(user2users)
#user_id = 190
#print(user2users[user_id])
#print(user2items[user_id])
#
#center_user_emb = user_emb[user_id]
#neighbor_users = user_emb[user2users[user_id]]
#neighbor_items = item_emb[user2items[user_id]]
#neighbor_users_ids = user2users[user_id]
#
#tmp_array = neighbor_users_ids.copy()
## only 2-hop
##2_hop_neighbor_users = []
#hop2_neighbor_users = []
#hop3_neighbor_users = []
#for index, user_tmp_id in enumerate(tmp_array):
#    #if len(user2users[user_tmp_id]) > 50:
#    #    continue
#    #print(tmp_array)
#    #neighbor_users_ids.extend(user2users[user_tmp_id])
#    hop2_neighbor_users.extend(user2users[user_tmp_id]) # 2-hop
#    tmp_3_array = user2users[user_tmp_id].copy()
#    for index, user_tmp_id_3 in enumerate(tmp_3_array):
#        hop3_neighbor_users.extend(user2users[user_tmp_id_3]) # 3-hop
#
##neighbor_users = user_emb[neighbor_users_ids]
#hop2_neighbor_users_emb = user_emb[hop2_neighbor_users]
#hop3_neighbor_users_emb = user_emb[hop3_neighbor_users]
#
#
##pdb.set_trace()
#total_array = np.concatenate([neighbor_users, neighbor_items, np.reshape(center_user_emb, (1, -1))])
##total_array = np.concatenate([hop3_neighbor_users_emb, neighbor_users, neighbor_items, np.reshape(center_user_emb, (1, -1))])
##total_array = np.append(total_array, center_user_emb)
##pdb.set_trace()
##tsne = TSNE(n_components=2)
##results = tsne.fit_transform(total_array)
##print(results)
##label = [0]* len(hop3_neighbor_users_emb) + [1]*len(neighbor_users) + [2] * len(neighbor_items) + [3]
#label = [1]*len(neighbor_users) + [2] * len(neighbor_items) + [3]

#pca = PCA(n_components=2)
#results = pca.fit_transform(total_array)
#
#perplexity = 50
#n_components = 2
#tsne = manifold.TSNE(
#    n_components=n_components,
#    init="random",
#    random_state=0,
#    perplexity=perplexity,
#    learning_rate="auto",
#    n_iter=300,
#)
##results_v2 = tsne.fit_transform(item_emb)
#results = tsne.fit_transform(total_array)
#
#print("neighbor_user: {}, neighbor_item: {}".format(len(neighbor_users), len(neighbor_items)))
##print(results)
#plt.title(model_name)
#
#plt.scatter(results[:, 0], results[:, 1], c=label)
#plt.show()
#
#plt.savefig('{}.png'.format(model_name))

# show embedding of users and items
from sklearn.preprocessing import normalize
import seaborn as sns
#fig, axs = plt.subplots(2,2)
fig, axs = plt.subplots(1,1, figsize=(5,5))
#plt.title(model_name)

#perplexity = 30
#perplexity = 80
perplexity = 60
#perplexity = 50
#perplexity = 60 # best
n_components = 2
tsne = manifold.TSNE(
    n_components=n_components,
    init="random",
    random_state=0,
    perplexity=perplexity,
    learning_rate="auto",
    n_iter=500,
)


#results_v2 = tsne.fit_transform(item_emb)
# concat 2-hop embs
#user_double = np.concatenate([user_emb, sim_total], axis=0)
#user_emb_2d = tsne.fit_transform(user_double)
#user_emb_2d = tsne.fit_transform(item_emb)
user_emb_2d = tsne.fit_transform(user_emb)

#print(user_emb_2d.shape)
#print(user_emb_2d[:10])
user_emb_2d = normalize(user_emb_2d, axis=1,norm='l2')
#user_emb_ori = user_emb_2d[:len(user_emb_2d)//2, :]
#user_emb_hop = user_emb_2d[len(user_emb_2d)//2:, :]

#user_emb_ori = tsne.fit_transform(user_emb)
#user_emb_hop = tsne.fit_transform(np.array(sim_total))
#user_emb_ori = normalize(user_emb_ori, axis=1,norm='l2')
#user_emb_hop = normalize(user_emb_hop, axis=1,norm='l2')


#cmap = plt.cm.jet  # define the colormap
#cmap = plt.cm.get_cmap('GnBu', 10)
cmap = plt.cm.get_cmap('GnBu', 8)
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (1., 1., 1., 1.0)
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
# define the bins and normalize
#bounds = np.linspace(0, 20, 21)
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


#sns.set_style("darkgrid")
#sns.set(rc={'axes.facecolor':'cornflowerblue', 'figure.facecolor':'cornflowerblue'})
kwargs = {'levels': np.arange(0, 4.2, 0.5)}

sns.kdeplot(data=user_emb_2d, bw=0.05, shade=True, cmap=cmap, legend=True, **kwargs)
#sns.kdeplot(data=user_emb_2d, bw=0.05, shade=True, cmap=cmap, norm=norm, legend=True, **kwargs)
#sns.kdeplot(data=user_emb_2d, bw=0.05, shade=True, cmap="GnBu", legend=True, **kwargs)
#sns.kdeplot(data=user_emb_ori, bw=0.05, shade=True, cmap="GnBu", ax=axs[0][0], legend=True, **kwargs)
#sns.kdeplot(data=user_emb_hop, bw=0.05, shade=True, cmap="GnBu", ax=axs[0][1], legend=True, **kwargs)


#x = [p[0] for p in user_emb_ori]
#y = [p[1] for p in user_emb_ori]
#angles_ori = np.arctan2(y,x)
#print("mean density: ", np.mean(y))
#print("var density: ", np.var(y))
##print("angles_ori density: ", np.var(angles_ori))
#kde = KernelDensity(kernel='gaussian', bandwidth=0.15).fit(angles_ori.reshape(-1, 1))
#kde_scores = kde.score_samples(angles_ori.reshape(-1,1))
#print(kde_scores)
#print("angles_ori density: ", np.var(kde_scores))
#pdb.set_trace()

#x = [p[0] for p in user_emb_hop]
#y = [p[1] for p in user_emb_hop]
#angles_hop = np.arctan2(y,x)
#sns.kdeplot(data=angles_ori, bw=0.15, shade=True,legend=True,ax=axs[1][0],color='green')
#sns.kdeplot(data=angles_hop, bw=0.15, shade=True,legend=True,ax=axs[1][1],color='green')




#sns.kdeplot(data=user_emb_2d, bw=0.05, shade=True, cmap='GnBu', ax=axs[0], legend=True, **kwargs)
#plt.set_title('SimGCL', fontsize = 9,fontweight="bold")
#x = [p[0] for p in user_emb_2d]
#y = [p[1] for p in user_emb_2d]
#angles = np.arctan2(y,x)
#sns.kdeplot(data=angles, bw=0.15, shade=True,legend=True,ax=axs[1][4],color='green')
#sns.kdeplot(data=angles, bw=0.15, shade=True,legend=True,color='green')
#plt.show()
#plt.savefig('{}_user_emb_rel_item.png'.format(model_name))
#plt.rcParams['axes.facecolor']='snow'
#axs.set_facecolor("orange")


my_x_ticks = np.arange(-1, 1.2, 1)
plt.xticks(my_x_ticks, fontsize=10)
axs.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

#plt.tick_params(width=20, labelsize=4)
plt.title(model_name,fontdict={'size':14}, fontweight="bold", fontname="Times New Roman")

plt.savefig('{}_final.pdf'.format(model_name))

exit(0)

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

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = np.reshape(kernel(positions).T, X.shape)
    #print(Z.shape)
    ax.imshow(np.rot90(Z), cmap=plt.cm.viridis,
              extent=[xmin, xmax, ymin, ymax])

fig, axarr = plt.subplots(1, 2)
kde1(results_v2[:, 0], results_v2[:, 1], axarr[0])
plt.tight_layout()
plt.savefig('kde.png')
plt.show()
