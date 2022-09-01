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
from scipy.spatial.distance import cdist
import pickle5 as pickle
from sklearn import preprocessing




#model_name = 'only_ppr'
#model_name = 'ppr_v2'
#model_name = '0_1_score' # wrong
#model_name = 'only_output_subgraph'
#model_name = 'add_inter_social_rep'
#model_name = 'only_cl'
#model_name = 'diffNet'
#model_name = 'LightGCN'
#model_name = 'LightGCN_v1'
#model_name = 'diffNet_v3'
#model_name = 'diffNet_v2'
#model_name = 'ESRF'
#model_name = 'MHCN'
#model_name = 'SEPT'
#model_name = 'SEPT_v1'
#model_name = 'SCIL_v1'
#model_name = 'SCIL'
#model_name = 'SCIL_v2'
#model_name = 'SCIL_v3'
#model_name = 'DANSER'
model_name = 'fuse'
#model_name = 'danser_v2'
print(model_name)

user_emb = np.load('./{}/user_emb.npy'.format(model_name))
#user_emb = preprocessing.scale(user_emb)
item_emb = np.load('./{}/item_emb.npy'.format(model_name))
#item_emb = preprocessing.scale(item_emb)
#user_emb = (user_emb) / np.reshape(np.linalg.norm(user_emb, axis=1), (-1, 1)) /2
#item_emb = (item_emb) / np.reshape(np.linalg.norm(item_emb, axis=1), (-1, 1))

from sklearn.preprocessing import normalize
import seaborn as sns
#fig, axs = plt.subplots(2,2)
fig, axs = plt.subplots(1,1, figsize=(5,5))
#plt.title(model_name)
#pdb.set_trace()


# read dataset
dataset_name = 'lastfm'
#dataset_name = 'FilmTrust'
#dataset_path = '/pub/data/kyyx/wbc/QRec/dataset/{}/'.format(dataset_name)
#dataset_path = '/pub/data/kyyx/wbc/QRec/dataset/{}/'.format(dataset_name)
#dataset_path = './FilmTrust/'
dataset_path = './lastfm/'

rating = dataset_path + 'ratings.txt'
social = dataset_path + 'trusts.txt'

# 随机选择一个用户, 检索用户的社交邻居和交互的items;
delim = ' |,|\t'

#user2id = {}

with open(rating) as f:
    ratings = f.readlines()


# DANSER模型对应的user2id
#user2id = {}
#item2id = {}
#user2items = {}
#for lineno, line in enumerate(ratings):
#    order = split(delim,line.strip())
#    #if order[0] not in user2id or order[1] not in item2id:
#    #    continue
#    #userid = user2id[order[0]]
#    #itemid = item2id[order[1]]
#    if order[0] not in user2id:
#        user2id[order[0]] = len(user2id)
#    if order[1] not in item2id:
#        item2id[order[1]] = len(item2id)
#    userid = user2id[order[0]]
#    itemid = item2id[order[1]]
#    if userid not in user2items:
#        user2items[userid] = [itemid]
#    else:
#        user2items[userid].append(itemid)


with open('./{}/id2user.pickle'.format(model_name), 'rb') as handle:
    id2user = pickle.load(handle)

with open('./{}/id2item.pickle'.format(model_name), 'rb') as handle:
    id2item = pickle.load(handle)

#user2id = dict([(value, key) for key, value in id2user.items()])
#item2id = dict([(value, key) for key, value in id2item.items()])
user2id = id2user
item2id = id2item

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

pdb.set_trace()
user2users = {}
for lineno, line in enumerate(social_raw):
    order = split(delim,line.strip())
    if int(order[0]) not in user2id or int(order[1]) not in user2id:
        continue
    userid = user2id[int(order[0])]
    userid2 = user2id[int(order[1])]

    if userid not in user2users:
        user2users[userid] = [userid2]
    else:
        user2users[userid].append(userid2)


# 随机选一个用户
def calculate_distance(user_id_str):
# 随机选一个用户
#print(user2users)
    #user_id = 190
    #user_id = user2id[user_id_str]
    #print(user2users[user_id])
    #print(user2items[user_id])
    user_id = user_id_str

    center_user_emb = user_emb[user_id]
    neighbor_users = user_emb[user2users[user_id]]
    #neighbor_items = item_emb[user2items[user_id]]
    #neighbor_users_ids = user2users[user_id]

    #tmp_array = neighbor_users_ids.copy()
# on#ly 2-hop
#2_h#op_neighbor_users = []
    #hop2_neighbor_users = []
    #hop3_neighbor_users = []
    #for index, user_tmp_id in enumerate(tmp_array):
    #    #if len(user2users[user_tmp_id]) > 50:
    #    #    continue
    #    #print(tmp_array)
    #    #neighbor_users_ids.extend(user2users[user_tmp_id])

    #    if user_tmp_id not in user2users:
    #        continue
    #    hop2_neighbor_users.extend(user2users[user_tmp_id]) # 2-hop
    #    tmp_3_array = user2users[user_tmp_id].copy()
    #    #for index, user_tmp_id_3 in enumerate(tmp_3_array):
    #    #    hop3_neighbor_users.extend(user2users[user_tmp_id_3]) # 3-hop

    ##neighbor_users = user_emb[neighbor_users_ids]
    #hop2_neighbor_users_emb = user_emb[hop2_neighbor_users]
    #hop3_neighbor_users_emb = user_emb[hop3_neighbor_users]

    # distance
    #centor_user_dis = np.mean(np.reshape(center_user_emb, (1, -1)) - neighbor_users, axis=0)
    #centor_item_dis = np.mean(np.reshape(center_user_emb, (1, -1)) - neighbor_items, axis=0)

    # l2 distance, alignment
    #distance_user_item = np.linalg.norm(centor_user_dis-centor_item_dis)
    #print("distance: ", distance_user_item)
    #centor_user_dis =np.power( np.mean(np.linalg.norm(np.reshape(center_user_emb, (1, -1)) - neighbor_users)), 2) * 20
    #centor_user_dis =np.mean(np.power(np.linalg.norm(np.reshape(center_user_emb, (1, -1)) - neighbor_users), 2))
    #centor_user_dis =np.mean(np.power(np.linalg.norm(np.reshape(center_user_emb, (1, -1)) - neighbor_users), 2))
    centor_user_dis = np.power(np.linalg.norm(center_user_emb - np.mean(neighbor_users, 0), axis=0), 2)
    return centor_user_dis

    # uniform
    #random_user_id = random.randint(0, len(user_emb)) -1
    ##while random_user_id == user_id or random_user_id not in id2user.keys():
    #while random_user_id == user_id:
    #    random_user_id = random.randint(0, len(user_emb)) - 1

    #random_user_emb = user_emb[random_user_id]
    ##pdb.set_trace()
    ##return np.exp(-2*np.power(np.linalg.norm(np.linalg.norm(center_user_emb) - np.linalg.norm(random_user_emb)), 2)) /20.0
    #return np.exp(-2*np.power(np.linalg.norm(np.linalg.norm(center_user_emb) - np.linalg.norm(random_user_emb)), 2))
    ##return np.exp(-2*np.power(np.linalg.norm(center_user_emb - random_user_emb), 2))

    # cos sim
    #cos_sim = cosine_similarity(center_user_emb.reshape(1, -1), np.mean(hop2_neighbor_users_emb, axis=0).reshape(1, -1))
    #return cos_sim

    # return 1-hop neighbor emb
    #return np.mean(hop2_neighbor_users_emb, axis=0)
    #return np.mean(neighbor_users, axis=0)
    #return np.mean(neighbor_items, axis=0)

# uniform
Y = cdist(user_emb, user_emb, 'minkowski', p=2.)
uniform = np.log(np.mean(np.exp(-2*np.power(Y, 2))))
print("uniform score:", uniform)
#pdb.set_trace()

# get 1-hop
sim_total = []
for index in range(max(user2id.keys())):
    #pdb.set_trace()
    #index_data = "{}".format(index)
    index_data = index
    #index_data = index + 1
    if index_data not in user2id:
        continue

    if user2id[index_data] not in user2users:
        continue

    #if len(user2users[user2id[index_data]]) < 2 or len(user2users[user2id[index_data]]) > 10:
    #    continue
    #if index > 1000:
    #    break

    sim = calculate_distance(user2id[index_data])
    sim_total.append(sim)

#pdb.set_trace()
print("align score: ", np.mean(sim_total))
#print("user item match score: ",np.log(np.mean(sim_total)))


#perplexity = 30
#perplexity = 80
perplexity = 60
#perplexity = 50
#perplexity = 60 # best

n_components = 2
#tsne = manifold.TSNE(
#    n_components=n_components,
#    init="random",
#    random_state=0,
#    perplexity=perplexity,
#    learning_rate="auto",
#    n_iter=500,
#)

# good results
#tsne = manifold.TSNE(
#    n_components=n_components,
#    init="random",
#    random_state=8,
#    perplexity=perplexity,
#    learning_rate="auto",
#    n_iter=500,
#)

tsne = manifold.TSNE(
    n_components=n_components,
    init="random",
    random_state=2,
    perplexity=perplexity,
    learning_rate="auto",
    n_iter=600,
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
#cmap = plt.cm.get_cmap('GnBu', 8)
#cmap = plt.cm.get_cmap('BrBG', 8)
cmap = plt.cm.get_cmap('BuPu', 15)

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
plt.xticks(my_x_ticks, fontsize=15)
my_y_ticks = np.arange(-1, 1.2, 1)
plt.yticks(my_y_ticks, fontsize=15)
my_y_ticks = np.arange(-1, 1.2, 1)
axs.spines['bottom'].set_linewidth(2.2);###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(2.2);####设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(2.2);###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(2.2);####设置上部坐标轴的粗细
axs.set_xlabel("Features", fontsize=15)
axs.set_ylabel("Features", fontsize=15, labelpad=-5.5)

#plt.tick_params(width=20, labelsize=4)
#plt.title(model_name,fontdict={'size':14}, fontweight="bold", fontname="Times New Roman")

plt.savefig('./result/{}_final_v2.pdf'.format(model_name))
plt.show()

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
