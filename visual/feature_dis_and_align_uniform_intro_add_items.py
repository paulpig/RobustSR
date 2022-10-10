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
from sklearn.preprocessing import normalize
from sklearn import preprocessing
#import pickle5 as pickle




# model_name = 'only_ppr'
# model_name = 'ppr_v2' # true
# model_name = '0_1_score' # wrong
#model_name = 'only_output_subgraph'
#model_name = 'add_inter_social_rep'
# model_name = 'only_cl' # wrong;
# model_name = 'diffNet'
# model_name = 'LightGCN'
# model_name = 'LightGCN_v1'
# model_name = 'diffNet_v3'
# model_name = 'diffNet_v2'
# model_name = 'diffNet_v5'
# model_name = 'diffNet_v5_true'
# model_name = 'ESRF'
# model_name = 'MHCN'
# model_name = 'MHCN_rm_self'
# model_name = 'MHCN_v2'
# model_name = 'SEPT'
# model_name = 'SEPT_v1'
#model_name = 'SCIL_v1'
#model_name = 'SCIL'
#model_name = 'SCIL_v2'
model_name = 'SCIL_v3'
#model_name = 'DANSER'
#model_name = 'fuse'
#model_name = 'SERec'


# model_name = 'douban_book/only_cl_v2' # right;
# model_name = 'douban_book/ppr_v2' # right;
# model_name = 'douban_book/LightGCN_v1' # right;
# model_name = 'douban_book/SCIL_v3' # right;


print(model_name)

user_emb = np.load('./{}/user_emb.npy'.format(model_name))
item_emb = np.load('./{}/item_emb.npy'.format(model_name))

# user_emb = preprocessing.normalize(user_emb, norm='l2')
# item_emb = preprocessing.normalize(item_emb, norm='l2')

# min_max_scaler = preprocessing.MinMaxScaler()
# user_emb = min_max_scaler.fit_transform(user_emb)
# item_emb = min_max_scaler.fit_transform(item_emb)

# pdb.set_trace()
# user_emb = (user_emb) / np.reshape(np.linalg.norm(user_emb, axis=1), (-1, 1))
# item_emb = (item_emb) / np.reshape(np.linalg.norm(item_emb, axis=1), (-1, 1))


from sklearn.preprocessing import normalize
import seaborn as sns
#fig, axs = plt.subplots(2,2)
# fig, axs = plt.subplots(1,1, figsize=(5,5))
fig, axs = plt.subplots(2,1, figsize=(8,8))
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

user2id = dict([(value, key) for key, value in id2user.items()])
item2id = dict([(value, key) for key, value in id2item.items()])

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


# uniform
#pdb.set_trace()
Y = cdist(user_emb, user_emb, 'minkowski', p=2.)
#norm_tmp = np.mean(np.power(Y, 2))
#true_tmp = ( np.power(Y, 2) - norm_tmp ) / norm_tmp
#uniform = np.log(np.mean(np.exp(-2*true_tmp)))
uniform = np.log(np.mean(np.exp(-2*np.power(Y, 2))))
print("uniform score:", uniform)
#pdb.set_trace()

# 随机选一个用户
def calculate_distance(user_id_str):
    user_id = user_id_str
    center_user_emb = user_emb[user_id]
    neighbor_users = user_emb[user2users[user_id]]
    #centor_user_dis =np.mean(np.power(np.linalg.norm(np.reshape(center_user_emb, (1, -1)) - neighbor_users),
    #centor_user_dis =np.mean(np.power(np.linalg.norm(np.reshape(center_user_emb, (1, -1)) - neighbor_users), 2))
    #centor_user_dis = np.power(np.linalg.norm(np.reshape(center_user_emb, (1, -1)) - neighbor_users, axis=1), 2)
    #centor_user_dis = np.linalg.norm(np.reshape(center_user_emb, (1, -1)) - neighbor_users, axis=1)
    #true_value = np.power(np.linalg.norm(np.reshape(center_user_emb, (1, -1)) - neighbor_users), 2)
    #true_value = (true_value - norm_tmp) / norm_tmp
    #centor_user_dis =np.mean(np.power(np.linalg.norm(np.reshape(center_user_emb, (1, -1)) - neighbor_users), 2))
    #centor_user_dis = np.mean(true_value)

    # v1
    #centor_user_dis = np.power(np.linalg.norm(np.reshape(center_user_emb, (1, -1)) - neighbor_users, axis=1), 2)
    # v2
    centor_user_dis = [np.power(np.linalg.norm(center_user_emb - np.mean(neighbor_users, 0), axis=0), 2)]
    return centor_user_dis


def calculate_distance_for_items(user_id_str):
    user_id = user_id_str
    center_user_emb = user_emb[user_id]
    neighbor_items = item_emb[user2items[user_id]]
    #centor_user_dis =np.mean(np.power(np.linalg.norm(np.reshape(center_user_emb, (1, -1)) - neighbor_items), 2))
    centor_user_dis = [np.power(np.linalg.norm(center_user_emb - np.mean(neighbor_items, 0), axis=0), 2)]
    return centor_user_dis



# get 1-hop
sim_total = []
for index in range(len(user_emb)):
    index_data = "{}".format(index)
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
    #pdb.set_trace()
    #sim_total.append(sim)
    sim_total.extend(sim)

#pdb.set_trace()
print("UU-align score: ", np.mean(sim_total))
#print("user item match score: ",np.log(np.mean(sim_total)))

for index in range(len(user_emb)):
    index_data = "{}".format(index)
    #index_data = index + 1
    if index_data not in user2id:
        continue

    if user2id[index_data] not in user2users:
        continue

    #if len(user2users[user2id[index_data]]) < 2 or len(user2users[user2id[index_data]]) > 10:
    #    continue
    #if index > 1000:
    #    break
    sim = calculate_distance_for_items(user2id[index_data])
    sim_total.extend(sim)

print("UI-align score: ", np.mean(sim_total))


# exit(0)
#perplexity = 30
#perplexity = 80
#perplexity = 80
#perplexity = 50
perplexity = 60 # best

n_components = 2
# tsne = manifold.TSNE(
#    n_components=n_components,
#    init="random",
#    random_state=4,
#    perplexity=perplexity,
#    learning_rate="auto",
#    n_iter=700,
# )

# good results
# tsne = manifold.TSNE(
#    n_components=n_components,
#    init="random",
#    random_state=8,
#    perplexity=perplexity,
#    learning_rate="auto",
#    n_iter=500,
# )

#PCA
# pca = PCA(n_components=2)
# user_emb_2d = pca.fit_transform(user_emb)

tsne = manifold.TSNE(
    n_components=n_components,
    init="random",
    random_state=0,
    perplexity=perplexity,
    learning_rate="auto",
    n_iter=600,
)



# plt.scatter(user_emb_2d[:,0], user_emb_2d[:,1])
# plt.show()
# exit(0)

#results_v2 = tsne.fit_transform(item_emb)
# concat 2-hop embs
#user_double = np.concatenate([user_emb, sim_total], axis=0)
#user_emb_2d = tsne.fit_transform(user_double)
#user_emb_2d = tsne.fit_transform(item_emb)
# t-sne
user_emb_2d = tsne.fit_transform(user_emb)

#print(user_emb_2d.shape)
#print(user_emb_2d[:10])
user_emb_2d = normalize(user_emb_2d, axis=1,norm='l2')


#cmap = plt.cm.jet  # define the colormap
#cmap = plt.cm.get_cmap('GnBu', 10)
#cmap = plt.cm.get_cmap('GnBu', 8)
#cmap = plt.cm.get_cmap('BrBG', 8)
cmap = plt.cm.get_cmap('BuPu', 20)
# cmap = plt.cm.get_cmap('BuPu', 10)
# cmap = plt.cm.get_cmap('BuPu', 25)
print(cmap.N)
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey

cmaplist = cmaplist[-15:]
cmaplist[0] = (1., 1., 1., 1.0)
# create the new map
# cmap = mpl.colors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap.N)

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, len(cmaplist))
# define the bins and normalize
#bounds = np.linspace(0, 20, 21)
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


#sns.set_style("darkgrid")
#sns.set(rc={'axes.facecolor':'cornflowerblue', 'figure.facecolor':'cornflowerblue'})
kwargs = {'levels': np.arange(0, 4.2, 0.5)}

sns.kdeplot(data=user_emb_2d, bw=0.05, shade=True, cmap=cmap, ax=axs[0], legend=True, **kwargs)
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
x = [p[0] for p in user_emb_2d]
y = [p[1] for p in user_emb_2d]
angles = np.arctan2(y,x)
# sns.kdeplot(data=angles, bw=0.15, shade=True,legend=True,ax=axs[1][4],color='green')
sns.kdeplot(data=angles, bw=0.15, shade=True,legend=True, ax=axs[1], color='blue')
#plt.show()
#plt.savefig('{}_user_emb_rel_item.png'.format(model_name))
#plt.rcParams['axes.facecolor']='snow'
#axs.set_facecolor("orange")


# my_x_ticks = np.arange(-1, 1.2, 1)
# plt.xticks(my_x_ticks, fontproperties = 'Times New Roman', size=23)
# my_y_ticks = np.arange(-1, 1.2, 1)
# plt.yticks(my_y_ticks, fontproperties = 'Times New Roman', size=23)

# axs.spines['bottom'].set_linewidth(2.2);###设置底部坐标轴的粗细
# axs.spines['left'].set_linewidth(2.2);####设置左边坐标轴的粗细
# axs.spines['right'].set_linewidth(2.2);###设置右边坐标轴的粗细
# axs.spines['top'].set_linewidth(2.2);####设置上部坐标轴的粗细
# axs.set_xlabel("Features", fontsize=30, fontdict={'family': 'Times New Roman'})
# axs.set_ylabel("Features", fontsize=30, fontdict={'family': 'Times New Roman'}, labelpad=-5.5)

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
