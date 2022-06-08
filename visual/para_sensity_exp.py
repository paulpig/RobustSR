# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import MultipleLocator, colorbar

# 全局设置字体为：Times New Roman
plt.rc('font',family='Times New Roman')


# 0.6137		0.6131		0.6157
# 0.6204		0.62		0.6217
# 0.62		0.6177		0.629
# 0.6245		0.6231		0.6279
# 0.6194		0.6231		0.6189
# 数据准备
# x = [10, 20, 30, 40]
#x = [10, 20, 30, 40, 50]
x = [10, 20, 30, 40, 50, 60, 70]
# y1 = [5.825, 2.938, 1.573, 0.994, 0.729]
y1 = [0.5570, 0.5994, 0.5265, 0.6048]
y2 = [0.328, 0.191, 0.154, 0.310]
# y3 = [1.297, 0.826, 0.815, 0.741, 0.698]

x_min = 0.10
x_max = 0.55
dataset = '(a) Amazon Beauty'
metrics = ""

# 在柱状图上显示数字
def auto_text(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 1 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)

# 创建一个画布
# fig, ax_list = plt.subplots(ncols=2, nrows=2, figsize=(10,5),dpi=100)
#fig, ax_list = plt.subplots(ncols=4, nrows=1, figsize=(16,4),dpi=100)
#fig, ax_list = plt.subplots(ncols=3, nrows=1, figsize=(24,4.6))
fig, ax_list = plt.subplots(ncols=1, nrows=1, figsize=(7,5))
#plt.subplots_adjust(wspace =0.25, hspace =0.5)
plt.subplots_adjust(wspace =0.6, hspace =0.0)
# fig, ax = plt.subplots(figsize=(7,5),dpi=100)
# print("fig, ax:", fig, ax)

# plt.subplots_adjust(wspace =0.25, hspace =0)

tmp_index = 1
#for index, ax in enumerate(ax_list):
if ax_list is not None:
    ax = ax_list
    tmp_index = 5
    #for ax in ax_two
    if tmp_index ==0:
        # y1 = [0.502, 0.330, 0.297, 0.477]
#             y1 = [0.5570, 0.5994, 0.5265, 0.6204] # HR@10
#             y2 = [0.328, 0.191, 0.154, 0.310] #HR@5

        #y1 = [0.6137, 0.6204, 0.6245, 0.6194] #multi
        y1 = [0.20099, 0.20115, 0.20208, 0.20133, 0.20203, 0.20053, 0.20122]
        y2 = [0.20385, 0.20420, 0.20521, 0.20434, 0.20512, 0.20381, 0.20414]
        y3 = [0.24621, 0.24529, 0.24746, 0.24651, 0.24704, 0.24636, 0.24703]
#         y2 = [0.4919, 0.5994] #tranR(IA)
#         y3 = [0.4199, 0.5265] #tranR(CB)
#         y4 = [0.5123, 0.6204] #crbiaSR
        # y3 = [1.297, 0.826, 0.815, 0.741, 0.698]
        labels = ['30', '20', '10', '5.0', '2.0', '1.0', '0.5']

        x_min = 0.16 * 100
        x_max = 0.25 * 100
        # ax.set_ylabel('NDCG@10')
        #ax.set_ylabel('HR@K')
        metrics = "Performance %"
        dataset = 'alpha'
        metric = "lastfm"
        title_name = "LastFM"


    if tmp_index ==1:
        # y1 = [0.502, 0.330, 0.297, 0.477]
#             y1 = [0.5570, 0.5994, 0.5265, 0.6204] # HR@10
#             y2 = [0.328, 0.191, 0.154, 0.310] #HR@5

#         y1 = [0.3345, 0.3696] #multi
#         y2 = [0.3700, 0.4049] #tranR(IA)
#         y3 = [0.3113, 0.3457] #tranR(CB)
#         y4 = [0.3875, 0.4225] #crbiaSR
        #y1 = [0.6131, 0.62, 0.6177, 0.6231, 0.6231] #multi
        #y1 = [0.6131, 0.62, 0.6231, 0.6231] #multi
        # y3 = [1.297, 0.826, 0.815, 0.741, 0.698]
        y1 = [0.17913, 0.17917, 0.17975, 0.20038, 0.20237, 0.20147, 0.20136]
        y2 = [0.18174, 0.18185, 0.18219, 0.20353, 0.20582, 0.20473, 0.20452]
        y3 = [0.22074, 0.22007, 0.22167, 0.24552, 0.24827, 0.24738, 0.24663]

        labels = ['1.0', '0.5', '0.1', '0.01', '0.005', '0.001', '0.0005']

        x_min = 0.17 * 100
        x_max = 0.25 * 100
        # ax.set_ylabel('NDCG@10')
        #ax.set_ylabel('NDCG@K')
        metrics = "Performance %"
        dataset = 'beta'
        #metrics = "HR@10"
        #dataset = 'Number of layers'
        metric = "lastfm"
        title_name = "LastFM"


    if tmp_index ==2:
#         y1 = [0.4117, 0.5432] #multi
#         y2 = [0.4671, 0.5989] #tranR(IA)
#         y3 = [0.3995, 0.5377] #tranR(CB)
#         y4 = [0.4860, 0.6200] #crbiaSR
        # y3 = [1.297, 0.826, 0.815, 0.741, 0.698]
        #y1 = [0.6157, 0.6217, 0.6279, 0.6189] #multi
        # origin
        #y1 = [0.08286, 0.08318, 0.08289, 0.08192, 0.08127, 0.08058, 0.07993, 0.07992]
        #y2 = [0.10868, 0.10960, 0.10919, 0.10755, 0.10695, 0.10414, 0.10457, 0.10300]
        #y3 = [0.12723, 0.12686, 0.12674, 0.12424, 0.12326, 0.12017, 0.12019, 0.11903]

        # remove 20
        y1 = [0.08286, 0.08370, 0.08192, 0.08127, 0.08058, 0.07993, 0.07992]
        y2 = [0.10868, 0.10967, 0.10755, 0.10695, 0.10414, 0.10457, 0.10300]
        y3 = [0.12723, 0.12857, 0.12424, 0.12326, 0.12017, 0.12019, 0.11903]

        labels = ['50.0', '25.0', '10.0', '5.0', '1.0', '0.5', '0.005']

        x_min = 0.055 * 100
        x_max = 0.13 * 100
        # ax.set_ylabel('NDCG@10')
        #ax.set_ylabel('HR@K')
        metrics = "Performance %"
        dataset = 'alpha'
        #metrics = "HR@10"
        #dataset = 'Number of layers'
        metric = "douban_book"
        title_name = "Douban-Book"

    if tmp_index ==3:
        # y1 = [0.502, 0.330, 0.297, 0.477]
#             y1 = [0.5570, 0.5994, 0.5265, 0.6204] # HR@10
#             y2 = [0.328, 0.191, 0.154, 0.310] #HR@5

        #y1 = [0.2971, 0.3395] #multi
        #y2 = [0.3414, 0.3841] #tranR(IA)
        #y3 = [0.2849, 0.3295] #tranR(CB)
        #y4 = [0.3554, 0.3988] #crbiaSR
        # y3 = [1.297, 0.826, 0.815, 0.741, 0.698]
        y1 = [0.07565, 0.07541, 0.07700, 0.08330, 0.08318, 0.08284, 0.08261]
        y2 = [0.09502, 0.09439, 0.09987, 0.10916, 0.10960, 0.10894, 0.10886]
        y3 = [0.11034, 0.11017, 0.11809, 0.12726, 0.12686, 0.12648, 0.12611]

        #labels = ['50.0', '25.0', '10.0', '5.0', '1.0', '0.5', '0.005']
        labels = ['1.0', '0.5', '0.1', '0.01', '0.005', '0.001', '0.0005']

        x_min = 0.065 * 100
        x_max = 0.13 * 100
        # ax.set_ylabel('NDCG@10')
        #ax.set_ylabel('NDCG@K')
        metrics = "Performance %"
        dataset = 'beta'
        metric = "douban_book"
        title_name = "Douban-Book"

    if tmp_index ==4:
        # y1 = [0.502, 0.330, 0.297, 0.477]
#             y1 = [0.5570, 0.5994, 0.5265, 0.6204] # HR@10
#             y2 = [0.328, 0.191, 0.154, 0.310] #HR@5

        #y1 = [0.2971, 0.3395] #multi
        #y2 = [0.3414, 0.3841] #tranR(IA)
        #y3 = [0.2849, 0.3295] #tranR(CB)
        #y4 = [0.3554, 0.3988] #crbiaSR
        # y3 = [1.297, 0.826, 0.815, 0.741, 0.698]
        #y1 = [0.07565, 0.07541, 0.07700, 0.08330, 0.08318, 0.08284, 0.08261]
        #y2 = [0.09502, 0.09439, 0.09987, 0.10916, 0.10960, 0.10894, 0.10886]
        #y3 = [0.11034, 0.11017, 0.11809, 0.12726, 0.12686, 0.12648, 0.12611]
        y1 = [0.2019, 0.2024, 0.2039, 0.2021, 0.1797, 0.1794, 0.1792]
        y2 = [0.2054, 0.2058, 0.2072, 0.2053, 0.1822, 0.1820, 0.1818]
        y3 = [0.2477, 0.2483, 0.2505, 0.2490, 0.2208, 0.2212, 0.2199]

        #labels = ['50.0', '25.0', '10.0', '5.0', '1.0', '0.5', '0.005']
        #labels = ['1.0', '0.5', '0.1', '0.01', '0.005', '0.001', '0.0005']
        labels = ['0.005', '0.01', '0.05', '0.1', '5.0', '10.0', '20.0']

        #x_min = 0.065 * 100
        #x_max = 0.13 * 100
        x_min = 0.16 * 100
        x_max = 0.27 * 100
        # ax.set_ylabel('NDCG@10')
        #ax.set_ylabel('NDCG@K')
        metrics = "Performance %"
        dataset = 'delta'
        metric = "lastfm"
        title_name = "LastFM"
    if tmp_index ==5:
        # y1 = [0.502, 0.330, 0.297, 0.477]
#             y1 = [0.5570, 0.5994, 0.5265, 0.6204] # HR@10
#             y2 = [0.328, 0.191, 0.154, 0.310] #HR@5

        #y1 = [0.2971, 0.3395] #multi
        #y2 = [0.3414, 0.3841] #tranR(IA)
        #y3 = [0.2849, 0.3295] #tranR(CB)
        #y4 = [0.3554, 0.3988] #crbiaSR
        # y3 = [1.297, 0.826, 0.815, 0.741, 0.698]
        #y1 = [0.07565, 0.07541, 0.07700, 0.08330, 0.08318, 0.08284, 0.08261]
        #y2 = [0.09502, 0.09439, 0.09987, 0.10916, 0.10960, 0.10894, 0.10886]
        #y3 = [0.11034, 0.11017, 0.11809, 0.12726, 0.12686, 0.12648, 0.12611]
        #y1 = [0.2193, 0.2024, 0.2039, 0.2021, 0.1797, 0.1794, 0.1792]
        #y2 = [0.2054, 0.2058, 0.2072, 0.2053, 0.1822, 0.1820, 0.1818]
        #y3 = [0.2477, 0.2483, 0.2505, 0.2490, 0.2208, 0.2212, 0.2199]
        y1 = [0.08301, 0.08238, 0.08370, 0.07709, 0.07489, 0.07536, 0.07558]
        y2 = [0.10795, 0.10631, 0.10967, 0.10037, 0.09616, 0.09556, 0.09491]
        y3 = [0.12577, 0.12411, 0.12857, 0.11674, 0.11139, 0.11119, 0.11041]

        #labels = ['50.0', '25.0', '10.0', '5.0', '1.0', '0.5', '0.005']
        #labels = ['1.0', '0.5', '0.1', '0.01', '0.005', '0.001', '0.0005']
        labels = ['0.005', '0.05', '0.1', '1.0', '5.0', '10.0', '20.0']

        x_min = 0.065 * 100
        x_max = 0.131 * 100
        #x_min = 0.16 * 100
        #x_max = 0.25 * 100
        # ax.set_ylabel('NDCG@10')
        #ax.set_ylabel('NDCG@K')
        metrics = "Performance %"
        dataset = 'delta'
        #metric = "lastfm"
        #title_name = "LastFM"
        metric = "douban_book"
        title_name = "Douban-Book"
        #metric = "douban_book"
        #title_name = "Douban-Book"
        #metrics = "NDCG@K"
        #dataset = '(d) NDCG@K on Amazon Sports'
#         if tmp_index > 0:
#             break
#         if tmp_index == 1:
#             y1 = [0.5570, 0.5994, 0.5265, 0.6204]
#             y2 = [0.328, 0.191, 0.154, 0.310]
#             # y3 = [1.297, 0.826, 0.815, 0.741, 0.698]

#             x_min = 0.4
#             x_max = 0.62
#             # ax.set_ylabel('NDCG@10')
#             ax.set_ylabel('HR@10')
#             dataset = '(a) Amazon Beauty'
#         if index == 1:
#             # y1 = [0.552, 0.525, 0.546, 0.575]
#             y1 = [0.5432, 0.5989, 0.5377, 0.6200]
#             y2 = [0.358, 0.328, 0.347, 0.372]
#             # x_min = 0.3
#             # x_max = 0.62
#             x_min = 0.4
#             x_max = 0.62
#             dataset = '(b) Amazon Sports'

#     if index == 2:
#         y1 = [0.5621, 0.6049, 0.5537, 0.6217]
#         y2 = [0.358, 0.328, 0.347, 0.372]
#         x_min = 0.4
#         x_max = 0.62
#         dataset = '(c) Amazon Toys'



    # ax.spines['right'].set_visible(True)
    # ax = ax_list[0]
    # print("fig, ax:", fig, ax)
    # 画柱状
    width = 2 # 柱状的宽度
    # rect1 = ax.bar(np.asarray(x)-width, y1, width=width, color='C0', alpha=0.5, label='1')
    # rect1 = ax.bar(np.asarray(x) - width/2, y1, width=width, color='#B0E0E6', alpha=0.4, label='HR@10')
    # rect2 = ax.bar(np.asarray(x) + width/2, y2, width=width, color='#FFDEAD', alpha=0.4, label='NDCG@10')

    #ax.plot(x,[item*100 for item in y1[::-1]],'b^-',label='P@10', linewidth=3, markersize=8)
    #ax.plot(x,[item*100 for item in y1[::-1]],'^-', color='#ED8A3F', label='P@10', linewidth=3, markersize=8, zorder=3)
    #ax.plot(x,[item*100 for item in y2[::-1]],'r*-',color='#F5D769', label='R@10', linewidth=3, markersize=8, zorder=3)
    #ax.plot(x,[item*100 for item in y3[::-1]],'gd-',color='#8DB87C', label='N@10', linewidth=3, markersize=8, zorder=3)
    if tmp_index == 4 or tmp_index ==5:
        ax.plot(x,[item*100 for item in y1],'b^-', label='P@10', linewidth=3, markersize=8, zorder=3)
        ax.plot(x,[item*100 for item in y2],'r*-', label='R@10', linewidth=3, markersize=8, zorder=3)
        ax.plot(x,[item*100 for item in y3],'gd-', label='N@10', linewidth=3, markersize=8, zorder=3)

    else:
        ax.plot(x,[item*100 for item in y1[::-1]],'b^-', label='P@10', linewidth=3, markersize=8, zorder=3)
        ax.plot(x,[item*100 for item in y2[::-1]],'r*-', label='R@10', linewidth=3, markersize=8, zorder=3)
        ax.plot(x,[item*100 for item in y3[::-1]],'gd-', label='N@10', linewidth=3, markersize=8, zorder=3)
    # ax.plot(x,y2,'g+-',label='w/o KG')

#         ax.bar(np.asarray(x), y1, width=width, color=['#00008B', 'C0', '#8F2109', '#'], alpha=0.5, label='1')
#         ax.bar(np.asarray(x), y2, width=width, color=['#00008B', 'C0', '#8F2109', '#54504F'], alpha=0.5, label='2')

    #ax.bar(np.asarray(x) -width/2*3, y1, width=width, color='#00008B', alpha=0.5, label='DisMult')
    #ax.bar(np.asarray(x)-width/2, y2, width=width, color='C0', alpha=0.5, label='TranR(IA)')
    #ax.bar(np.asarray(x)+width/2, y3, width=width, color='#8F2109', alpha=0.5, label='TranR(CB)')
    #ax.bar(np.asarray(x)+width/2*3, y4, width=width, color='#54504F', alpha=0.5, label='CrbiaSR')

#     ax.bar(np.asarray(x) -width/2*3, y1, width=width, color='#bdc9e1', label='DisMult')
#     ax.bar(np.asarray(x)-width/2, y2, width=width, color='#74a9cf', label='TranR(IA)')
#     ax.bar(np.asarray(x)+width/2, y3, width=width, color='#2b8cbe', label='TranR(CB)')
#     ax.bar(np.asarray(x)+width/2*3, y4, width=width, color='#045a8d', label='CrbiaSR')
    # ax.bar(np.asarray(x)+width, y2, width=width, color='C0', alpha=0.5, label='1')
    # ax.bar(np.asarray(x)-width, y3, width=width, color='C0', alpha=0.5, label='1')
    # rect3 = ax.bar(np.asarray(x)+width, y3, width=width, color='C2', alpha=0.5, label='3')

    # legend 字体风格的大小设置
    #plt.legend(prop={'family': 'Times New Roman', 'size': 16}, loc='lower left').set_zorder(1)

    if tmp_index == 3 or tmp_index==5:
        plt.legend(prop={'family': 'Times New Roman', 'size': 16}, loc='center left').set_zorder(1)
    else:
        plt.legend(prop={'family': 'Times New Roman', 'size': 16}, loc='lower left').set_zorder(1)
    # if index == 0:
#     if tmp_index ==0:
#         ax.legend(prop={'family': 'Times New Roman', 'size': 10}, loc = 'upper left')

    # title 字体风格的大小设置
    # plt.title('Comparison', fontdict={'family': 'Times New Roman', 'size': 20})
    ax.set_title(title_name, fontdict={'family': 'Times New Roman', 'size': 30})

    # x/y 轴标签的设置
    # plt.xlabel('X', fontdict={'family': 'Times New Roman', 'size': 10})
    # plt.ylabel('Y', fontdict={'family': 'Times New Roman', 'size': 25})
    #ax.set_xlabel(dataset, fontdict={'family': 'Times New Roman', 'size': 15}, labelpad=2.0)
    ax.set_xlabel(dataset, fontdict={'family': 'Times New Roman', 'size': 25}, labelpad=5.0)

    ax.set_ylabel(metrics, fontdict={'family': 'Times New Roman', 'size': 25})

    # 设置y轴刻度间距
    ax.yaxis.set_major_locator(MultipleLocator(1.5))

    # 设置y轴数值显示风格
    # 例如：显示小数点后两位 --> %.2f
    # 整数显示 --> %d
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # 设置x/y轴刻度的显示
    # plt.xticks(x, ['x1', 'x2', 'x3', 'x4', 'x5'], fontproperties = 'Times New Roman', size = 20)
    # plt.yticks(fontproperties = 'Times New Roman', size = 20)

    # ax.grid(ls=":",c='b',)
    # plt.vlines(0, 0, 0.5, colors = "r", linestyles = "dashed")

    # 设置x轴的显示
    ax.set_xticks(x)
    # ax.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
#         ax.set_xticklabels(['DisMult', 'TransR(IA)', 'TransR(CB)',  'CbiaSR'])
    #ax.set_xticklabels(['1', '2', '4', '5'])
    if tmp_index == 4 or tmp_index ==5:
        ax.set_xticklabels(labels)
    else:
        ax.set_xticklabels(labels[::-1])
    # 设置刻度的字体大小
    ax.tick_params(labelsize=20)
    # 设置x/y轴的字体风格
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # 不显示x轴的标签以及刻度
    # ax.axes.get_xaxis().set_visible(False)

    # 不显示上面和右边的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 显示柱状上面的数字
    # auto_text(rect1)
    # auto_text(rect2)

    ax.yaxis.grid(True, zorder=0)

#         for a, b in zip(x, y1):
#             ax.text(a+0.3, b+0.002, b, ha='center', va='bottom', fontsize=10)

    # for a, b in zip(x, y2):
    #     ax.text(a+0.3, b+0.012, b, ha='center', va='bottom', fontsize=10)
    # auto_text(rect3)

    # 转置x轴
    # ax.invert_xaxis()
    # 转置y轴
    # ax.invert_yaxis()

    # 设置y轴的范围a
    # plt.ylim(0, 6)
#     ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.set_ylim(x_min, x_max)
    tmp_index += 1
# %matplotlib inline
#plt.savefig('graph_depth.pdf')
plt.savefig('{}_{}.pdf'.format(dataset, metric), bbox_inches='tight')
plt.show()
