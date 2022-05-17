# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import MultipleLocator, colorbar

# 全局设置字体为：Times New Roman
plt.rc('font',family='Times New Roman')

# 数据准备
x = [10, 20, 30, 40]
# y1 = [5.825, 2.938, 1.573, 0.994, 0.729]
y1 = [0.5570, 0.5994, 0.5265, 0.6048]
y2 = [0.328, 0.191, 0.154, 0.310]
# y3 = [1.297, 0.826, 0.815, 0.741, 0.698]

x_min = 0.10
x_max = 0.55
dataset = '(a) Amazon Beauty'


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
fig, ax_list = plt.subplots(ncols=2, nrows=1, figsize=(16,4),dpi=100)
plt.subplots_adjust(wspace =0.25, hspace =0)
# fig, ax = plt.subplots(figsize=(7,5),dpi=100)
# print("fig, ax:", fig, ax)

for index, ax in enumerate(ax_list):
    if index ==0:
        # y1 = [0.502, 0.330, 0.297, 0.477]
        y1 = [0.5570, 0.5994, 0.5265, 0.6204]
        y2 = [0.328, 0.191, 0.154, 0.310]
        # y3 = [1.297, 0.826, 0.815, 0.741, 0.698]

        x_min = 0.4
        x_max = 0.62
        # ax.set_ylabel('NDCG@10')
        ax.set_ylabel('HR@10')
        dataset = '(a) Amazon Beauty'
    if index == 1:
        # y1 = [0.552, 0.525, 0.546, 0.575]
        y1 = [0.5432, 0.5989, 0.5377, 0.6200]
        y2 = [0.358, 0.328, 0.347, 0.372]
        # x_min = 0.3
        # x_max = 0.62
        x_min = 0.4
        x_max = 0.62
        dataset = '(b) Amazon Sports'

    if index == 2:
        y1 = [0.5621, 0.6049, 0.5537, 0.6217]
        y2 = [0.358, 0.328, 0.347, 0.372]
        x_min = 0.4
        x_max = 0.62
        dataset = '(c) Amazon Toys'



    # ax.spines['right'].set_visible(True)
    # ax = ax_list[0]
    # print("fig, ax:", fig, ax)
    # 画柱状
    width = 6 # 柱状的宽度
    # rect1 = ax.bar(np.asarray(x)-width, y1, width=width, color='C0', alpha=0.5, label='1')
    # rect1 = ax.bar(np.asarray(x) - width/2, y1, width=width, color='#B0E0E6', alpha=0.4, label='HR@10')
    # rect2 = ax.bar(np.asarray(x) + width/2, y2, width=width, color='#FFDEAD', alpha=0.4, label='NDCG@10')

    # ax.plot(x,y1,'r^-',label='CbiaSR')
    # ax.plot(x,y2,'g+-',label='w/o KG')

    ax.bar(np.asarray(x), y1, width=width, color=['#bdc9e1', '#74a9cf', '#2b8cbe', '#045a8d'], label='1')
    # ax.bar(np.asarray(x)+width, y2, width=width, color='C0', alpha=0.5, label='1')
    # ax.bar(np.asarray(x)-width, y3, width=width, color='C0', alpha=0.5, label='1')
    # rect3 = ax.bar(np.asarray(x)+width, y3, width=width, color='C2', alpha=0.5, label='3')

    # legend 字体风格的大小设置
    # plt.legend(prop={'family': 'Times New Roman', 'size': 20})
    # if index == 0:
    # ax.legend(prop={'family': 'Times New Roman', 'size': 9}, loc = 'upper center')

    # title 字体风格的大小设置
    # plt.title('Comparison', fontdict={'family': 'Times New Roman', 'size': 20})
    # ax.set_title('Comparison', fontdict={'family': 'Times New Roman', 'size': 20})

    # x/y 轴标签的设置
    # plt.xlabel('X', fontdict={'family': 'Times New Roman', 'size': 10})
    # plt.ylabel('Y', fontdict={'family': 'Times New Roman', 'size': 25})
    ax.set_xlabel(dataset, fontdict={'family': 'Times New Roman', 'size': 12}, labelpad=5.0)

    # ax.set_ylabel('HR@10', fontdict={'family': 'Times New Roman', 'size': 18})

    # 设置y轴刻度间距
    ax.yaxis.set_major_locator(MultipleLocator(0.05))

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
    ax.set_xticklabels(['DisMult', 'TransR(IA)', 'TransR(CB)',  'CbiaSR'])
    # 设置刻度的字体大小
    ax.tick_params(labelsize=10)
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

    ax.yaxis.grid(True)

    for a, b in zip(x, y1):
        ax.text(a+0.3, b+0.002, b, ha='center', va='bottom', fontsize=10)

    # for a, b in zip(x, y2):
    #     ax.text(a+0.3, b+0.012, b, ha='center', va='bottom', fontsize=10)
    # auto_text(rect3)

    # 转置x轴
    # ax.invert_xaxis()
    # 转置y轴
    # ax.invert_yaxis()

    # 设置y轴的范围
    # plt.ylim(0, 6)
    ax.set_ylim(x_min, x_max)
# %matplotlib inline
#plt.savefig('diff_encoding.pdf')
plt.show()

