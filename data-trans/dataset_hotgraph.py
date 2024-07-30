from ocpmodels.datasets import OC22LmdbDataset
import ase.io
import numpy as np
from ase.io.extxyz import write_extxyz
from ase.io.cif import write_cif
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import os
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
from matplotlib.colors import ListedColormap

# 将催化系统的元素组成提取出来
dataset = OC22LmdbDataset({"src": "F://data/train-26745/train-18721/data.lmdb"})
l = len(dataset)

# symbols
chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

data_hot = []
for i in range(0, l):
    data = dataset[i]
    # Atom参数
    numbers = data.atomic_numbers
    s_c, s_ads = [], []
    for i in range(0, len(numbers)):
        num = int(numbers[i])
        symbol = str(chemical_symbols[num])
        # print(symbol)
        if data.tags[i]==2:
            s_ads.append(symbol)
        else:
            s_c.append(symbol)

    positions = data.pos
    cell = data.cell
    cell = cell.reshape(3, 3)    # 将1*3*3转成3*3
    pbc = [True, True, True]
    tag = data.tags

    # 将Data对象（LMDB）转换成Atom对象
    atoms_ads = Atoms(symbols=s_ads)
    # print(atoms_ads.symbols)
    # print(atoms_c.symbols)
    data_temp = []
    data_temp.append(set(s_c))
    data_temp.append(str(atoms_ads.symbols))
    data_hot.append(data_temp)
    # print(data_hot)

# 提取催化剂和吸附质的元素组成
catalyst_compositions = []
adsorbate_compositions = []

for catalyst, adsorbate in data_hot:
    catalyst_compositions.append(catalyst)
    adsorbate_compositions.append(adsorbate)
# # 将catalyst_compositions保存到txt文件
# with open('catalyst_compositions.txt', 'w') as file:
#     for composition in catalyst_compositions:
#         file.write(f"{composition}\n")



# 获取所有催化剂和吸附质的元素集合
# 使用集合去重并将集合中的元素组合在一起
all_catalyst_elements = list(set(''.join(sorted(item, key=lambda x: x[0], reverse=False)) for item in catalyst_compositions))
all_adsorbate_elements = list(set(adsorbate_compositions))

# 组合元素数据
A = list(''.join(sorted(item, key=lambda x: x[0], reverse=False)) for item in catalyst_compositions)
B = adsorbate_compositions

import random

# 创建DataFrame，记录组合元素出现的次数
data = {'catalyst': A, 'adsorbate': B}
df = pd.DataFrame(data)
df['Count'] = 1  # 默认计数为1

# 根据A和B分组，并计算每组的数量
heatmap_data = df.groupby(['catalyst', 'adsorbate']).size().reset_index(name='Count')
# 按照出现次数进行排序
heatmap_data_sorted = heatmap_data.sort_values(by='Count', ascending=False)
# # 去除相同出现次数的重复项，只保留一个
# heatmap_data_sorted_unique = heatmap_data_sorted.drop_duplicates(subset=['Count'], keep='first')
# print(heatmap_data_sorted_unique)
# # 计数为0的只保留10个
# count_0 = heatmap_data_sorted[heatmap_data_sorted['Count'] == 0].head(10)

# # 计数为1的保留50个
# count_1 = heatmap_data_sorted[heatmap_data_sorted['Count'] == 1].head(20)
#
# # 计数为2的保留50个
# count_2 = heatmap_data_sorted[heatmap_data_sorted['Count'] == 2].head(20)
# # 计数为3的保留50个
# count_3 = heatmap_data_sorted[heatmap_data_sorted['Count'] == 3].head(20)
# # 计数为4的保留50个
# count_4 = heatmap_data_sorted[heatmap_data_sorted['Count'] == 3].head(20)
# 初始化一个空的 DataFrame 用于保存结果
filtered_data = pd.DataFrame()

# 循环遍历计数为1到20的每个计数
for i in range(1, 21):
    # 筛选出计数为 i 的数据，并保留其中的前5个
    count_i = heatmap_data_sorted[heatmap_data_sorted['Count'] == i].head(10)
    # 将筛选后的数据添加到结果 DataFrame 中
    filtered_data = pd.concat([filtered_data, count_i], ignore_index=True)


# 剩余的全部保留
remaining_data = heatmap_data_sorted[(heatmap_data_sorted['Count'] > 20)]

# 合并保留的样本
selected_data = pd.concat([filtered_data, remaining_data], ignore_index=True)

# 将数据转换为透视表
heatmap_table = selected_data.pivot_table(index='adsorbate', columns='catalyst', values='Count', aggfunc='sum').fillna(0)

# # 随机取100条数据
# random_sample_data = heatmap_data.sample(n=60, random_state=42)
# selected_data = pd.concat([heatmap_data_sorted_unique, random_sample_data], ignore_index=True)
#
# # 将数据转换为透视表
# heatmap_table = selected_data.pivot_table(index='adsorbate', columns='catalyst', values='Count', aggfunc='sum').fillna(0)

# # 定义颜色映射，如果计数为0，则设置为白色
# cmap = ListedColormap(sns.color_palette("RdYlGn_r", as_cmap=True)(np.linspace(0, 1, 256)))
# cmap.set_bad(color='white')

# 使用seaborn库绘制热力图
plt.figure(figsize=(12, 8), dpi=1000)
cbar_kws = {'orientation': 'vertical'}  # 颜色条的设置

# 创建一个掩码，将统计数为0的格子掩盖起来
mask = heatmap_table == 0

# 绘制热力图，统计数为0的格子将被掩盖
sns.heatmap(heatmap_table, annot=False, cmap='RdYlGn_r', cbar_kws=cbar_kws, linewidths=.3, linecolor='grey', fmt='', mask=mask)

# 设置热力图的标题和轴标签
plt.xlabel('catalyst', fontsize=15)
plt.ylabel('adsorbate', fontsize=15)
plt.title('Heatmap', fontsize=16)
# 获取数据集路径的目录
save_path = os.path.dirname(dataset)
# 设置保存热力图的文件名
heatmap_filename = os.path.join(save_path, "heatmap.png")

# 保存热力图
plt.savefig(heatmap_filename)
plt.show()

print(f"热力图已保存到: {heatmap_filename}")
# 显示热力图
plt.show()












