import numpy as np

a = np.load('/home/xin/nyx/ocp-main-change/results/2023-11-29-15-49-20/is2re_predictions.npz', allow_pickle=True)

print(a.files)

# # S2EF
# ids = a['ids']
# energy = a['energy']
# forces = a['forces']
# chunk_idx = a['chunk_idx']
# print(ids)
# print(energy)
# print(forces)
# print(chunk_idx)

# IS2RE
ids = a['ids']
energy = a['energy']
print(ids)
print(energy)
#
# # IS2RS
# ids = a['ids']
# pos = a['pos']
# chunk_idx = a['chunk_idx']
# print(ids)
# print(pos)
# print(chunk_idx)

# 将结果写入dataset2.txt
with open("dataset2.txt", "a") as f:
    f.write(str(energy))
    f.write("\n")
