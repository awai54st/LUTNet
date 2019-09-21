import h5py
import numpy as np

from shutil import copyfile

copyfile("dummy_lutnet.h5", "pretrained_bin.h5") # create pretrained.h5 using datastructure from dummy.h5

bl = h5py.File("baseline_pruned.h5", 'r')
#dummy = h5py.File("dummy.h5", 'r')
pretrained = h5py.File("pretrained_bin.h5", 'r+')

# conv layer 1

bl_w1 = bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["Variable_1:0"]
#bl_rand_map = bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["Variable:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_conv_1"]["binary_conv_1"]["Variable_1:0"]
#pret_rand_map = pretrained["model_weights"]["binary_conv_1"]["binary_conv_1"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_conv_1"]["binary_conv_1"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_conv_1"]["binary_conv_1"]["Variable:0"]

pret_w1[...] = np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# conv layer 2

bl_w1 = bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_1:0"]
#bl_w2 = bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_2:0"]
#bl_w3 = bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_3:0"]
#bl_w4 = bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_4:0"]
#bl_rand_map = bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_1:0"]
#pret_rand_map = pretrained["model_weights"]["binary_conv_2"]["binary_conv_2"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_conv_2"]["binary_conv_2"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]

#weight_shape = np.shape(bl_w1)
#
pret_w1[...] = np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# conv layer 3

bl_w1 = bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_1:0"]
#bl_rand_map = bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_2"]["residual_sign_2"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_1:0"]
#pret_rand_map = pretrained["model_weights"]["binary_conv_3"]["binary_conv_3"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_conv_3"]["binary_conv_3"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_2"]["residual_sign_2"]["means:0"]

pret_w1[...] = np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# conv layer 4

bl_w1 = bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_1:0"]
#bl_rand_map = bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_3"]["residual_sign_3"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_1:0"]
#pret_rand_map = pretrained["model_weights"]["binary_conv_4"]["binary_conv_4"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_conv_4"]["binary_conv_4"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_3"]["residual_sign_3"]["means:0"]

pret_w1[...] = np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# conv layer 5

bl_w1 = bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_1:0"]
#bl_rand_map = bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_1:0"]
#pret_rand_map = pretrained["model_weights"]["binary_conv_5"]["binary_conv_5"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_conv_5"]["binary_conv_5"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]

pret_w1[...] = np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# conv layer 6

bl_w1 = bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_1:0"]
#bl_w2 = bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_2:0"]
#bl_w3 = bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_3:0"]
#bl_w4 = bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_4:0"]
bl_rand_map_0 = bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_0:0"]
bl_rand_map_1 = bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_1:0"]
bl_rand_map_2 = bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_2:0"]
bl_pruning_mask = bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_5"]["residual_sign_5"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_1:0"]
pret_w2 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_2:0"]
pret_w3 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_3:0"]
pret_w4 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_4:0"]
pret_w5 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_5:0"]
pret_w6 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_6:0"]
pret_w7 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_7:0"]
pret_w8 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_8:0"]
pret_w9 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_9:0"]
pret_w10 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_10:0"]
pret_w11 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_11:0"]
pret_w12 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_12:0"]
pret_w13 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_13:0"]
pret_w14 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_14:0"]
pret_w15 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_15:0"]
pret_w16 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_16:0"]
pret_w17 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_17:0"]
pret_w18 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_18:0"]
pret_w19 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_19:0"]
pret_w20 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_20:0"]
pret_w21 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_21:0"]
pret_w22 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_22:0"]
pret_w23 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_23:0"]
pret_w24 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_24:0"]
pret_w25 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_25:0"]
pret_w26 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_26:0"]
pret_w27 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_27:0"]
pret_w28 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_28:0"]
pret_w29 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_29:0"]
pret_w30 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_30:0"]
pret_w31 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_31:0"]
pret_w32 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_32:0"]
pret_rand_map_0 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_0:0"]
pret_rand_map_1 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_1:0"]
pret_rand_map_2 = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_2:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_5"]["residual_sign_5"]["means:0"]

weight_shape = np.shape(bl_w1)

# randomisation and pruning recovery
bl_w1_unroll = np.reshape(np.array(bl_w1), (-1,weight_shape[3]))
bl_w1 = np.array(bl_w1)

rand_map_0 = np.arange(weight_shape[0]*weight_shape[1]*weight_shape[2])
np.random.shuffle(rand_map_0)
rand_map_1 = np.arange(weight_shape[0]*weight_shape[1]*weight_shape[2])
np.random.shuffle(rand_map_1)
rand_map_2 = np.arange(weight_shape[0]*weight_shape[1]*weight_shape[2])
np.random.shuffle(rand_map_2)

pruning_mask = np.array(bl_pruning_mask).astype(bool)

# weights for extra input 0

init_mask = np.logical_not(pruning_mask[rand_map_0])
pruning_mask_recover = np.logical_and(pruning_mask, init_mask)[np.argsort(rand_map_0)]
pruning_mask = np.logical_or(pruning_mask, pruning_mask_recover)
init_mask = np.reshape(init_mask, weight_shape)

bl_w1_rand = bl_w1_unroll[rand_map_0]
bl_w1_rand = np.reshape(bl_w1_rand, weight_shape)

w1 = bl_w1
w2 = bl_w1
w3 = bl_w1
w4 = bl_w1
w5 = bl_w1
w6 = bl_w1
w7 = bl_w1
w8 = bl_w1
w9 = -bl_w1
w10 = -bl_w1
w11 = -bl_w1
w12 = -bl_w1
w13 = -bl_w1
w14 = -bl_w1
w15 = -bl_w1
w16 = -bl_w1
w17 = bl_w1
w18 = bl_w1
w19 = bl_w1
w20 = bl_w1
w21 = bl_w1
w22 = bl_w1
w23 = bl_w1
w24 = bl_w1
w25 = -bl_w1
w26 = -bl_w1
w27 = -bl_w1
w28 = -bl_w1
w29 = -bl_w1
w30 = -bl_w1
w31 = -bl_w1
w32 = -bl_w1

w1[init_mask] = w1[init_mask] + bl_w1_rand[init_mask]
w2[init_mask] = w2[init_mask] + bl_w1_rand[init_mask]
w3[init_mask] = w3[init_mask] + bl_w1_rand[init_mask]
w4[init_mask] = w4[init_mask] + bl_w1_rand[init_mask]
w5[init_mask] = w5[init_mask] - bl_w1_rand[init_mask]
w6[init_mask] = w6[init_mask] - bl_w1_rand[init_mask]
w7[init_mask] = w7[init_mask] - bl_w1_rand[init_mask]
w8[init_mask] = w8[init_mask] - bl_w1_rand[init_mask]
w9[init_mask] = w9[init_mask] + bl_w1_rand[init_mask]
w10[init_mask] = w10[init_mask] + bl_w1_rand[init_mask]
w11[init_mask] = w11[init_mask] + bl_w1_rand[init_mask]
w12[init_mask] = w12[init_mask] + bl_w1_rand[init_mask]
w13[init_mask] = w13[init_mask] - bl_w1_rand[init_mask]
w14[init_mask] = w14[init_mask] - bl_w1_rand[init_mask]
w15[init_mask] = w15[init_mask] - bl_w1_rand[init_mask]
w16[init_mask] = w16[init_mask] - bl_w1_rand[init_mask]
w17[init_mask] = w17[init_mask] + bl_w1_rand[init_mask]
w18[init_mask] = w18[init_mask] + bl_w1_rand[init_mask]
w19[init_mask] = w19[init_mask] + bl_w1_rand[init_mask]
w20[init_mask] = w20[init_mask] + bl_w1_rand[init_mask]
w21[init_mask] = w21[init_mask] - bl_w1_rand[init_mask]
w22[init_mask] = w22[init_mask] - bl_w1_rand[init_mask]
w23[init_mask] = w23[init_mask] - bl_w1_rand[init_mask]
w24[init_mask] = w24[init_mask] - bl_w1_rand[init_mask]
w25[init_mask] = w25[init_mask] + bl_w1_rand[init_mask]
w26[init_mask] = w26[init_mask] + bl_w1_rand[init_mask]
w27[init_mask] = w27[init_mask] + bl_w1_rand[init_mask]
w28[init_mask] = w28[init_mask] + bl_w1_rand[init_mask]
w29[init_mask] = w29[init_mask] - bl_w1_rand[init_mask]
w30[init_mask] = w30[init_mask] - bl_w1_rand[init_mask]
w31[init_mask] = w31[init_mask] - bl_w1_rand[init_mask]
w32[init_mask] = w32[init_mask] - bl_w1_rand[init_mask]

# weights for extra input 2

init_mask = np.logical_not(pruning_mask[rand_map_1])
pruning_mask_recover = np.logical_and(pruning_mask, init_mask)[np.argsort(rand_map_1)]
pruning_mask = np.logical_or(pruning_mask, pruning_mask_recover)
init_mask = np.reshape(init_mask, weight_shape)

bl_w1_rand = bl_w1_unroll[rand_map_1]
bl_w1_rand = np.reshape(bl_w1_rand, weight_shape)

w1[init_mask] = w1[init_mask] + bl_w1_rand[init_mask]
w2[init_mask] = w2[init_mask] + bl_w1_rand[init_mask]
w3[init_mask] = w3[init_mask] - bl_w1_rand[init_mask]
w4[init_mask] = w4[init_mask] - bl_w1_rand[init_mask]
w5[init_mask] = w5[init_mask] + bl_w1_rand[init_mask]
w6[init_mask] = w6[init_mask] + bl_w1_rand[init_mask]
w7[init_mask] = w7[init_mask] - bl_w1_rand[init_mask]
w8[init_mask] = w8[init_mask] - bl_w1_rand[init_mask]
w9[init_mask] = w9[init_mask] + bl_w1_rand[init_mask]
w10[init_mask] = w10[init_mask] + bl_w1_rand[init_mask]
w11[init_mask] = w11[init_mask] - bl_w1_rand[init_mask]
w12[init_mask] = w12[init_mask] - bl_w1_rand[init_mask]
w13[init_mask] = w13[init_mask] + bl_w1_rand[init_mask]
w14[init_mask] = w14[init_mask] + bl_w1_rand[init_mask]
w15[init_mask] = w15[init_mask] - bl_w1_rand[init_mask]
w16[init_mask] = w16[init_mask] - bl_w1_rand[init_mask]
w17[init_mask] = w17[init_mask] + bl_w1_rand[init_mask]
w18[init_mask] = w18[init_mask] + bl_w1_rand[init_mask]
w19[init_mask] = w19[init_mask] - bl_w1_rand[init_mask]
w20[init_mask] = w20[init_mask] - bl_w1_rand[init_mask]
w21[init_mask] = w21[init_mask] + bl_w1_rand[init_mask]
w22[init_mask] = w22[init_mask] + bl_w1_rand[init_mask]
w23[init_mask] = w23[init_mask] - bl_w1_rand[init_mask]
w24[init_mask] = w24[init_mask] - bl_w1_rand[init_mask]
w25[init_mask] = w25[init_mask] + bl_w1_rand[init_mask]
w26[init_mask] = w26[init_mask] + bl_w1_rand[init_mask]
w27[init_mask] = w27[init_mask] - bl_w1_rand[init_mask]
w28[init_mask] = w28[init_mask] - bl_w1_rand[init_mask]
w29[init_mask] = w29[init_mask] + bl_w1_rand[init_mask]
w30[init_mask] = w30[init_mask] + bl_w1_rand[init_mask]
w31[init_mask] = w31[init_mask] - bl_w1_rand[init_mask]
w32[init_mask] = w32[init_mask] - bl_w1_rand[init_mask]

# weights for extra input 3

init_mask = np.logical_not(pruning_mask[rand_map_2])
pruning_mask_recover = np.logical_and(pruning_mask, init_mask)[np.argsort(rand_map_2)]
pruning_mask = np.logical_or(pruning_mask, pruning_mask_recover)
init_mask = np.reshape(init_mask, weight_shape)

bl_w1_rand = bl_w1_unroll[rand_map_2]
bl_w1_rand = np.reshape(bl_w1_rand, weight_shape)

w1[init_mask] = w1[init_mask] + bl_w1_rand[init_mask]
w2[init_mask] = w2[init_mask] - bl_w1_rand[init_mask]
w3[init_mask] = w3[init_mask] + bl_w1_rand[init_mask]
w4[init_mask] = w4[init_mask] - bl_w1_rand[init_mask]
w5[init_mask] = w5[init_mask] + bl_w1_rand[init_mask]
w6[init_mask] = w6[init_mask] - bl_w1_rand[init_mask]
w7[init_mask] = w7[init_mask] + bl_w1_rand[init_mask]
w8[init_mask] = w8[init_mask] - bl_w1_rand[init_mask]
w9[init_mask] = w9[init_mask] + bl_w1_rand[init_mask]
w10[init_mask] = w10[init_mask] - bl_w1_rand[init_mask]
w11[init_mask] = w11[init_mask] + bl_w1_rand[init_mask]
w12[init_mask] = w12[init_mask] - bl_w1_rand[init_mask]
w13[init_mask] = w13[init_mask] + bl_w1_rand[init_mask]
w14[init_mask] = w14[init_mask] - bl_w1_rand[init_mask]
w15[init_mask] = w15[init_mask] + bl_w1_rand[init_mask]
w16[init_mask] = w16[init_mask] - bl_w1_rand[init_mask]
w17[init_mask] = w17[init_mask] + bl_w1_rand[init_mask]
w18[init_mask] = w18[init_mask] - bl_w1_rand[init_mask]
w19[init_mask] = w19[init_mask] + bl_w1_rand[init_mask]
w20[init_mask] = w20[init_mask] - bl_w1_rand[init_mask]
w21[init_mask] = w21[init_mask] + bl_w1_rand[init_mask]
w22[init_mask] = w22[init_mask] - bl_w1_rand[init_mask]
w23[init_mask] = w23[init_mask] + bl_w1_rand[init_mask]
w24[init_mask] = w24[init_mask] - bl_w1_rand[init_mask]
w25[init_mask] = w25[init_mask] + bl_w1_rand[init_mask]
w26[init_mask] = w26[init_mask] - bl_w1_rand[init_mask]
w27[init_mask] = w27[init_mask] + bl_w1_rand[init_mask]
w28[init_mask] = w28[init_mask] - bl_w1_rand[init_mask]
w29[init_mask] = w29[init_mask] + bl_w1_rand[init_mask]
w30[init_mask] = w30[init_mask] - bl_w1_rand[init_mask]
w31[init_mask] = w31[init_mask] + bl_w1_rand[init_mask]
w32[init_mask] = w32[init_mask] - bl_w1_rand[init_mask]

pret_w1[...] = w1 
pret_w2[...] = w2 
pret_w3[...] = w3 
pret_w4[...] = w4 
pret_w5[...] = w5 
pret_w6[...] = w6 
pret_w7[...] = w7 
pret_w8[...] = w8 
pret_w9[...] = w9 
pret_w10[...] = w10 
pret_w11[...] = w11 
pret_w12[...] = w12 
pret_w13[...] = w13 
pret_w14[...] = w14 
pret_w15[...] = w15 
pret_w16[...] = w16 
pret_w17[...] = w17 
pret_w18[...] = w18 
pret_w19[...] = w19 
pret_w20[...] = w20 
pret_w21[...] = w21 
pret_w22[...] = w22 
pret_w23[...] = w23 
pret_w24[...] = w24 
pret_w25[...] = w25 
pret_w26[...] = w26 
pret_w27[...] = w27 
pret_w28[...] = w28 
pret_w29[...] = w29 
pret_w30[...] = w30 
pret_w31[...] = w31 
pret_w32[...] = w32 

pret_rand_map_0[...] = np.reshape(rand_map_0, (-1,1)).astype(float)
pret_rand_map_1[...] = np.reshape(rand_map_1, (-1,1)).astype(float)
pret_rand_map_2[...] = np.reshape(rand_map_2, (-1,1)).astype(float)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# dense layer 1

bl_w1 = bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_1:0"]
#bl_rand_map = bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_6"]["residual_sign_6"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_1:0"]
#pret_rand_map = pretrained["model_weights"]["binary_dense_1"]["binary_dense_1"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_1"]["binary_dense_1"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_6"]["residual_sign_6"]["means:0"]

pret_w1[...] = np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# dense layer 2

bl_w1 = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"]
#bl_rand_map = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_7"]["residual_sign_7"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"]
#pret_rand_map = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_7"]["residual_sign_7"]["means:0"]

pret_w1[...] = np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# dense layer 3

bl_w1 = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"]
#bl_rand_map = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_8"]["residual_sign_8"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"]
#pret_rand_map = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_8"]["residual_sign_8"]["means:0"]

pret_w1[...] = np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# bn 1

bl_beta = bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["beta:0"]
bl_gamma = bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["gamma:0"]
bl_moving_mean = bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["moving_mean:0"]
bl_moving_variance = bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["moving_variance:0"]
p_beta = pretrained["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["beta:0"]
p_gamma = pretrained["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["gamma:0"]
p_moving_mean = pretrained["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["moving_mean:0"]
p_moving_variance = pretrained["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["moving_variance:0"]

p_beta[...] = np.array(bl_beta)
p_gamma[...] = np.array(bl_gamma)
p_moving_mean[...] = np.array(bl_moving_mean)
p_moving_variance[...] = np.array(bl_moving_variance)

# bn 2

bl_beta = bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["beta:0"]
bl_gamma = bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["gamma:0"]
bl_moving_mean = bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["moving_mean:0"]
bl_moving_variance = bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["moving_variance:0"]
p_beta = pretrained["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["beta:0"]
p_gamma = pretrained["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["gamma:0"]
p_moving_mean = pretrained["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["moving_mean:0"]
p_moving_variance = pretrained["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["moving_variance:0"]

p_beta[...] = np.array(bl_beta)
p_gamma[...] = np.array(bl_gamma)
p_moving_mean[...] = np.array(bl_moving_mean)
p_moving_variance[...] = np.array(bl_moving_variance)

# bn 3

bl_beta = bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["beta:0"]
bl_gamma = bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["gamma:0"]
bl_moving_mean = bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["moving_mean:0"]
bl_moving_variance = bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["moving_variance:0"]
p_beta = pretrained["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["beta:0"]
p_gamma = pretrained["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["gamma:0"]
p_moving_mean = pretrained["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["moving_mean:0"]
p_moving_variance = pretrained["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["moving_variance:0"]

p_beta[...] = np.array(bl_beta)
p_gamma[...] = np.array(bl_gamma)
p_moving_mean[...] = np.array(bl_moving_mean)
p_moving_variance[...] = np.array(bl_moving_variance)

# bn 4

bl_beta = bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["beta:0"]
bl_gamma = bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["gamma:0"]
bl_moving_mean = bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["moving_mean:0"]
bl_moving_variance = bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["moving_variance:0"]
p_beta = pretrained["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["beta:0"]
p_gamma = pretrained["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["gamma:0"]
p_moving_mean = pretrained["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["moving_mean:0"]
p_moving_variance = pretrained["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["moving_variance:0"]

p_beta[...] = np.array(bl_beta)
p_gamma[...] = np.array(bl_gamma)
p_moving_mean[...] = np.array(bl_moving_mean)
p_moving_variance[...] = np.array(bl_moving_variance)

# bn 5

bl_beta = bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["beta:0"]
bl_gamma = bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["gamma:0"]
bl_moving_mean = bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_mean:0"]
bl_moving_variance = bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_variance:0"]
p_beta = pretrained["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["beta:0"]
p_gamma = pretrained["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["gamma:0"]
p_moving_mean = pretrained["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_mean:0"]
p_moving_variance = pretrained["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_variance:0"]

p_beta[...] = np.array(bl_beta)
p_gamma[...] = np.array(bl_gamma)
p_moving_mean[...] = np.array(bl_moving_mean)
p_moving_variance[...] = np.array(bl_moving_variance)

# bn 6

bl_beta = bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["beta:0"]
bl_gamma = bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["gamma:0"]
bl_moving_mean = bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["moving_mean:0"]
bl_moving_variance = bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["moving_variance:0"]
p_beta = pretrained["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["beta:0"]
p_gamma = pretrained["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["gamma:0"]
p_moving_mean = pretrained["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["moving_mean:0"]
p_moving_variance = pretrained["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["moving_variance:0"]

p_beta[...] = np.array(bl_beta)
p_gamma[...] = np.array(bl_gamma)
p_moving_mean[...] = np.array(bl_moving_mean)
p_moving_variance[...] = np.array(bl_moving_variance)

# bn 7

bl_beta = bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["beta:0"]
bl_gamma = bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["gamma:0"]
bl_moving_mean = bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["moving_mean:0"]
bl_moving_variance = bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["moving_variance:0"]
p_beta = pretrained["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["beta:0"]
p_gamma = pretrained["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["gamma:0"]
p_moving_mean = pretrained["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["moving_mean:0"]
p_moving_variance = pretrained["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["moving_variance:0"]

p_beta[...] = np.array(bl_beta)
p_gamma[...] = np.array(bl_gamma)
p_moving_mean[...] = np.array(bl_moving_mean)
p_moving_variance[...] = np.array(bl_moving_variance)

# bn 8

bl_beta = bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["beta:0"]
bl_gamma = bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["gamma:0"]
bl_moving_mean = bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_mean:0"]
bl_moving_variance = bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_variance:0"]
p_beta = pretrained["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["beta:0"]
p_gamma = pretrained["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["gamma:0"]
p_moving_mean = pretrained["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_mean:0"]
p_moving_variance = pretrained["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_variance:0"]

p_beta[...] = np.array(bl_beta)
p_gamma[...] = np.array(bl_gamma)
p_moving_mean[...] = np.array(bl_moving_mean)
p_moving_variance[...] = np.array(bl_moving_variance)

# bn 7

bl_beta = bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["beta:0"]
bl_gamma = bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["gamma:0"]
bl_moving_mean = bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["moving_mean:0"]
bl_moving_variance = bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["moving_variance:0"]
p_beta = pretrained["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["beta:0"]
p_gamma = pretrained["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["gamma:0"]
p_moving_mean = pretrained["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["moving_mean:0"]
p_moving_variance = pretrained["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["moving_variance:0"]

p_beta[...] = np.array(bl_beta)
p_gamma[...] = np.array(bl_gamma)
p_moving_mean[...] = np.array(bl_moving_mean)
p_moving_variance[...] = np.array(bl_moving_variance)




pretrained.close()
