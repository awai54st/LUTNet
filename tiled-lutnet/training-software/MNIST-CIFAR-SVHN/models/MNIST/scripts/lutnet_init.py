import h5py
import numpy as np
np.set_printoptions(threshold=np.nan)

from shutil import copyfile

copyfile("dummy_lutnet.h5", "pretrained_bin.h5") # create pretrained.h5 using datastructure from dummy.h5

bl = h5py.File("baseline_pruned.h5", 'r')
#dummy = h5py.File("dummy.h5", 'r')
pretrained = h5py.File("pretrained_bin.h5", 'r+')

# dense layer 1

bl_w1 = bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_1:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_1:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_1"]["binary_dense_1"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable:0"]

pret_w1[...] = np.array(bl_w1)
p_gamma[...] = np.array(bl_gamma)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# dense layer 2

bl_w1 = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"]
bl_rand_map_0 = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_0:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]
pret_rand_map_0 = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_0:0"]
pret_rand_map_1 = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_1:0"]
pret_rand_map_2 = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_2:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]

pret_c1 =  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"]
pret_c2 =  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_2:0"]
pret_c3 =  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_3:0"]
pret_c4 =  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_4:0"]
pret_c5 =  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_5:0"]
pret_c6 =  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_6:0"]
pret_c7 =  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_7:0"]
pret_c8 =  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_8:0"]
pret_c9 =  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_9:0"]
pret_c10=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_10:0"]
pret_c11=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_11:0"]
pret_c12=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_12:0"]
pret_c13=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_13:0"]
pret_c14=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_14:0"]
pret_c15=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_15:0"]
pret_c16=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_16:0"]
pret_c17=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_17:0"]
pret_c18=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_18:0"]
pret_c19=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_19:0"]
pret_c20=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_20:0"]
pret_c21=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_21:0"]
pret_c22=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_22:0"]
pret_c23=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_23:0"]
pret_c24=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_24:0"]
pret_c25=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_25:0"]
pret_c26=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_26:0"]
pret_c27=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_27:0"]
pret_c28=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_28:0"]
pret_c29=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_29:0"]
pret_c30=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_30:0"]
pret_c31=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_31:0"]
pret_c32=  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_32:0"]
pret_w1 =  pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_33:0"]

pret_rand_map_exp_0 = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_exp_0:0"]
pret_rand_map_exp_1 = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_exp_1:0"]
pret_rand_map_exp_2 = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_exp_2:0"]

weight_shape = np.shape(bl_w1)
tile_shape = np.shape(pret_c1)
zero_fill = np.zeros(tile_shape)
one_fill = np.ones(tile_shape)
neg_one_fill = -np.ones(tile_shape)

# randomisation and pruning recovery
bl_w1_unroll = np.array(bl_w1)
bl_w1 = np.array(bl_w1)

rand_map_0 = np.arange(tile_shape[0])
np.random.shuffle(rand_map_0)
rand_map_1 = np.arange(tile_shape[0])
np.random.shuffle(rand_map_1)
rand_map_2 = np.arange(tile_shape[0])
np.random.shuffle(rand_map_2)

pruning_mask = np.array(bl_pruning_mask).astype(bool)
init_mask = np.logical_not(pruning_mask[rand_map_0])
pruning_mask_recover = np.logical_and(pruning_mask, init_mask)[np.argsort(rand_map_0)]
pruning_mask = np.logical_or(pruning_mask, pruning_mask_recover)
init_mask = np.reshape(init_mask, tile_shape)

# expand randomisation map across tiles

rand_map_0_expand = np.tile(rand_map_0,[weight_shape[0]/tile_shape[0]])
rand_map_1_expand = np.tile(rand_map_1,[weight_shape[0]/tile_shape[0]])
rand_map_2_expand = np.tile(rand_map_2,[weight_shape[0]/tile_shape[0]])
for i in range(weight_shape[0]):
        rand_map_0_expand[i] = rand_map_0_expand[i] + (tile_shape[0]*(weight_shape[0]/tile_shape[0]-1)) * (rand_map_0_expand[i]/tile_shape[0]) + tile_shape[0]*(i%weight_shape[0]/tile_shape[0])
        rand_map_1_expand[i] = rand_map_1_expand[i] + (tile_shape[0]*(weight_shape[0]/tile_shape[0]-1)) * (rand_map_1_expand[i]/tile_shape[0]) + tile_shape[0]*(i%weight_shape[0]/tile_shape[0])
        rand_map_2_expand[i] = rand_map_2_expand[i] + (tile_shape[0]*(weight_shape[0]/tile_shape[0]-1)) * (rand_map_2_expand[i]/tile_shape[0]) + tile_shape[0]*(i%weight_shape[0]/tile_shape[0])

bl_w1_rand_0 = bl_w1_unroll[rand_map_0_expand]
bl_w1_rand_0 = np.reshape(bl_w1_rand_0, weight_shape)

w1 = bl_w1

# connect1 only
c1  = one_fill
c2  = neg_one_fill
c3  = one_fill
c4  = neg_one_fill
c5  = one_fill
c6  = neg_one_fill
c7  = one_fill
c8  = neg_one_fill
c9  = one_fill
c10 = neg_one_fill
c11 = one_fill
c12 = neg_one_fill
c13 = one_fill
c14 = neg_one_fill
c15 = one_fill
c16 = neg_one_fill
c17 = neg_one_fill
c18 = one_fill
c19 = neg_one_fill
c20 = one_fill
c21 = neg_one_fill
c22 = one_fill
c23 = neg_one_fill
c24 = one_fill
c25 = neg_one_fill
c26 = one_fill
c27 = neg_one_fill
c28 = one_fill
c29 = neg_one_fill
c30 = one_fill
c31 = neg_one_fill
c32 = one_fill

pret_w1 [...] = w1
pret_c1 [...] = c1
pret_c2 [...] = c2
pret_c3 [...] = c3
pret_c4 [...] = c4
pret_c5 [...] = c5
pret_c6 [...] = c6
pret_c7 [...] = c7
pret_c8 [...] = c8
pret_c9 [...] = c9
pret_c10[...] = c10
pret_c11[...] = c11
pret_c12[...] = c12
pret_c13[...] = c13
pret_c14[...] = c14
pret_c15[...] = c15
pret_c16[...] = c16
pret_c17[...] = c17
pret_c18[...] = c18
pret_c19[...] = c19
pret_c20[...] = c20
pret_c21[...] = c21
pret_c22[...] = c22
pret_c23[...] = c23
pret_c24[...] = c24
pret_c25[...] = c25
pret_c26[...] = c26
pret_c27[...] = c27
pret_c28[...] = c28
pret_c29[...] = c29
pret_c30[...] = c30
pret_c31[...] = c31
pret_c32[...] = c32

pret_rand_map_0[...] = np.reshape(rand_map_0, (-1,1)).astype(float)
pret_rand_map_1[...] = np.reshape(rand_map_1, (-1,1)).astype(float)
pret_rand_map_2[...] = np.reshape(rand_map_2, (-1,1)).astype(float)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

rand_map_0_expand = np.reshape(rand_map_0_expand, [-1,1]).astype(float)
pret_rand_map_exp_0[...] = rand_map_0_expand
rand_map_1_expand = np.reshape(rand_map_1_expand, [-1,1]).astype(float)
pret_rand_map_exp_1[...] = rand_map_1_expand
rand_map_2_expand = np.reshape(rand_map_2_expand, [-1,1]).astype(float)
pret_rand_map_exp_2[...] = rand_map_2_expand

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# dense layer 3

bl_w1 = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"]
bl_rand_map_0 = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_0:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_2"]["residual_sign_2"]["means:0"]
pret_rand_map_0 = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_0:0"]
pret_rand_map_1 = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_1:0"]
pret_rand_map_2 = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_2:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_2"]["residual_sign_2"]["means:0"]

pret_c1 =  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"]
pret_c2 =  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_2:0"]
pret_c3 =  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_3:0"]
pret_c4 =  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_4:0"]
pret_c5 =  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_5:0"]
pret_c6 =  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_6:0"]
pret_c7 =  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_7:0"]
pret_c8 =  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_8:0"]
pret_c9 =  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_9:0"]
pret_c10=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_10:0"]
pret_c11=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_11:0"]
pret_c12=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_12:0"]
pret_c13=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_13:0"]
pret_c14=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_14:0"]
pret_c15=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_15:0"]
pret_c16=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_16:0"]
pret_c17=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_17:0"]
pret_c18=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_18:0"]
pret_c19=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_19:0"]
pret_c20=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_20:0"]
pret_c21=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_21:0"]
pret_c22=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_22:0"]
pret_c23=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_23:0"]
pret_c24=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_24:0"]
pret_c25=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_25:0"]
pret_c26=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_26:0"]
pret_c27=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_27:0"]
pret_c28=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_28:0"]
pret_c29=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_29:0"]
pret_c30=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_30:0"]
pret_c31=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_31:0"]
pret_c32=  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_32:0"]
pret_w1 =  pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_33:0"]

pret_rand_map_exp_0 = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_exp_0:0"]
pret_rand_map_exp_1 = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_exp_1:0"]
pret_rand_map_exp_2 = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_exp_2:0"]

weight_shape = np.shape(bl_w1)
tile_shape = np.shape(pret_c1)
zero_fill = np.zeros(tile_shape)
one_fill = np.ones(tile_shape)
neg_one_fill = -np.ones(tile_shape)

# randomisation and pruning recovery
bl_w1_unroll = np.array(bl_w1)
bl_w1 = np.array(bl_w1)

rand_map_0 = np.arange(tile_shape[0])
np.random.shuffle(rand_map_0)
rand_map_1 = np.arange(tile_shape[0])
np.random.shuffle(rand_map_1)
rand_map_2 = np.arange(tile_shape[0])
np.random.shuffle(rand_map_2)

pruning_mask = np.array(bl_pruning_mask).astype(bool)
init_mask = np.logical_not(pruning_mask[rand_map_0])
pruning_mask_recover = np.logical_and(pruning_mask, init_mask)[np.argsort(rand_map_0)]
pruning_mask = np.logical_or(pruning_mask, pruning_mask_recover)
init_mask = np.reshape(init_mask, tile_shape)

# expand randomisation map across tiles

rand_map_0_expand = np.tile(rand_map_0,[weight_shape[0]/tile_shape[0]])
rand_map_1_expand = np.tile(rand_map_1,[weight_shape[0]/tile_shape[0]])
rand_map_2_expand = np.tile(rand_map_2,[weight_shape[0]/tile_shape[0]])
for i in range(weight_shape[0]):
        rand_map_0_expand[i] = rand_map_0_expand[i] + (tile_shape[0]*(weight_shape[0]/tile_shape[0]-1)) * (rand_map_0_expand[i]/tile_shape[0]) + tile_shape[0]*(i%weight_shape[0]/tile_shape[0])
        rand_map_1_expand[i] = rand_map_1_expand[i] + (tile_shape[0]*(weight_shape[0]/tile_shape[0]-1)) * (rand_map_1_expand[i]/tile_shape[0]) + tile_shape[0]*(i%weight_shape[0]/tile_shape[0])
        rand_map_2_expand[i] = rand_map_2_expand[i] + (tile_shape[0]*(weight_shape[0]/tile_shape[0]-1)) * (rand_map_2_expand[i]/tile_shape[0]) + tile_shape[0]*(i%weight_shape[0]/tile_shape[0])

bl_w1_rand_0 = bl_w1_unroll[rand_map_0_expand]
bl_w1_rand_0 = np.reshape(bl_w1_rand_0, weight_shape)

w1 = bl_w1

# connect1 only
c1  = one_fill
c2  = neg_one_fill
c3  = one_fill
c4  = neg_one_fill
c5  = one_fill
c6  = neg_one_fill
c7  = one_fill
c8  = neg_one_fill
c9  = one_fill
c10 = neg_one_fill
c11 = one_fill
c12 = neg_one_fill
c13 = one_fill
c14 = neg_one_fill
c15 = one_fill
c16 = neg_one_fill
c17 = neg_one_fill
c18 = one_fill
c19 = neg_one_fill
c20 = one_fill
c21 = neg_one_fill
c22 = one_fill
c23 = neg_one_fill
c24 = one_fill
c25 = neg_one_fill
c26 = one_fill
c27 = neg_one_fill
c28 = one_fill
c29 = neg_one_fill
c30 = one_fill
c31 = neg_one_fill
c32 = one_fill

pret_w1 [...] = w1
pret_c1 [...] = c1
pret_c2 [...] = c2
pret_c3 [...] = c3
pret_c4 [...] = c4
pret_c5 [...] = c5
pret_c6 [...] = c6
pret_c7 [...] = c7
pret_c8 [...] = c8
pret_c9 [...] = c9
pret_c10[...] = c10
pret_c11[...] = c11
pret_c12[...] = c12
pret_c13[...] = c13
pret_c14[...] = c14
pret_c15[...] = c15
pret_c16[...] = c16
pret_c17[...] = c17
pret_c18[...] = c18
pret_c19[...] = c19
pret_c20[...] = c20
pret_c21[...] = c21
pret_c22[...] = c22
pret_c23[...] = c23
pret_c24[...] = c24
pret_c25[...] = c25
pret_c26[...] = c26
pret_c27[...] = c27
pret_c28[...] = c28
pret_c29[...] = c29
pret_c30[...] = c30
pret_c31[...] = c31
pret_c32[...] = c32

pret_rand_map_0[...] = np.reshape(rand_map_0, (-1,1)).astype(float)
pret_rand_map_1[...] = np.reshape(rand_map_1, (-1,1)).astype(float)
pret_rand_map_2[...] = np.reshape(rand_map_2, (-1,1)).astype(float)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

rand_map_0_expand = np.reshape(rand_map_0_expand, [-1,1]).astype(float)
pret_rand_map_exp_0[...] = rand_map_0_expand
rand_map_1_expand = np.reshape(rand_map_1_expand, [-1,1]).astype(float)
pret_rand_map_exp_1[...] = rand_map_1_expand
rand_map_2_expand = np.reshape(rand_map_2_expand, [-1,1]).astype(float)
pret_rand_map_exp_2[...] = rand_map_2_expand

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# dense layer 4

bl_w1 = bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_1:0"]
bl_rand_map_0 = bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["rand_map_0:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_3"]["residual_sign_3"]["means:0"]
pret_rand_map_0 = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["rand_map_0:0"]
pret_rand_map_1 = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["rand_map_1:0"]
pret_rand_map_2 = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["rand_map_2:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_3"]["residual_sign_3"]["means:0"]

pret_c1 =  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_1:0"]
pret_c2 =  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_2:0"]
pret_c3 =  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_3:0"]
pret_c4 =  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_4:0"]
pret_c5 =  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_5:0"]
pret_c6 =  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_6:0"]
pret_c7 =  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_7:0"]
pret_c8 =  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_8:0"]
pret_c9 =  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_9:0"]
pret_c10=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_10:0"]
pret_c11=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_11:0"]
pret_c12=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_12:0"]
pret_c13=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_13:0"]
pret_c14=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_14:0"]
pret_c15=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_15:0"]
pret_c16=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_16:0"]
pret_c17=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_17:0"]
pret_c18=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_18:0"]
pret_c19=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_19:0"]
pret_c20=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_20:0"]
pret_c21=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_21:0"]
pret_c22=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_22:0"]
pret_c23=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_23:0"]
pret_c24=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_24:0"]
pret_c25=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_25:0"]
pret_c26=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_26:0"]
pret_c27=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_27:0"]
pret_c28=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_28:0"]
pret_c29=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_29:0"]
pret_c30=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_30:0"]
pret_c31=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_31:0"]
pret_c32=  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_32:0"]
pret_w1 =  pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_33:0"]

pret_rand_map_exp_0 = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["rand_map_exp_0:0"]
pret_rand_map_exp_1 = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["rand_map_exp_1:0"]
pret_rand_map_exp_2 = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["rand_map_exp_2:0"]

weight_shape = np.shape(bl_w1)
tile_shape = np.shape(pret_c1)
zero_fill = np.zeros(tile_shape)
one_fill = np.ones(tile_shape)
neg_one_fill = -np.ones(tile_shape)

# randomisation and pruning recovery
bl_w1_unroll = np.array(bl_w1)
bl_w1 = np.array(bl_w1)

rand_map_0 = np.arange(tile_shape[0])
np.random.shuffle(rand_map_0)
rand_map_1 = np.arange(tile_shape[0])
np.random.shuffle(rand_map_1)
rand_map_2 = np.arange(tile_shape[0])
np.random.shuffle(rand_map_2)

pruning_mask = np.array(bl_pruning_mask).astype(bool)
init_mask = np.logical_not(pruning_mask[rand_map_0])
pruning_mask_recover = np.logical_and(pruning_mask, init_mask)[np.argsort(rand_map_0)]
pruning_mask = np.logical_or(pruning_mask, pruning_mask_recover)
init_mask = np.reshape(init_mask, tile_shape)

# expand randomisation map across tiles

rand_map_0_expand = np.tile(rand_map_0,[weight_shape[0]/tile_shape[0]])
rand_map_1_expand = np.tile(rand_map_1,[weight_shape[0]/tile_shape[0]])
rand_map_2_expand = np.tile(rand_map_2,[weight_shape[0]/tile_shape[0]])
for i in range(weight_shape[0]):
        rand_map_0_expand[i] = rand_map_0_expand[i] + (tile_shape[0]*(weight_shape[0]/tile_shape[0]-1)) * (rand_map_0_expand[i]/tile_shape[0]) + tile_shape[0]*(i%weight_shape[0]/tile_shape[0])
        rand_map_1_expand[i] = rand_map_1_expand[i] + (tile_shape[0]*(weight_shape[0]/tile_shape[0]-1)) * (rand_map_1_expand[i]/tile_shape[0]) + tile_shape[0]*(i%weight_shape[0]/tile_shape[0])
        rand_map_2_expand[i] = rand_map_2_expand[i] + (tile_shape[0]*(weight_shape[0]/tile_shape[0]-1)) * (rand_map_2_expand[i]/tile_shape[0]) + tile_shape[0]*(i%weight_shape[0]/tile_shape[0])

bl_w1_rand_0 = bl_w1_unroll[rand_map_0_expand]
bl_w1_rand_0 = np.reshape(bl_w1_rand_0, weight_shape)

w1 = bl_w1

# connect1 only
c1  = one_fill
c2  = neg_one_fill
c3  = one_fill
c4  = neg_one_fill
c5  = one_fill
c6  = neg_one_fill
c7  = one_fill
c8  = neg_one_fill
c9  = one_fill
c10 = neg_one_fill
c11 = one_fill
c12 = neg_one_fill
c13 = one_fill
c14 = neg_one_fill
c15 = one_fill
c16 = neg_one_fill
c17 = neg_one_fill
c18 = one_fill
c19 = neg_one_fill
c20 = one_fill
c21 = neg_one_fill
c22 = one_fill
c23 = neg_one_fill
c24 = one_fill
c25 = neg_one_fill
c26 = one_fill
c27 = neg_one_fill
c28 = one_fill
c29 = neg_one_fill
c30 = one_fill
c31 = neg_one_fill
c32 = one_fill

pret_w1 [...] = w1
pret_c1 [...] = c1
pret_c2 [...] = c2
pret_c3 [...] = c3
pret_c4 [...] = c4
pret_c5 [...] = c5
pret_c6 [...] = c6
pret_c7 [...] = c7
pret_c8 [...] = c8
pret_c9 [...] = c9
pret_c10[...] = c10
pret_c11[...] = c11
pret_c12[...] = c12
pret_c13[...] = c13
pret_c14[...] = c14
pret_c15[...] = c15
pret_c16[...] = c16
pret_c17[...] = c17
pret_c18[...] = c18
pret_c19[...] = c19
pret_c20[...] = c20
pret_c21[...] = c21
pret_c22[...] = c22
pret_c23[...] = c23
pret_c24[...] = c24
pret_c25[...] = c25
pret_c26[...] = c26
pret_c27[...] = c27
pret_c28[...] = c28
pret_c29[...] = c29
pret_c30[...] = c30
pret_c31[...] = c31
pret_c32[...] = c32

pret_rand_map_0[...] = np.reshape(rand_map_0, (-1,1)).astype(float)
pret_rand_map_1[...] = np.reshape(rand_map_1, (-1,1)).astype(float)
pret_rand_map_2[...] = np.reshape(rand_map_2, (-1,1)).astype(float)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

rand_map_0_expand = np.reshape(rand_map_0_expand, [-1,1]).astype(float)
pret_rand_map_exp_0[...] = rand_map_0_expand
rand_map_1_expand = np.reshape(rand_map_1_expand, [-1,1]).astype(float)
pret_rand_map_exp_1[...] = rand_map_1_expand
rand_map_2_expand = np.reshape(rand_map_2_expand, [-1,1]).astype(float)
pret_rand_map_exp_2[...] = rand_map_2_expand

print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

# dense layer 5

bl_w1 = bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_1:0"]
bl_rand_map_0 = bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["rand_map_0:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]
pret_rand_map_0 = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["rand_map_0:0"]
pret_rand_map_1 = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["rand_map_1:0"]
pret_rand_map_2 = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["rand_map_2:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]

pret_c1 =  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_1:0"]
pret_c2 =  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_2:0"]
pret_c3 =  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_3:0"]
pret_c4 =  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_4:0"]
pret_c5 =  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_5:0"]
pret_c6 =  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_6:0"]
pret_c7 =  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_7:0"]
pret_c8 =  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_8:0"]
pret_c9 =  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_9:0"]
pret_c10=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_10:0"]
pret_c11=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_11:0"]
pret_c12=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_12:0"]
pret_c13=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_13:0"]
pret_c14=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_14:0"]
pret_c15=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_15:0"]
pret_c16=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_16:0"]
pret_c17=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_17:0"]
pret_c18=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_18:0"]
pret_c19=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_19:0"]
pret_c20=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_20:0"]
pret_c21=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_21:0"]
pret_c22=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_22:0"]
pret_c23=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_23:0"]
pret_c24=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_24:0"]
pret_c25=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_25:0"]
pret_c26=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_26:0"]
pret_c27=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_27:0"]
pret_c28=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_28:0"]
pret_c29=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_29:0"]
pret_c30=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_30:0"]
pret_c31=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_31:0"]
pret_c32=  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_32:0"]
pret_w1 =  pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_33:0"]

pret_rand_map_exp_0 = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["rand_map_exp_0:0"]
pret_rand_map_exp_1 = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["rand_map_exp_1:0"]
pret_rand_map_exp_2 = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["rand_map_exp_2:0"]

weight_shape = np.shape(bl_w1)
tile_shape = np.shape(pret_c1)
zero_fill = np.zeros(tile_shape)
one_fill = np.ones(tile_shape)
neg_one_fill = -np.ones(tile_shape)

# randomisation and pruning recovery
bl_w1_unroll = np.array(bl_w1)
bl_w1 = np.array(bl_w1)

rand_map_0 = np.arange(tile_shape[0])
np.random.shuffle(rand_map_0)
rand_map_1 = np.arange(tile_shape[0])
np.random.shuffle(rand_map_1)
rand_map_2 = np.arange(tile_shape[0])
np.random.shuffle(rand_map_2)

pruning_mask = np.array(bl_pruning_mask).astype(bool)
init_mask = np.logical_not(pruning_mask[rand_map_0])
pruning_mask_recover = np.logical_and(pruning_mask, init_mask)[np.argsort(rand_map_0)]
pruning_mask = np.logical_or(pruning_mask, pruning_mask_recover)
init_mask = np.reshape(init_mask, tile_shape)

# expand randomisation map across tiles

rand_map_0_expand = np.tile(rand_map_0,[weight_shape[0]/tile_shape[0]])
rand_map_1_expand = np.tile(rand_map_1,[weight_shape[0]/tile_shape[0]])
rand_map_2_expand = np.tile(rand_map_2,[weight_shape[0]/tile_shape[0]])
for i in range(weight_shape[0]):
        rand_map_0_expand[i] = rand_map_0_expand[i] + (tile_shape[0]*(weight_shape[0]/tile_shape[0]-1)) * (rand_map_0_expand[i]/tile_shape[0]) + tile_shape[0]*(i%weight_shape[0]/tile_shape[0])
        rand_map_1_expand[i] = rand_map_1_expand[i] + (tile_shape[0]*(weight_shape[0]/tile_shape[0]-1)) * (rand_map_1_expand[i]/tile_shape[0]) + tile_shape[0]*(i%weight_shape[0]/tile_shape[0])
        rand_map_2_expand[i] = rand_map_2_expand[i] + (tile_shape[0]*(weight_shape[0]/tile_shape[0]-1)) * (rand_map_2_expand[i]/tile_shape[0]) + tile_shape[0]*(i%weight_shape[0]/tile_shape[0])

bl_w1_rand_0 = bl_w1_unroll[rand_map_0_expand]
bl_w1_rand_0 = np.reshape(bl_w1_rand_0, weight_shape)

w1 = bl_w1

# connect1 only
c1  = one_fill
c2  = neg_one_fill
c3  = one_fill
c4  = neg_one_fill
c5  = one_fill
c6  = neg_one_fill
c7  = one_fill
c8  = neg_one_fill
c9  = one_fill
c10 = neg_one_fill
c11 = one_fill
c12 = neg_one_fill
c13 = one_fill
c14 = neg_one_fill
c15 = one_fill
c16 = neg_one_fill
c17 = neg_one_fill
c18 = one_fill
c19 = neg_one_fill
c20 = one_fill
c21 = neg_one_fill
c22 = one_fill
c23 = neg_one_fill
c24 = one_fill
c25 = neg_one_fill
c26 = one_fill
c27 = neg_one_fill
c28 = one_fill
c29 = neg_one_fill
c30 = one_fill
c31 = neg_one_fill
c32 = one_fill

pret_w1 [...] = w1
pret_c1 [...] = c1
pret_c2 [...] = c2
pret_c3 [...] = c3
pret_c4 [...] = c4
pret_c5 [...] = c5
pret_c6 [...] = c6
pret_c7 [...] = c7
pret_c8 [...] = c8
pret_c9 [...] = c9
pret_c10[...] = c10
pret_c11[...] = c11
pret_c12[...] = c12
pret_c13[...] = c13
pret_c14[...] = c14
pret_c15[...] = c15
pret_c16[...] = c16
pret_c17[...] = c17
pret_c18[...] = c18
pret_c19[...] = c19
pret_c20[...] = c20
pret_c21[...] = c21
pret_c22[...] = c22
pret_c23[...] = c23
pret_c24[...] = c24
pret_c25[...] = c25
pret_c26[...] = c26
pret_c27[...] = c27
pret_c28[...] = c28
pret_c29[...] = c29
pret_c30[...] = c30
pret_c31[...] = c31
pret_c32[...] = c32

pret_rand_map_0[...] = np.reshape(rand_map_0, (-1,1)).astype(float)
pret_rand_map_1[...] = np.reshape(rand_map_1, (-1,1)).astype(float)
pret_rand_map_2[...] = np.reshape(rand_map_2, (-1,1)).astype(float)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)
pret_pruning_mask[...] = np.array(bl_pruning_mask)

rand_map_0_expand = np.reshape(rand_map_0_expand, [-1,1]).astype(float)
pret_rand_map_exp_0[...] = rand_map_0_expand
rand_map_1_expand = np.reshape(rand_map_1_expand, [-1,1]).astype(float)
pret_rand_map_exp_1[...] = rand_map_1_expand
rand_map_2_expand = np.reshape(rand_map_2_expand, [-1,1]).astype(float)
pret_rand_map_exp_2[...] = rand_map_2_expand

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

pretrained.close()
