import h5py
import numpy as np

from shutil import copyfile

copyfile("baseline_reg.h5", "pretrained_pruned.h5") # create pretrained.h5 using datastructure from dummy.h5

bl = h5py.File("baseline_reg.h5", 'r')
#dummy = h5py.File("dummy.h5", 'r')
pretrained = h5py.File("pretrained_pruned.h5", 'r+')

normalisation="l2"

channel_threshold=0.5

p_d1=-1
p_d2=0.80
p_d3=0.80
p_d4=0.80
p_d5=-1

# dense layer 1

bl_w1 = bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_1:0"]
#bl_rand_map = bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_1:0"]
#pret_rand_map = pretrained["model_weights"]["binary_conv_1"]["binary_conv_1"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_1"]["binary_dense_1"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable:0"]

pret_w1[...] = np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)

weight = np.array(bl_w1)
TM = 8
TN = 8
Tsize_M = np.shape(weight)[0]/TM
Tsize_N = np.shape(weight)[1]/TN
one_tile = np.zeros([Tsize_M,Tsize_N])
# set up pruning_mask
#mean=np.mean(abs(weight),axis=3)
norm=one_tile
if normalisation=="l1":
        for n in range(TN):
                for m in range(TM):
                        norm = norm + weight[(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]
        norm = norm / (TRC*TRC*TM*TN)
elif normalisation=="l2":
        for n in range(TN):
                for m in range(TM):
                                norm = norm + weight[(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]**2
        norm = norm / (TM*TN)
        norm = np.sqrt(norm)

#l1_norm=np.reshape(l1_norm, [-1,np.shape(l1_norm)[3]])
pruning_mask = np.greater(norm, p_d1)
pret_pruning_mask[...] = np.array(pruning_mask,dtype=float)
print(np.sum(np.array(pret_pruning_mask)))

# dense layer 2

bl_w1 = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"]
#bl_w2 = bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_2:0"]
#bl_w3 = bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_3:0"]
#bl_w4 = bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_4:0"]
#bl_rand_map = bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"]
#pret_w2 = pretrained["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_2:0"]
#pret_w3 = pretrained["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_3:0"]
#pret_w4 = pretrained["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_4:0"]
#pret_rand_map = pretrained["model_weights"]["binary_conv_2"]["binary_conv_2"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]

pret_w1[...] = np.array(bl_w1)
#pret_w2[...] = zero_fill
#pret_w3[...] = zero_fill
#pret_w4[...] = -np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)

weight = np.array(bl_w1)
TM = 8
TN = 8
Tsize_M = np.shape(weight)[0]/TM
Tsize_N = np.shape(weight)[1]/TN
one_tile = np.zeros([Tsize_M,Tsize_N])
# set up pruning_mask
#mean=np.mean(abs(weight),axis=3)
norm=one_tile
if normalisation=="l1":
        for n in range(TN):
                for m in range(TM):
                        norm = norm + weight[(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]
        norm = norm / (TRC*TRC*TM*TN)
elif normalisation=="l2":
        for n in range(TN):
                for m in range(TM):
                                norm = norm + weight[(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]**2
        norm = norm / (TM*TN)
        norm = np.sqrt(norm)

#l1_norm=np.reshape(l1_norm, [-1,np.shape(l1_norm)[3]])
pruning_mask = np.greater(norm, p_d2)
pret_pruning_mask[...] = np.array(pruning_mask,dtype=float)
print(np.sum(np.array(pret_pruning_mask)))

# dense layer 3

bl_w1 = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"]
#bl_w2 = bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_2:0"]
#bl_w3 = bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_3:0"]
#bl_w4 = bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_4:0"]
#bl_rand_map = bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_2"]["residual_sign_2"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"]
#pret_w2 = pretrained["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_2:0"]
#pret_w3 = pretrained["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_3:0"]
#pret_w4 = pretrained["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_4:0"]
#pret_rand_map = pretrained["model_weights"]["binary_conv_3"]["binary_conv_3"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_2"]["residual_sign_2"]["means:0"]

pret_w1[...] = np.array(bl_w1)
#pret_w2[...] = zero_fill
#pret_w3[...] = zero_fill
#pret_w4[...] = -np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)

weight = np.array(bl_w1)
TM = 8
TN = 8
Tsize_M = np.shape(weight)[0]/TM
Tsize_N = np.shape(weight)[1]/TN
one_tile = np.zeros([Tsize_M,Tsize_N])
# set up pruning_mask
#mean=np.mean(abs(weight),axis=3)
norm=one_tile
if normalisation=="l1":
        for n in range(TN):
                for m in range(TM):
                        norm = norm + weight[(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]
        norm = norm / (TRC*TRC*TM*TN)
elif normalisation=="l2":
        for n in range(TN):
                for m in range(TM):
                                norm = norm + weight[(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]**2
        norm = norm / (TM*TN)
        norm = np.sqrt(norm)

#l1_norm=np.reshape(l1_norm, [-1,np.shape(l1_norm)[3]])
pruning_mask = np.greater(norm, p_d3)
pret_pruning_mask[...] = np.array(pruning_mask,dtype=float)
print(np.sum(np.array(pret_pruning_mask)))

# dense layer 4

bl_w1 = bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_1:0"]
#bl_w2 = bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_2:0"]
#bl_w3 = bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_3:0"]
#bl_w4 = bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_4:0"]
#bl_rand_map = bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_3"]["residual_sign_3"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_1:0"]
#pret_w2 = pretrained["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_2:0"]
#pret_w3 = pretrained["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_3:0"]
#pret_w4 = pretrained["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_4:0"]
#pret_rand_map = pretrained["model_weights"]["binary_conv_4"]["binary_conv_4"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_3"]["residual_sign_3"]["means:0"]

pret_w1[...] = np.array(bl_w1)
#pret_w2[...] = zero_fill
#pret_w3[...] = zero_fill
#pret_w4[...] = -np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)

weight = np.array(bl_w1)
TM = 8
TN = 8
Tsize_M = np.shape(weight)[0]/TM
Tsize_N = np.shape(weight)[1]/TN
one_tile = np.zeros([Tsize_M,Tsize_N])
# set up pruning_mask
#mean=np.mean(abs(weight),axis=3)
norm=one_tile
if normalisation=="l1":
        for n in range(TN):
                for m in range(TM):
                        norm = norm + weight[(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]
        norm = norm / (TRC*TRC*TM*TN)
elif normalisation=="l2":
        for n in range(TN):
                for m in range(TM):
                                norm = norm + weight[(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]**2
        norm = norm / (TM*TN)
        norm = np.sqrt(norm)

#l1_norm=np.reshape(l1_norm, [-1,np.shape(l1_norm)[3]])
pruning_mask = np.greater(norm, p_d4)
pret_pruning_mask[...] = np.array(pruning_mask,dtype=float)
print(np.sum(np.array(pret_pruning_mask)))

# dense layer 5

bl_w1 = bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_1:0"]
#bl_w2 = bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_2:0"]
#bl_w3 = bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_3:0"]
#bl_w4 = bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_4:0"]
#bl_rand_map = bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["rand_map:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_1:0"]
#pret_w2 = pretrained["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_2:0"]
#pret_w3 = pretrained["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_3:0"]
#pret_w4 = pretrained["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_4:0"]
#pret_rand_map = pretrained["model_weights"]["binary_conv_5"]["binary_conv_5"]["rand_map:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]

pret_w1[...] = np.array(bl_w1)
#pret_w2[...] = zero_fill
#pret_w3[...] = zero_fill
#pret_w4[...] = -np.array(bl_w1)
#pret_rand_map[...] = np.array(bl_rand_map)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)

weight = np.array(bl_w1)
TM = 8
TN = 10
Tsize_M = np.shape(weight)[0]/TM
Tsize_N = np.shape(weight)[1]/TN
one_tile = np.zeros([Tsize_M,Tsize_N])
# set up pruning_mask
#mean=np.mean(abs(weight),axis=3)
norm=one_tile
if normalisation=="l1":
        for n in range(TN):
                for m in range(TM):
                        norm = norm + weight[(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]
        norm = norm / (TRC*TRC*TM*TN)
elif normalisation=="l2":
        for n in range(TN):
                for m in range(TM):
                                norm = norm + weight[(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]**2
        norm = norm / (TM*TN)
        norm = np.sqrt(norm)

#l1_norm=np.reshape(l1_norm, [-1,np.shape(l1_norm)[3]])
pruning_mask = np.greater(norm, p_d5)
pret_pruning_mask[...] = np.array(pruning_mask,dtype=float)
print(np.sum(np.array(pret_pruning_mask)))

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
