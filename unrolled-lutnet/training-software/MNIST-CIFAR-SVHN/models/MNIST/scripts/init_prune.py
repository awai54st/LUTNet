import h5py
import numpy as np

from shutil import copyfile

copyfile("baseline_reg.h5", "pretrained_pruned.h5") # create pretrained.h5 using datastructure from dummy.h5

bl = h5py.File("baseline_reg.h5", 'r')
#dummy = h5py.File("dummy.h5", 'r')
pretrained = h5py.File("pretrained_pruned.h5", 'r+')

normalisation="l2"

p_d1=-1
p_d2=0.6
p_d3=0.6
p_d4=0.6
p_d5=0.6

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

# set up pruning_mask
if normalisation=="l1":
	norm=abs(np.array(bl_w1))
elif normalisation=="l2":
	norm=np.array(bl_w1)**2
	norm=np.sqrt(norm)

#l1_norm=np.reshape(l1_norm, [-1,np.shape(l1_norm)[3]])
pruning_mask = np.greater(norm, p_d1)
pret_pruning_mask[...] = np.array(pruning_mask,dtype=float)
print(np.sum(np.array(pret_pruning_mask)))

# dense layer 2

bl_w1 = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]

pret_w1[...] = np.array(bl_w1)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)

# set up pruning_mask
if normalisation=="l1":
	norm=abs(np.array(bl_w1))
elif normalisation=="l2":
	norm=np.array(bl_w1)**2
	norm=np.sqrt(norm)

pruning_mask = np.greater(norm, p_d2)
pret_pruning_mask[...] = np.array(pruning_mask,dtype=float)
print(np.sum(np.array(pret_pruning_mask)))

# dense layer 3

bl_w1 = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_2"]["residual_sign_2"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_2"]["residual_sign_2"]["means:0"]

pret_w1[...] = np.array(bl_w1)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)

# set up pruning_mask
if normalisation=="l1":
	norm=abs(np.array(bl_w1))
elif normalisation=="l2":
	norm=np.array(bl_w1)**2
	norm=np.sqrt(norm)

pruning_mask = np.greater(norm, p_d3)
pret_pruning_mask[...] = np.array(pruning_mask,dtype=float)
print(np.sum(np.array(pret_pruning_mask)))

# dense layer 4

bl_w1 = bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_1:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_3"]["residual_sign_3"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_1:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_3"]["residual_sign_3"]["means:0"]

pret_w1[...] = np.array(bl_w1)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)

# set up pruning_mask
if normalisation=="l1":
	norm=abs(np.array(bl_w1))
elif normalisation=="l2":
	norm=np.array(bl_w1)**2
	norm=np.sqrt(norm)

pruning_mask = np.greater(norm, p_d4)
pret_pruning_mask[...] = np.array(pruning_mask,dtype=float)
print(np.sum(np.array(pret_pruning_mask)))

# dense layer 5

bl_w1 = bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_1:0"]
bl_pruning_mask = bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["pruning_mask:0"]
bl_gamma = bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable:0"]
bl_means = bl["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]
zero_fill = np.zeros(np.shape(np.array(bl_w1)))
pret_w1 = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_1:0"]
pret_pruning_mask = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["pruning_mask:0"]
p_gamma = pretrained["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable:0"]
pret_means = pretrained["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]

pret_w1[...] = np.array(bl_w1)
p_gamma[...] = np.array(bl_gamma)
pret_means[...] = np.array(bl_means)

# set up pruning_mask
if normalisation=="l1":
	norm=abs(np.array(bl_w1))
elif normalisation=="l2":
	norm=np.array(bl_w1)**2
	norm=np.sqrt(norm)

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
