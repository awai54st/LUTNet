import h5py
import numpy as np

def SignNumpy(x):
  return np.greater(x,0)

# convert a fully connected binarized layer plus batch normalization into 
# the simplified form (binary weight and positive threshold)
# note that the neurons are assumed to be in the columns of the weight
# matrix
def makeBNComplex(after_bn_thres, fanin, beta, gamma, mean, invstd, use_rowmajor=False, usePopCount=True):
  outs = fanin.shape[0]
  print ("Extracting FCBN complex, outs = %d" % (outs))
  # we'll fill in the binarized weights and thresholds iteratively
#  w_bin = range(ins*outs)
  thresholds = range(outs)
  for neuron in range(outs):
    # compute a preliminary threshold from the batchnorm parameters
    thres = mean[neuron] + ((after_bn_thres - beta[neuron]) / (abs(gamma[neuron]*invstd[neuron])+1e-4))
    need_flip = 0
    # ensure all neurons activate on the "positive" side, so we can use
    # greater-than-threshold activation
#    if gamma[neuron]*invstd[neuron] < 0:
#        need_flip = 1
#        thres = -thres
#    if thres > 32767:
#        thres = 32767
#    if thres < -32768:
#        thres = -32768
    # turn threshold into "number of 1s" (popcount) instead of signed sum
    if usePopCount:
        #thresholds[neuron] = int((fanin[neuron] + thres) / 2)
        thresholds[neuron] = (fanin[neuron] + thres) / 2
    else:
        thresholds[neuron] = thres
#    # binarize the synapses
#    for synapse in range(ins):
#      # note how we change from col major to row major if requested
#      dest_ind = neuron*ins+synapse if use_rowmajor else synapse*outs+neuron
#      if need_flip:
#        w_bin[dest_ind] = binarize(-weights[synapse][neuron])
#      else:
#        w_bin[dest_ind] = binarize(weights[synapse][neuron])
#  # reshape the output as desired
#  if use_rowmajor:
#    w_bin = np.asarray(w_bin).reshape((outs, ins))
#  else:
#    w_bin = np.asarray(w_bin).reshape((ins, outs))
    
#return (w_bin, thresholds)
  return thresholds


# binarize and pack convolutional layer weights into a matrix and compute
# thresholds from the conv bias and batchnorm parameters
def makeConvBNComplex(fanin, beta, gamma, mean, invstd, interleaveChannels=False, usePopCount=True):
  numOut = fanin.shape[0]
  print ("Extracting conv-BN complex, OFM=%d" % (numOut))
  # the fanin is used to ensure positive-only threshold
#  w_bin = range(numOut * numIn * k * k)
  # one threshold per output channel
  thresholds = range(numOut)
#  dest_ind = 0
  # we'll fill in the binarized weights and thresholds iteratively
  for neuron in range(numOut):
    # compute a preliminary threshold from the batchnorm parameters,
    # subtracting the conv bias from the batchnorm mean
    thres = mean[neuron] - (beta[neuron] / (gamma[neuron]*invstd[neuron]))
#    need_flip = 0
    # ensure all neurons activate on the "positive" side, so we can use
    # greater-than-threshold activation
    if gamma[neuron]*invstd[neuron] < 0:
#        need_flip = 1
        thres = -thres
    # turn threshold into "number of 1s" (popcount) instead of signed sum
    if usePopCount:
        thresholds[neuron] = int((fanin[neuron] + thres) / 2)
    else:
        thresholds[neuron] = thres
#    # go through each weight of each convolutional kernel
#    if interleaveChannels:
#      for ky in range(k):
#        for kx in range(k):
#          for ifm in range(numIn):
#            f = -1 if need_flip else +1
#            w_bin[dest_ind] = binarize(f*weights[neuron][ifm][ky][kx])
#            dest_ind += 1
#    else:
#      for ifm in range(numIn):
#        for ky in range(k):
#          for kx in range(k):
#            f = -1 if need_flip else +1
#            w_bin[dest_ind] = binarize(f*weights[neuron][ifm][ky][kx])
#            dest_ind += 1
#          
#  # reshape the output as desired
#  w_bin = np.asarray(w_bin).reshape((numOut, fanin))
#  return (w_bin, thresholds)
  return thresholds

if __name__ == "__main__":

    print("Loading the pretrained parameters...")

    bl = h5py.File("pretrained_network_reb_tm.h5", 'r')
    #bl = h5py.File("dummy.h5", 'r')
    
    # init model parameter lists

    batch_norm_eps=1e-4
    weights_w = []
    gammas = []
    means = []
    pruning_masks = []
    rand_maps = []
    bn_betas = []
    bn_gammas = []
    bn_means = []
    bn_inv_stds = []
    TN = [3,8,8,8,8,8,8,8,8] # hand-coded tiling factors for all layers
    TM = [8,8,8,8,8,8,8,8,10]
    
    # conv layer 1
    
    bl_w1 = np.array(bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["Variable_1:0"])
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["rand_map_0:0"])
#    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]

    ##Pruning
    #bl_w1 = bl_w1 * np.reshape(bl_pruning_mask, (bl_w1.shape))
 
    w_bram = [bl_w1]
    weights_w.extend([w_bram])

    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([0])  
    #gammas = [gammas, bl_gamma]
    gammas=[bl_gamma]
    #pruning_masks = [pruning_masks, bl_pruning_mask]
#    pruning_masks=[bl_pruning_mask]
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps=[bl_rand_map_0]
    #means = [means, bl_means]
    means=[bl_means]
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas=[bl_bn_beta]
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas=[bl_bn_gamma]
    #bn_means = [bn_means, bl_bn_mean]
    bn_means=[bl_bn_mean]
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds=[bl_bn_inv_std]
    
    # conv layer 2
 
    bl_w1  = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_1:0"])
  
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["pruning_mask:0"]).reshape(bl_w1.shape[0],bl_w1.shape[1],bl_w1.shape[2]/TN[1],bl_w1.shape[3]/TM[1])
    bl_gamma = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_2"]["residual_sign_2"]["means:0"]
 
    w_bram = [bl_w1]
    #weights = [weights, w_lut]
    weights_w.extend([w_bram])
   
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map_0])
    #means = [means, bl_means]
    means.extend([bl_means])
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])
   
    
    # conv layer 3
    bl_w1  = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_1:0"])
   
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["pruning_mask:0"]).reshape(bl_w1.shape[0],bl_w1.shape[1],bl_w1.shape[2]/TN[2],bl_w1.shape[3]/TM[2])
    bl_gamma = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_3"]["residual_sign_3"]["means:0"]
 
    w_bram = [bl_w1]
    #weights = [weights, w_lut]
    weights_w.extend([w_bram])
   
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map_0])
    #means = [means, bl_means]
    means.extend([bl_means])
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])
   
    
    # conv layer 4

    bl_w1  = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_1:0"])
   
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["pruning_mask:0"]).reshape(bl_w1.shape[0],bl_w1.shape[1],bl_w1.shape[2]/TN[3],bl_w1.shape[3]/TM[3])
    bl_gamma = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]
 
    w_bram = [bl_w1]
    #weights = [weights, w_lut]
    weights_w.extend([w_bram])
   
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map_0])
    #means = [means, bl_means]
    means.extend([bl_means])
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])
   
     
    # conv layer 5
    bl_w1  = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_1:0"])
   
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["pruning_mask:0"]).reshape(bl_w1.shape[0],bl_w1.shape[1],bl_w1.shape[2]/TN[4],bl_w1.shape[3]/TM[4])
    bl_gamma = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_5"]["residual_sign_5"]["means:0"]
 
    w_bram = [bl_w1]
    #weights = [weights, w_lut]
    weights_w.extend([w_bram])
   
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map_0])
    #means = [means, bl_means]
    means.extend([bl_means])
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])

    
    # conv layer 6
    bl_w1  = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_1:0"])
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["pruning_mask:0"]).reshape(bl_w1.shape[0],bl_w1.shape[1],bl_w1.shape[2]/TN[5],bl_w1.shape[3]/TM[5])
    bl_gamma = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_6"]["residual_sign_6"]["means:0"]
    
    w_bram = [bl_w1]
    #weights = [weights, w_lut]
    weights_w.extend([w_bram])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map_0])
    #means = [means, bl_means]
    means.extend([bl_means])
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])

   
    # dense layer 1
  
    bl_w1  = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_1:0"])
   
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["pruning_mask:0"]).reshape(bl_w1.shape[0]/TN[6],bl_w1.shape[1]/TM[6])
    bl_gamma = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_7"]["residual_sign_7"]["means:0"]
 
    w_bram = [bl_w1]
    #weights = [weights, w_lut]
    weights_w.extend([w_bram])
   
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map_0])
    #means = [means, bl_means]
    means.extend([bl_means])
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])


  
    # dense layer 2
 
    bl_w1  = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"])
   
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]).reshape(bl_w1.shape[0]/TN[7],bl_w1.shape[1]/TM[7])
    bl_gamma = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_8"]["residual_sign_8"]["means:0"]
 
    w_bram = [bl_w1]
    #weights = [weights, w_lut]
    weights_w.extend([w_bram])
   
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map_0])
    #means = [means, bl_means]
    means.extend([bl_means])
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])


    # dense layer 3
 
    bl_w1  = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"])
   
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]).reshape(bl_w1.shape[0]/TN[8],bl_w1.shape[1]/TM[8])
    bl_gamma = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["moving_variance:0"])+batch_norm_eps)
    
    #bl_means = bl["model_weights"]["residual_sign_9"]["residual_sign_9"]["means:0"]
 
    w_bram = [bl_w1]
    #weights = [weights, w_lut]
    weights_w.extend([w_bram])
   
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map_0])
    #means = [means, bl_means]
    #means.extend(bl_means)
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])


    print("Binarizing the pretrained parameters...")

    # Binarize the weights
    for i in range(0,9):
        for j in range(1):
            weights_w[i][j] = SignNumpy(weights_w[i][j])

    # write header file
    with open('../codegen_output/weights.h', 'w') as f:
        f.write('#pragma once\n')
    with open('../codegen_output/weights.h', 'a') as f:
        f.write('//Generated weights for CIFAR-10\n')

    for layer_id in range(9):
        # generate weights
        weights_w_per_act = 1

        dims_w = np.shape(weights_w[layer_id][0])
        if len(dims_w)==2:
            layer_type = "fc"
            word_length_w = dims_w[0]
            word_length_c = dims_w[0]/TN[layer_id]
            nfilters_w = dims_w[1]
            nfilters_c = dims_w[1]/TM[layer_id]
            ninch_w = dims_w[0]
            ninch_c = dims_w[0]/TN[layer_id]
        elif len(dims_w)==4:
            layer_type = "conv"
            word_length_w = dims_w[0]*dims_w[1]*dims_w[2]
#            if layer_id != 0: 
            word_length_c = dims_w[0]*dims_w[1]*dims_w[2]/TN[layer_id]
            nfilters_w = dims_w[3]
#            if layer_id != 0: 
            nfilters_c = dims_w[3]/TM[layer_id]
            ninch_w = dims_w[2]
            ninch_c = dims_w[2]/TN[layer_id]


#        # write weights to header file
#        for weight_id in range(weights_w_per_act):
#            mat = weights_w[layer_id][weight_id]
#            if layer_type=="fc":
#                mat_flat = mat.transpose(1,0).flatten()
#            elif layer_type=="conv":
#                mat_flat = mat.transpose(3,0,1,2).flatten()
#            else:
#                print("unknown weight format!")
#
#            with open('../codegen_output/weights.h', 'a') as f:
#                f.write('//Array shape: {}\n'.format(dims_w))
#                fold = (word_length_w-1)/32 + 1
#                f.write("const ap_uint<32> " + "weights_w_" + layer_type + str(layer_id+1) + "_" + str(weight_id+1) + "["+str(nfilters_w*fold) + "] = {")
#                bin_append = 0
#                for i, ele in enumerate(mat_flat):
#                    #bin_append = (bin_append << 1) | (int(ele) # left-first bit-push
#                    bin_append = bin_append | (int(ele) << (i % word_length_w)) # right-first bit-push
#                    if (i % word_length_w == (word_length_w - 1)):
#                        mask = 0xFFFFFFFF
#                        for i_32b in range(fold):
#                            #word = (bin_append>>(32*(fold-i_32b-1))) & mask # Big-endian: left-first word-push
#                            word = (bin_append>>(32*i_32b)) & mask # Little-endian: right-first word-push
#                            hex_word = '%X' % word
#                            if i_32b!=0:
#                                f.write(', ')
#                            f.write('0x' + hex_word)
#                        bin_append = 0
#                        if i != nfilters_w*word_length_w-1:
#                            f.write(', ')
#                f.write('};\n')

        # write weights to header file
        for weight_id in range(weights_w_per_act):
            mat = weights_w[layer_id][weight_id]
            if layer_type=="conv":
                mat = np.stack(np.split(mat.reshape(-1,ninch_w,nfilters_w), ninch_w/ninch_c, axis=1), axis=3)
                mat = np.stack(np.split(mat.reshape(-1,nfilters_w,ninch_w/ninch_c), nfilters_w/nfilters_c, axis=1), axis=3).transpose(1,2,3,0) # mat[M/TM][TN][TM][K*K*N/TN]
                if layer_id!=0:
                    pruning_mask = pruning_masks[layer_id].transpose(3,0,1,2).reshape(nfilters_c,-1)# pruning_mask[M/TM][K*K*N/TN]
            elif layer_type=="fc":
                mat = np.stack(np.split(mat, ninch_w/ninch_c, axis=0), axis=2)
                mat = np.stack(np.split(mat, nfilters_w/nfilters_c, axis=1), axis=3).transpose(1,2,3,0) # mat[M/TM][TN][TM][N/TN]
                pruning_mask = pruning_masks[layer_id].transpose(1,0)# pruning_mask[M/TM][N/TN]
            else:
                print("unknown weight format!")

            with open('../codegen_output/weights.h', 'a') as f:
                fold = (word_length_c-1)/32 + 1
                f.write('//Array shape: {}\n'.format([nfilters_c,ninch_w/ninch_c,nfilters_w/nfilters_c,fold]))
                f.write("static ap_uint<32> " + "weights_w_" + layer_type + str(layer_id+1) + "_" + str(weight_id+1) + "["+str(nfilters_c) + "]["+str(ninch_w/ninch_c) + "]["+str(nfilters_w/nfilters_c) + "]["+str(fold) + "] = {")
                for t1 in range(nfilters_c):
                    if t1!=0:
                        f.write(",")
                    f.write("{")
                    for t2 in range(ninch_w/ninch_c):
                        if t2!=0:
                            f.write(",")
                        f.write("{")
                        for t3 in range(nfilters_w/nfilters_c):
                            if t3!=0:
                                f.write(",")
                            f.write("{")
                            bin_append = 0
                            for i, ele in enumerate(mat[t1][t2][t3]):
                                #bin_append = (bin_append << 1) | (int(ele) # left-first bit-push
                                bin_append = bin_append | (int(ele) << (i % word_length_c)) # right-first bit-push
                                if (i == word_length_c-1):
                                    mask = 0xFFFFFFFF
                                    for i_32b in range(fold):
                                        #word = (bin_append>>(32*(fold-i_32b-1))) & mask # Big-endian: left-first word-push
                                        word = (bin_append>>(32*i_32b)) & mask # Little-endian: right-first word-push
                                        hex_word = '%X' % word
                                        if i_32b!=0:
                                            f.write(', ')
                                        f.write('0x' + hex_word)
                                    bin_append = 0
                            f.write("}")
                        f.write("}")
                    f.write("}")


                f.write('};\n')


        # generate threshold
        if layer_id!=8: # the last layer does not need threshold
            use_popcount = not(layer_id==0)
            next_means_b0 = abs(means[layer_id][0])
            print(next_means_b0)
            next_means_b1 = abs(means[layer_id][1])
            print(next_means_b1)
            if layer_type=="conv":
                if layer_id != 0: 
                    fanin = np.sum(np.tile(pruning_masks[layer_id], [1,1,TN[layer_id],TM[layer_id]]).reshape(-1,dims_w[3]),axis=0)
                else:
                    fanin = np.sum(np.ones((dims_w[0]*dims_w[1]*dims_w[2],dims_w[3])),axis=0)
            elif layer_type=="fc":
                fanin = np.sum(pruning_masks[layer_id],axis=0) * (TN[layer_id]*TM[layer_id])
            if layer_id!=0:
                fanin = fanin * abs(gammas[layer_id] * means[layer_id-1][0]) + fanin * abs(gammas[layer_id] * means[layer_id-1][1])
            thresholds = np.array(makeBNComplex(0, fanin, bn_betas[layer_id], bn_gammas[layer_id], bn_means[layer_id], bn_inv_stds[layer_id], usePopCount=use_popcount))
            next_means_bn_b0 = np.array(makeBNComplex(next_means_b0, fanin, bn_betas[layer_id], bn_gammas[layer_id], bn_means[layer_id], bn_inv_stds[layer_id], usePopCount=use_popcount)) - thresholds

            with open('../codegen_output/weights.h', 'a') as f:
                f.write("const ap_fixed<24, 16> " + "thresh_" + layer_type + str(layer_id+1) + "["+str(len(thresholds))+"] = {")
                for i, ele in enumerate(thresholds):
                    if i == 0:
                        f.write(str(ele))
                    else:
                        f.write(','+ str(ele))
                f.write('};\n')
                f.write("const ap_fixed<24, 16> " + "next_layer_means_" + layer_type + str(layer_id+1) + "["+str(len(next_means_bn_b0))+"] = {")
                for i, ele in enumerate(next_means_bn_b0):
                    if i == 0:
                        f.write(str(ele))
                    else:
                        f.write(','+ str(ele))
                f.write('};\n')
        # generate pruning mask
        if layer_id!=0:
            with open('../codegen_output/weights.h', 'a') as f:

                fold = (word_length_c-1)/32 + 1
                #f.write('//Array shape: {}\n'.format([nfilters_c,ninch_w/ninch_c,nfilters_w/nfilters_c,fold]))
                f.write("static ap_uint<32> " + "pruning_mask_" + layer_type + str(layer_id+1) + "_" + str(weight_id+1) + "["+str(nfilters_c) + "]["+str(fold) + "] = {")
                for t1 in range(nfilters_c):
                    if t1!=0:
                        f.write(",")
                    f.write("{")
                    bin_append = 0
                    for i, ele in enumerate(pruning_mask[t1]):
                        #bin_append = (bin_append << 1) | (int(ele) # left-first bit-push
                        bin_append = bin_append | (int(ele) << (i % word_length_c)) # right-first bit-push
                        if (i == word_length_c-1):
                            mask = 0xFFFFFFFF
                            for i_32b in range(fold):
                                #word = (bin_append>>(32*(fold-i_32b-1))) & mask # Big-endian: left-first word-push
                                word = (bin_append>>(32*i_32b)) & mask # Little-endian: right-first word-push
                                hex_word = '%X' % word
                                if i_32b!=0:
                                    f.write(', ')
                                f.write('0x' + hex_word)
                            bin_append = 0
                    f.write("}")


                f.write('};\n')


        # generate random map
        with open('../codegen_output/weights.h', 'a') as f:
            rand_map = rand_maps[layer_id].flatten().astype(np.uint32)
            f.write("const unsigned int " + "rand_map_" + layer_type + str(layer_id+1) + "["+str(len(rand_map))+"] = {")
            for i, ele in enumerate(rand_map):
                if i == 0:
                    f.write(str(ele))
                else:
                    f.write(','+ str(ele))
            f.write('};\n')
        # generate alpha
        with open('../codegen_output/weights.h', 'a') as f:
            if layer_id!=0:
                alpha_b0 = abs(gammas[layer_id] * means[layer_id-1][0])
                alpha_b1 = abs(gammas[layer_id] * means[layer_id-1][1])
                f.write("const ap_fixed<24, 16> " + "alpha_" + layer_type + str(layer_id+1) + "[2] = {")
                f.write(str(alpha_b0))
                f.write(','+ str(alpha_b1))
                f.write('};\n')

            else:
                alpha_b0 = abs(gammas[layer_id])
                f.write("const ap_fixed<24, 16> " + "alpha_" + layer_type + str(layer_id+1) + "[1] = {")
                f.write(str(alpha_b0))
                f.write('};\n')




