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

    bl = h5py.File("pretrained_network_51lut_tm.h5", 'r')
    #bl = h5py.File("dummy.h5", 'r')
    
    # init model parameter lists

    batch_norm_eps=1e-4
    weights_w = []
    weights_c = []
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
    c_lut = [np.ones([3,3,1,8])]
    weights_c.extend([c_lut])

    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([np.ones([3,3,1,8])])  
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

    bl_c1  = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_1:0"])
    bl_c2  = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_2:0"])
    bl_c3  = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_3:0"])
    bl_c4  = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_4:0"])
    bl_c5  = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_5:0"])
    bl_c6  = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_6:0"])
    bl_c7  = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_7:0"])
    bl_c8  = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_8:0"])   
    bl_c9  = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_9:0"])
    bl_c10 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_10:0"])
    bl_c11 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_11:0"])
    bl_c12 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_12:0"])
    bl_c13 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_13:0"])
    bl_c14 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_14:0"])
    bl_c15 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_15:0"])
    bl_c16 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_16:0"])
    bl_c17 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_17:0"])
    bl_c18 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_18:0"])
    bl_c19 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_19:0"])
    bl_c20 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_20:0"])
    bl_c21 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_21:0"])
    bl_c22 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_22:0"])
    bl_c23 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_23:0"])
    bl_c24 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_24:0"])   
    bl_c25 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_25:0"])
    bl_c26 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_26:0"])
    bl_c27 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_27:0"])
    bl_c28 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_28:0"])
    bl_c29 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_29:0"])
    bl_c30 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_30:0"])
    bl_c31 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_31:0"])
    bl_c32 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_32:0"])
    bl_w1  = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_33:0"])
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["pruning_mask:0"]).reshape(bl_c1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_2"]["residual_sign_2"]["means:0"]
    
    w_bram = [bl_w1]
    c_lut = [bl_c1*bl_pruning_mask, bl_c2*bl_pruning_mask, bl_c3*bl_pruning_mask, bl_c4*bl_pruning_mask, bl_c5*bl_pruning_mask, bl_c6*bl_pruning_mask, bl_c7*bl_pruning_mask, bl_c8*bl_pruning_mask, bl_c9*bl_pruning_mask, bl_c10*bl_pruning_mask, bl_c11*bl_pruning_mask, bl_c12*bl_pruning_mask, bl_c13*bl_pruning_mask, bl_c14*bl_pruning_mask, bl_c15*bl_pruning_mask, bl_c16*bl_pruning_mask, bl_c17*bl_pruning_mask, bl_c18*bl_pruning_mask, bl_c19*bl_pruning_mask, bl_c20*bl_pruning_mask, bl_c21*bl_pruning_mask, bl_c22*bl_pruning_mask, bl_c23*bl_pruning_mask, bl_c24*bl_pruning_mask, bl_c25*bl_pruning_mask, bl_c26*bl_pruning_mask, bl_c27*bl_pruning_mask, bl_c28*bl_pruning_mask, bl_c29*bl_pruning_mask, bl_c30*bl_pruning_mask, bl_c31*bl_pruning_mask, bl_c32*bl_pruning_mask]
    r_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2]
    #weights = [weights, w_lut]
    weights_c.extend([c_lut])
    weights_w.extend([w_bram])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([r_map])
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


    bl_c1  = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_1:0"])
    bl_c2  = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_2:0"])
    bl_c3  = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_3:0"])
    bl_c4  = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_4:0"])
    bl_c5  = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_5:0"])
    bl_c6  = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_6:0"])
    bl_c7  = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_7:0"])
    bl_c8  = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_8:0"])   
    bl_c9  = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_9:0"])
    bl_c10 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_10:0"])
    bl_c11 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_11:0"])
    bl_c12 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_12:0"])
    bl_c13 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_13:0"])
    bl_c14 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_14:0"])
    bl_c15 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_15:0"])
    bl_c16 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_16:0"])
    bl_c17 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_17:0"])
    bl_c18 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_18:0"])
    bl_c19 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_19:0"])
    bl_c20 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_20:0"])
    bl_c21 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_21:0"])
    bl_c22 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_22:0"])
    bl_c23 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_23:0"])
    bl_c24 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_24:0"])   
    bl_c25 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_25:0"])
    bl_c26 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_26:0"])
    bl_c27 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_27:0"])
    bl_c28 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_28:0"])
    bl_c29 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_29:0"])
    bl_c30 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_30:0"])
    bl_c31 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_31:0"])
    bl_c32 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_32:0"])
    bl_w1  = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_33:0"])
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["pruning_mask:0"]).reshape(bl_c1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_3"]["residual_sign_3"]["means:0"]
    
    w_bram = [bl_w1]
    c_lut = [bl_c1*bl_pruning_mask, bl_c2*bl_pruning_mask, bl_c3*bl_pruning_mask, bl_c4*bl_pruning_mask, bl_c5*bl_pruning_mask, bl_c6*bl_pruning_mask, bl_c7*bl_pruning_mask, bl_c8*bl_pruning_mask, bl_c9*bl_pruning_mask, bl_c10*bl_pruning_mask, bl_c11*bl_pruning_mask, bl_c12*bl_pruning_mask, bl_c13*bl_pruning_mask, bl_c14*bl_pruning_mask, bl_c15*bl_pruning_mask, bl_c16*bl_pruning_mask, bl_c17*bl_pruning_mask, bl_c18*bl_pruning_mask, bl_c19*bl_pruning_mask, bl_c20*bl_pruning_mask, bl_c21*bl_pruning_mask, bl_c22*bl_pruning_mask, bl_c23*bl_pruning_mask, bl_c24*bl_pruning_mask, bl_c25*bl_pruning_mask, bl_c26*bl_pruning_mask, bl_c27*bl_pruning_mask, bl_c28*bl_pruning_mask, bl_c29*bl_pruning_mask, bl_c30*bl_pruning_mask, bl_c31*bl_pruning_mask, bl_c32*bl_pruning_mask]
    r_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2]
    #weights = [weights, w_lut]
    weights_c.extend([c_lut])
    weights_w.extend([w_bram])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([r_map])
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

    bl_c1  = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_1:0"])
    bl_c2  = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_2:0"])
    bl_c3  = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_3:0"])
    bl_c4  = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_4:0"])
    bl_c5  = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_5:0"])
    bl_c6  = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_6:0"])
    bl_c7  = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_7:0"])
    bl_c8  = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_8:0"])   
    bl_c9  = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_9:0"])
    bl_c10 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_10:0"])
    bl_c11 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_11:0"])
    bl_c12 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_12:0"])
    bl_c13 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_13:0"])
    bl_c14 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_14:0"])
    bl_c15 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_15:0"])
    bl_c16 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_16:0"])
    bl_c17 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_17:0"])
    bl_c18 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_18:0"])
    bl_c19 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_19:0"])
    bl_c20 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_20:0"])
    bl_c21 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_21:0"])
    bl_c22 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_22:0"])
    bl_c23 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_23:0"])
    bl_c24 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_24:0"])   
    bl_c25 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_25:0"])
    bl_c26 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_26:0"])
    bl_c27 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_27:0"])
    bl_c28 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_28:0"])
    bl_c29 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_29:0"])
    bl_c30 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_30:0"])
    bl_c31 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_31:0"])
    bl_c32 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_32:0"])
    bl_w1  = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_33:0"])
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["pruning_mask:0"]).reshape(bl_c1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]
    
    w_bram = [bl_w1]
    c_lut = [bl_c1*bl_pruning_mask, bl_c2*bl_pruning_mask, bl_c3*bl_pruning_mask, bl_c4*bl_pruning_mask, bl_c5*bl_pruning_mask, bl_c6*bl_pruning_mask, bl_c7*bl_pruning_mask, bl_c8*bl_pruning_mask, bl_c9*bl_pruning_mask, bl_c10*bl_pruning_mask, bl_c11*bl_pruning_mask, bl_c12*bl_pruning_mask, bl_c13*bl_pruning_mask, bl_c14*bl_pruning_mask, bl_c15*bl_pruning_mask, bl_c16*bl_pruning_mask, bl_c17*bl_pruning_mask, bl_c18*bl_pruning_mask, bl_c19*bl_pruning_mask, bl_c20*bl_pruning_mask, bl_c21*bl_pruning_mask, bl_c22*bl_pruning_mask, bl_c23*bl_pruning_mask, bl_c24*bl_pruning_mask, bl_c25*bl_pruning_mask, bl_c26*bl_pruning_mask, bl_c27*bl_pruning_mask, bl_c28*bl_pruning_mask, bl_c29*bl_pruning_mask, bl_c30*bl_pruning_mask, bl_c31*bl_pruning_mask, bl_c32*bl_pruning_mask]
    r_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2]
    #weights = [weights, w_lut]
    weights_c.extend([c_lut])
    weights_w.extend([w_bram])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([r_map])
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

    bl_c1  = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_1:0"])
    bl_c2  = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_2:0"])
    bl_c3  = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_3:0"])
    bl_c4  = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_4:0"])
    bl_c5  = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_5:0"])
    bl_c6  = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_6:0"])
    bl_c7  = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_7:0"])
    bl_c8  = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_8:0"])   
    bl_c9  = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_9:0"])
    bl_c10 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_10:0"])
    bl_c11 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_11:0"])
    bl_c12 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_12:0"])
    bl_c13 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_13:0"])
    bl_c14 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_14:0"])
    bl_c15 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_15:0"])
    bl_c16 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_16:0"])
    bl_c17 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_17:0"])
    bl_c18 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_18:0"])
    bl_c19 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_19:0"])
    bl_c20 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_20:0"])
    bl_c21 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_21:0"])
    bl_c22 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_22:0"])
    bl_c23 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_23:0"])
    bl_c24 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_24:0"])   
    bl_c25 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_25:0"])
    bl_c26 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_26:0"])
    bl_c27 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_27:0"])
    bl_c28 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_28:0"])
    bl_c29 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_29:0"])
    bl_c30 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_30:0"])
    bl_c31 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_31:0"])
    bl_c32 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_32:0"])
    bl_w1  = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_33:0"])
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["pruning_mask:0"]).reshape(bl_c1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_5"]["residual_sign_5"]["means:0"]
    
    w_bram = [bl_w1]
    c_lut = [bl_c1*bl_pruning_mask, bl_c2*bl_pruning_mask, bl_c3*bl_pruning_mask, bl_c4*bl_pruning_mask, bl_c5*bl_pruning_mask, bl_c6*bl_pruning_mask, bl_c7*bl_pruning_mask, bl_c8*bl_pruning_mask, bl_c9*bl_pruning_mask, bl_c10*bl_pruning_mask, bl_c11*bl_pruning_mask, bl_c12*bl_pruning_mask, bl_c13*bl_pruning_mask, bl_c14*bl_pruning_mask, bl_c15*bl_pruning_mask, bl_c16*bl_pruning_mask, bl_c17*bl_pruning_mask, bl_c18*bl_pruning_mask, bl_c19*bl_pruning_mask, bl_c20*bl_pruning_mask, bl_c21*bl_pruning_mask, bl_c22*bl_pruning_mask, bl_c23*bl_pruning_mask, bl_c24*bl_pruning_mask, bl_c25*bl_pruning_mask, bl_c26*bl_pruning_mask, bl_c27*bl_pruning_mask, bl_c28*bl_pruning_mask, bl_c29*bl_pruning_mask, bl_c30*bl_pruning_mask, bl_c31*bl_pruning_mask, bl_c32*bl_pruning_mask]
    r_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2]
    #weights = [weights, w_lut]
    weights_c.extend([c_lut])
    weights_w.extend([w_bram])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([r_map])
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


    bl_c1  = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_1:0"])
    bl_c2  = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_2:0"])
    bl_c3  = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_3:0"])
    bl_c4  = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_4:0"])
    bl_c5  = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_5:0"])
    bl_c6  = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_6:0"])
    bl_c7  = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_7:0"])
    bl_c8  = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_8:0"])   
    bl_c9  = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_9:0"])
    bl_c10 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_10:0"])
    bl_c11 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_11:0"])
    bl_c12 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_12:0"])
    bl_c13 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_13:0"])
    bl_c14 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_14:0"])
    bl_c15 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_15:0"])
    bl_c16 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_16:0"])
    bl_c17 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_17:0"])
    bl_c18 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_18:0"])
    bl_c19 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_19:0"])
    bl_c20 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_20:0"])
    bl_c21 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_21:0"])
    bl_c22 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_22:0"])
    bl_c23 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_23:0"])
    bl_c24 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_24:0"])   
    bl_c25 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_25:0"])
    bl_c26 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_26:0"])
    bl_c27 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_27:0"])
    bl_c28 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_28:0"])
    bl_c29 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_29:0"])
    bl_c30 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_30:0"])
    bl_c31 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_31:0"])
    bl_c32 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_32:0"])
    bl_w1  = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_33:0"])
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["pruning_mask:0"]).reshape(bl_c1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_6"]["residual_sign_6"]["means:0"]
    
    w_bram = [bl_w1]
    c_lut = [bl_c1*bl_pruning_mask, bl_c2*bl_pruning_mask, bl_c3*bl_pruning_mask, bl_c4*bl_pruning_mask, bl_c5*bl_pruning_mask, bl_c6*bl_pruning_mask, bl_c7*bl_pruning_mask, bl_c8*bl_pruning_mask, bl_c9*bl_pruning_mask, bl_c10*bl_pruning_mask, bl_c11*bl_pruning_mask, bl_c12*bl_pruning_mask, bl_c13*bl_pruning_mask, bl_c14*bl_pruning_mask, bl_c15*bl_pruning_mask, bl_c16*bl_pruning_mask, bl_c17*bl_pruning_mask, bl_c18*bl_pruning_mask, bl_c19*bl_pruning_mask, bl_c20*bl_pruning_mask, bl_c21*bl_pruning_mask, bl_c22*bl_pruning_mask, bl_c23*bl_pruning_mask, bl_c24*bl_pruning_mask, bl_c25*bl_pruning_mask, bl_c26*bl_pruning_mask, bl_c27*bl_pruning_mask, bl_c28*bl_pruning_mask, bl_c29*bl_pruning_mask, bl_c30*bl_pruning_mask, bl_c31*bl_pruning_mask, bl_c32*bl_pruning_mask]
    r_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2]
    #weights = [weights, w_lut]
    weights_c.extend([c_lut])
    weights_w.extend([w_bram])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([r_map])
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
 
    bl_c1  = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_1:0"])
    bl_c2  = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_2:0"])
    bl_c3  = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_3:0"])
    bl_c4  = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_4:0"])
    bl_c5  = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_5:0"])
    bl_c6  = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_6:0"])
    bl_c7  = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_7:0"])
    bl_c8  = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_8:0"])   
    bl_c9  = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_9:0"])
    bl_c10 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_10:0"])
    bl_c11 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_11:0"])
    bl_c12 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_12:0"])
    bl_c13 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_13:0"])
    bl_c14 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_14:0"])
    bl_c15 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_15:0"])
    bl_c16 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_16:0"])
    bl_c17 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_17:0"])
    bl_c18 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_18:0"])
    bl_c19 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_19:0"])
    bl_c20 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_20:0"])
    bl_c21 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_21:0"])
    bl_c22 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_22:0"])
    bl_c23 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_23:0"])
    bl_c24 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_24:0"])   
    bl_c25 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_25:0"])
    bl_c26 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_26:0"])
    bl_c27 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_27:0"])
    bl_c28 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_28:0"])
    bl_c29 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_29:0"])
    bl_c30 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_30:0"])
    bl_c31 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_31:0"])
    bl_c32 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_32:0"])
    bl_w1  = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_33:0"])
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["pruning_mask:0"]).reshape(bl_c1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_7"]["residual_sign_7"]["means:0"]
    
    w_bram = [bl_w1]
    c_lut = [bl_c1*bl_pruning_mask, bl_c2*bl_pruning_mask, bl_c3*bl_pruning_mask, bl_c4*bl_pruning_mask, bl_c5*bl_pruning_mask, bl_c6*bl_pruning_mask, bl_c7*bl_pruning_mask, bl_c8*bl_pruning_mask, bl_c9*bl_pruning_mask, bl_c10*bl_pruning_mask, bl_c11*bl_pruning_mask, bl_c12*bl_pruning_mask, bl_c13*bl_pruning_mask, bl_c14*bl_pruning_mask, bl_c15*bl_pruning_mask, bl_c16*bl_pruning_mask, bl_c17*bl_pruning_mask, bl_c18*bl_pruning_mask, bl_c19*bl_pruning_mask, bl_c20*bl_pruning_mask, bl_c21*bl_pruning_mask, bl_c22*bl_pruning_mask, bl_c23*bl_pruning_mask, bl_c24*bl_pruning_mask, bl_c25*bl_pruning_mask, bl_c26*bl_pruning_mask, bl_c27*bl_pruning_mask, bl_c28*bl_pruning_mask, bl_c29*bl_pruning_mask, bl_c30*bl_pruning_mask, bl_c31*bl_pruning_mask, bl_c32*bl_pruning_mask]
    r_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2]
    #weights = [weights, w_lut]
    weights_c.extend([c_lut])
    weights_w.extend([w_bram])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([r_map])
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

    bl_c1  = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"])
    bl_c2  = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_2:0"])
    bl_c3  = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_3:0"])
    bl_c4  = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_4:0"])
    bl_c5  = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_5:0"])
    bl_c6  = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_6:0"])
    bl_c7  = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_7:0"])
    bl_c8  = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_8:0"])   
    bl_c9  = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_9:0"])
    bl_c10 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_10:0"])
    bl_c11 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_11:0"])
    bl_c12 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_12:0"])
    bl_c13 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_13:0"])
    bl_c14 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_14:0"])
    bl_c15 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_15:0"])
    bl_c16 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_16:0"])
    bl_c17 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_17:0"])
    bl_c18 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_18:0"])
    bl_c19 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_19:0"])
    bl_c20 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_20:0"])
    bl_c21 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_21:0"])
    bl_c22 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_22:0"])
    bl_c23 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_23:0"])
    bl_c24 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_24:0"])   
    bl_c25 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_25:0"])
    bl_c26 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_26:0"])
    bl_c27 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_27:0"])
    bl_c28 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_28:0"])
    bl_c29 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_29:0"])
    bl_c30 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_30:0"])
    bl_c31 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_31:0"])
    bl_c32 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_32:0"])
    bl_w1  = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_33:0"])
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]).reshape(bl_c1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_8"]["residual_sign_8"]["means:0"]
    
    w_bram = [bl_w1]
    c_lut = [bl_c1*bl_pruning_mask, bl_c2*bl_pruning_mask, bl_c3*bl_pruning_mask, bl_c4*bl_pruning_mask, bl_c5*bl_pruning_mask, bl_c6*bl_pruning_mask, bl_c7*bl_pruning_mask, bl_c8*bl_pruning_mask, bl_c9*bl_pruning_mask, bl_c10*bl_pruning_mask, bl_c11*bl_pruning_mask, bl_c12*bl_pruning_mask, bl_c13*bl_pruning_mask, bl_c14*bl_pruning_mask, bl_c15*bl_pruning_mask, bl_c16*bl_pruning_mask, bl_c17*bl_pruning_mask, bl_c18*bl_pruning_mask, bl_c19*bl_pruning_mask, bl_c20*bl_pruning_mask, bl_c21*bl_pruning_mask, bl_c22*bl_pruning_mask, bl_c23*bl_pruning_mask, bl_c24*bl_pruning_mask, bl_c25*bl_pruning_mask, bl_c26*bl_pruning_mask, bl_c27*bl_pruning_mask, bl_c28*bl_pruning_mask, bl_c29*bl_pruning_mask, bl_c30*bl_pruning_mask, bl_c31*bl_pruning_mask, bl_c32*bl_pruning_mask]
    r_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2]
    #weights = [weights, w_lut]
    weights_c.extend([c_lut])
    weights_w.extend([w_bram])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([r_map])
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

    bl_c1  = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"])
    bl_c2  = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_2:0"])
    bl_c3  = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_3:0"])
    bl_c4  = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_4:0"])
    bl_c5  = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_5:0"])
    bl_c6  = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_6:0"])
    bl_c7  = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_7:0"])
    bl_c8  = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_8:0"])   
    bl_c9  = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_9:0"])
    bl_c10 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_10:0"])
    bl_c11 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_11:0"])
    bl_c12 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_12:0"])
    bl_c13 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_13:0"])
    bl_c14 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_14:0"])
    bl_c15 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_15:0"])
    bl_c16 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_16:0"])
    bl_c17 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_17:0"])
    bl_c18 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_18:0"])
    bl_c19 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_19:0"])
    bl_c20 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_20:0"])
    bl_c21 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_21:0"])
    bl_c22 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_22:0"])
    bl_c23 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_23:0"])
    bl_c24 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_24:0"])   
    bl_c25 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_25:0"])
    bl_c26 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_26:0"])
    bl_c27 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_27:0"])
    bl_c28 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_28:0"])
    bl_c29 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_29:0"])
    bl_c30 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_30:0"])
    bl_c31 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_31:0"])
    bl_c32 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_32:0"])
    bl_w1  = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_33:0"])
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]).reshape(bl_c1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_variance:0"])+batch_norm_eps)
    
    w_bram = [bl_w1]
    c_lut = [bl_c1*bl_pruning_mask, bl_c2*bl_pruning_mask, bl_c3*bl_pruning_mask, bl_c4*bl_pruning_mask, bl_c5*bl_pruning_mask, bl_c6*bl_pruning_mask, bl_c7*bl_pruning_mask, bl_c8*bl_pruning_mask, bl_c9*bl_pruning_mask, bl_c10*bl_pruning_mask, bl_c11*bl_pruning_mask, bl_c12*bl_pruning_mask, bl_c13*bl_pruning_mask, bl_c14*bl_pruning_mask, bl_c15*bl_pruning_mask, bl_c16*bl_pruning_mask, bl_c17*bl_pruning_mask, bl_c18*bl_pruning_mask, bl_c19*bl_pruning_mask, bl_c20*bl_pruning_mask, bl_c21*bl_pruning_mask, bl_c22*bl_pruning_mask, bl_c23*bl_pruning_mask, bl_c24*bl_pruning_mask, bl_c25*bl_pruning_mask, bl_c26*bl_pruning_mask, bl_c27*bl_pruning_mask, bl_c28*bl_pruning_mask, bl_c29*bl_pruning_mask, bl_c30*bl_pruning_mask, bl_c31*bl_pruning_mask, bl_c32*bl_pruning_mask]
    r_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2]
    #weights = [weights, w_lut]
    weights_c.extend([c_lut])
    weights_w.extend([w_bram])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([r_map])
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


 

    print("Binarizing the pretrained parameters...")

    # Binarize the weights
    for i in range(1,9):
        for j in range(32):
            weights_c[i][j] = SignNumpy(weights_c[i][j])# first layer has no c parameter

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
        if layer_id==0:  # first layer: fxp inputs and binary weights
            weights_c_per_act = 0
            weights_w_per_act = 1
            extra_activations = 0
        else:
            weights_c_per_act = 32 # weights_per_act = #_of_bits_per_act x 2 ^ #_of_lut_inputs
            weights_w_per_act = 1
            extra_activations = 3 # no. of extra_activations = no. of activations per LUT - 1

        dims_c = np.shape(pruning_masks[layer_id])
        dims_w = np.shape(weights_w[layer_id][0])
        if len(dims_w)==2:
            layer_type = "fc"
            word_length_w = dims_w[0]
            word_length_c = dims_c[0]
            nfilters_w = dims_w[1]
            nfilters_c = dims_c[1]
            ninch_w = dims_w[0]
            ninch_c = dims_c[0]
        elif len(dims_w)==4:
            layer_type = "conv"
            word_length_w = dims_w[0]*dims_w[1]*dims_w[2]
#            if layer_id != 0: 
            word_length_c = dims_c[0]*dims_c[1]*dims_c[2]
            nfilters_w = dims_w[3]
#            if layer_id != 0: 
            nfilters_c = dims_c[3]
            ninch_w = dims_w[2]
            ninch_c = dims_c[2]


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
                mat = np.stack(np.split(mat, nfilters_w/nfilters_c, axis=1), axis=3).transpose(1,2,3,0) # mat[M/TM][TN][TM][K*K*N/TN]
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

        if layer_id != 0: 
            # write lut parameters to header file
            for weight_id in range(weights_c_per_act):
                mat = weights_c[layer_id][weight_id]
                if layer_type=="fc":
                    mat_flat = mat.transpose(1,0).flatten()
                elif layer_type=="conv":
                    mat_flat = mat.transpose(3,0,1,2).flatten()
                else:
                    print("unknown weight format!")

                with open('../codegen_output/weights.h', 'a') as f:
                    f.write('//Array shape: {}\n'.format(dims_c))
                    fold = (word_length_c-1)/32 + 1
                    f.write("const ap_uint<32> " + "weights_c_" + layer_type + str(layer_id+1) + "_" + str(weight_id+1) + "["+str(nfilters_c*fold) + "] = {")
                    bin_append = 0
                    for i, ele in enumerate(mat_flat):
                        #bin_append = (bin_append << 1) | (int(ele) # left-first bit-push
                        bin_append = bin_append | (int(ele) << (i % word_length_c)) # right-first bit-push
                        if (i % word_length_c == (word_length_c - 1)):
                            mask = 0xFFFFFFFF
                            for i_32b in range(fold):
                                #word = (bin_append>>(32*(fold-i_32b-1))) & mask # Big-endian: left-first word-push
                                word = (bin_append>>(32*i_32b)) & mask # Little-endian: right-first word-push
                                hex_word = '%X' % word
                                if i_32b!=0:
                                    f.write(', ')
                                f.write('0x' + hex_word)
                            bin_append = 0
                            if i != nfilters_c*word_length_c-1:
                                f.write(', ')
                    f.write('};\n')

        if layer_id != 0:

            # generate verilog source file for LUTARRAY: Vivado HLS will take forever
            if layer_id==1:
                modname = 'LUTARRAY_1'
            elif layer_id==2:
                modname = 'LUTARRAY'
            elif layer_id==3:
                modname = 'LUTARRAY_2'
            elif layer_id==4:
                modname = 'LUTARRAY_5'
            elif layer_id==5:
                modname = 'LUTARRAY_6'
            elif layer_id==6:
                modname = 'LUTARRAY_3'
            elif layer_id==7:
                modname = 'LUTARRAY_4'
            elif layer_id==8:
                modname = 'LUTARRAY_7'

            if layer_id != 8: # the 8th layer has different variable names

                mat_flat = []
                for weight_id in range(weights_c_per_act):
                    mat = weights_c[layer_id][weight_id]
                    pm = pruning_masks[layer_id]#.transpose(3,0,1,2).flatten()
                    if layer_type=="fc":
                        mat = mat.transpose(1,0)
                        pm_flat = pm.transpose(1,0)
                    elif layer_type=="conv":
                        mat = mat.transpose(3,0,1,2).reshape((nfilters_c, -1))
                        pm_flat = pm.transpose(3,0,1,2).reshape((nfilters_c, -1))
                    else:
                        print("unknown weight format!")
                    mat_flat.extend([mat])
    
                with open('../codegen_output/'+modname+'.v', 'w') as v0:
                    v0.write('`timescale 1 ns / 1 ps\n\n')
                    v0.write('module '+modname+' (\n        in_V,\n        in_1_V,\n        in_2_V,\n        in_3_V')
                    for tm in range(nfilters_c):
                        v0.write(',\n        weight_0_' + str(tm) + '_V_read')
                    for tm in range(nfilters_c):
                        v0.write(',\n        ap_return_' + str(tm))
                    v0.write(');\n\n')
    
                with open('../codegen_output/'+modname+'.v', 'a') as v0:
                    v0.write('\n\n')
                    v0.write('input  [' + str(word_length_c-1) + ':0] in_V;\n')
                    v0.write('input  [' + str(word_length_c-1) + ':0] in_1_V;\n')
                    v0.write('input  [' + str(word_length_c-1) + ':0] in_2_V;\n')
                    v0.write('input  [' + str(word_length_c-1) + ':0] in_3_V;\n')
                    for tm in range(nfilters_c):
                        v0.write('input  [' + str(((word_length_c-1)/32+1)*32-1) + ':0] weight_0_' + str(tm) + '_V_read;\n')
                    for tm in range(nfilters_c):
                        v0.write('output  [' + str(word_length_c-1) + ':0] ap_return_' + str(tm) + ';\n')
                    for tm in range(nfilters_c):
                        for ti, ele in enumerate(pm_flat[tm]):
                            if ele==1:
                                v0.write('wire tmp_' + str(tm) + '_' + str(ti) + ';\n')
                                v0.write('assign tmp_' + str(tm) + '_' + str(ti) + ' = ')
                                v0.write('(' + str(int(mat_flat[0][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[1][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[2][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[3][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[4][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[5][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[6][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[7][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[8][tm][ti]))  + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[9][tm][ti]))  + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[10][tm][ti])) + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[11][tm][ti])) + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[12][tm][ti])) + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[13][tm][ti])) + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[14][tm][ti])) + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[15][tm][ti])) + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[16][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[17][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[18][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[19][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[20][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[21][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[22][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[23][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[24][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[25][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[26][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[27][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[28][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[29][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[30][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_0_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[31][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_0_' + str(tm) + '_V_read[' + str(ti) + ']);\n')
    
                        v0.write('assign ap_return_' + str(tm) + ' = {')
                        for ti, ele in enumerate(pm_flat[tm]):
                            if ele == 0:
                                v0.write("1'b0")
                            elif ele == 1:
                                v0.write('tmp_' + str(tm) + '_' + str(ti))
                            else:
                                print("pruning mask elements must be binary!")
                            if ti != word_length_c-1:
                                v0.write(',')
                            else:
                                v0.write('};\n')
                    v0.write('endmodule')


            else: # the 8th layer has different variable names

                mat_flat = []
                for weight_id in range(weights_c_per_act):
                    mat = weights_c[layer_id][weight_id]
                    pm = pruning_masks[layer_id]#.transpose(3,0,1,2).flatten()
                    if layer_type=="fc":
                        mat = mat.transpose(1,0)
                        pm_flat = pm.transpose(1,0)
                    elif layer_type=="conv":
                        mat = mat.transpose(3,0,1,2).reshape((nfilters_c, -1))
                        pm_flat = pm.transpose(3,0,1,2).reshape((nfilters_c, -1))
                    else:
                        print("unknown weight format!")
                    mat_flat.extend([mat])
    
                with open('../codegen_output/'+modname+'.v', 'w') as v0:
                    v0.write('`timescale 1 ns / 1 ps\n\n')
                    v0.write('module '+modname+' (\n        in_V,\n        in_1_V,\n        in_2_V,\n        in_3_V')
                    for tm in range(nfilters_c):
                        v0.write(',\n        weight_' + str(tm) + '_V_read')
                    v0.write(',\n        ap_return);\n\n')
    
                with open('../codegen_output/'+modname+'.v', 'a') as v0:
                    v0.write('\n\n')
                    v0.write('input  [' + str(word_length_c-1) + ':0] in_V;\n')
                    v0.write('input  [' + str(word_length_c-1) + ':0] in_1_V;\n')
                    v0.write('input  [' + str(word_length_c-1) + ':0] in_2_V;\n')
                    v0.write('input  [' + str(word_length_c-1) + ':0] in_3_V;\n')
                    for tm in range(nfilters_c):
                        v0.write('input  [' + str(((word_length_c-1)/32+1)*32-1) + ':0] weight_' + str(tm) + '_V_read;\n')
                    for tm in range(nfilters_c):
                        v0.write('output  [' + str(word_length_c-1) + ':0] ap_return;\n')
                    for tm in range(nfilters_c):
                        for ti, ele in enumerate(pm_flat[tm]):
                            if ele==1:
                                v0.write('wire tmp_' + str(tm) + '_' + str(ti) + ';\n')
                                v0.write('assign tmp_' + str(tm) + '_' + str(ti) + ' = ')
                                v0.write('(' + str(int(mat_flat[0][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[1][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[2][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[3][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[4][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[5][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[6][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[7][tm][ti]))  + ' &  in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[8][tm][ti]))  + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[9][tm][ti]))  + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[10][tm][ti])) + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[11][tm][ti])) + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[12][tm][ti])) + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[13][tm][ti])) + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[14][tm][ti])) + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[15][tm][ti])) + ' &  in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[16][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[17][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[18][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[19][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[20][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[21][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[22][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[23][tm][ti])) + ' & ~in_V[' + str(ti) + '] &  in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[24][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[25][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[26][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[27][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] &  in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[28][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[29][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] &  in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[30][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] &  weight_' + str(tm) + '_V_read[' + str(ti) + ']) | ')
                                v0.write('(' + str(int(mat_flat[31][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~weight_' + str(tm) + '_V_read[' + str(ti) + ']);\n')
    
                        v0.write('assign ap_return = {')
                        for ti, ele in enumerate(pm_flat[tm]):
                            if ele == 0:
                                v0.write("1'b0")
                            elif ele == 1:
                                v0.write('tmp_' + str(tm) + '_' + str(ti))
                            else:
                                print("pruning mask elements must be binary!")
                            if ti != word_length_c-1:
                                v0.write(',')
                            else:
                                v0.write('};\n')
                    v0.write('endmodule')


        # generate threshold
        if layer_id!=8: # the last layer does not need threshold
            use_popcount = not(layer_id==0)
            next_means_b0 = abs(means[layer_id][0])
            print(next_means_b0)
            next_means_b1 = abs(means[layer_id][1])
            print(next_means_b1)
            if layer_type=="conv":
                if layer_id != 0: 
                    fanin = np.sum(np.tile(pruning_masks[layer_id], [dims_w[0]/dims_c[0],dims_w[1]/dims_c[1],dims_w[2]/dims_c[2],dims_w[3]/dims_c[3]]).reshape(-1,dims_w[3]),axis=0)
                else:
                    fanin = np.sum(np.ones((dims_w[0]*dims_w[1]*dims_w[2],dims_w[3])),axis=0)
            elif layer_type=="fc":
                fanin = np.sum(pruning_masks[layer_id],axis=0) * (dims_w[0]*dims_w[1]/dims_c[0]/dims_c[1])
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
#        # generate next layer mean
#        if layer_id!=8:
#            with open('../codegen_output/weights.h', 'a') as f:
#                next_means_b0 = abs(means[layer_id][0])
#                next_means_b1 = abs(means[layer_id][1])
#                f.write("const ap_fixed<24, 16> " + "next_layer_means_" + layer_type + str(layer_id+1) + "[2] = {")
#                f.write(str(next_means_b0))
#                f.write(','+ str(next_means_b1))
#                f.write('};\n')

        # generate pruning mask
        if layer_id!=0:
            with open('../codegen_output/weights.h', 'a') as f:

                fold = (word_length_c-1)/32 + 1
                #f.write('//Array shape: {}\n'.format([nfilters_c,ninch_w/ninch_c,nfilters_w/nfilters_c,fold]))
                f.write("static ap_uint<32> " + "pruning_mask_" + layer_type + str(layer_id+1) + "_1["+str(nfilters_c) + "]["+str(fold) + "] = {")
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
            for rand_map_id in range(extra_activations):
                rand_map = rand_maps[layer_id][rand_map_id].flatten().astype(np.uint32)
                f.write("const unsigned int " + "rand_map_" + layer_type + str(layer_id+1) + "_" + str(rand_map_id+1) + "["+str(len(rand_map))+"] = {")
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



