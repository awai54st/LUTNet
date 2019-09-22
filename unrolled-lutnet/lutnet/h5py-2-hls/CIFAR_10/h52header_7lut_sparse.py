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

    bl = h5py.File("pretrained_network_7lut.h5", 'r')
    #bl = h5py.File("dummy.h5", 'r')
    
    # init model parameter lists

    batch_norm_eps=1e-4
    weights = []
    gammas = []
    means = []
    pruning_masks = []
    rand_maps = []
    bn_betas = []
    bn_gammas = []
    bn_means = []
    bn_inv_stds = []
    
    # conv layer 1
    
    bl_w1 = np.array(bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["Variable_1:0"])
    bl_rand_map = np.array(bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_1"]["binary_conv_1"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]

    ##Pruning
    #bl_w1 = bl_w1 * np.reshape(bl_pruning_mask, (bl_w1.shape))
    
    w_lut = [bl_w1]
    #weights = [weights, w_lut]
    weights = [w_lut]
    #gammas = [gammas, bl_gamma]
    gammas=[bl_gamma]
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks=[bl_pruning_mask]
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps=[bl_rand_map]
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
    
    bl_w1 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_1:0"])
    #bl_w2 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_2:0"])
    #bl_w3 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_3:0"])
    #bl_w4 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_4:0"])
    #bl_w5 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_5:0"])
    #bl_w6 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_6:0"])
    #bl_w7 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_7:0"])
    #bl_w8 = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable_8:0"])
    bl_rand_map = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_2"]["binary_conv_2"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_2"]["residual_sign_2"]["means:0"]
    
    #w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask, bl_w3*bl_pruning_mask, bl_w4*bl_pruning_mask, bl_w5*bl_pruning_mask, bl_w6*bl_pruning_mask, bl_w7*bl_pruning_mask, bl_w8*bl_pruning_mask]
    w_lut = [bl_w1]
    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map])
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
    
    bl_w1 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_1:0"])
    #bl_w2 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_2:0"])
    #bl_w3 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_3:0"])
    #bl_w4 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_4:0"])
    #bl_w5 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_5:0"])
    #bl_w6 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_6:0"])
    #bl_w7 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_7:0"])
    #bl_w8 = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable_8:0"])
    bl_rand_map = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_3"]["binary_conv_3"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_3"]["residual_sign_3"]["means:0"]
    
    #w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask, bl_w3*bl_pruning_mask, bl_w4*bl_pruning_mask, bl_w5*bl_pruning_mask, bl_w6*bl_pruning_mask, bl_w7*bl_pruning_mask, bl_w8*bl_pruning_mask]
    w_lut = [bl_w1]
    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map])
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
    
    bl_w1 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_1:0"])
    #bl_w2 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_2:0"])
    #bl_w3 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_3:0"])
    #bl_w4 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_4:0"])
    #bl_w5 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_5:0"])
    #bl_w6 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_6:0"])
    #bl_w7 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_7:0"])
    #bl_w8 = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable_8:0"])
    bl_rand_map = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_4"]["binary_conv_4"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]
    
    #w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask, bl_w3*bl_pruning_mask, bl_w4*bl_pruning_mask, bl_w5*bl_pruning_mask, bl_w6*bl_pruning_mask, bl_w7*bl_pruning_mask, bl_w8*bl_pruning_mask]
    w_lut = [bl_w1]
    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map])
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
    
    bl_w1 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_1:0"])
    #bl_w2 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_2:0"])
    #bl_w3 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_3:0"])
    #bl_w4 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_4:0"])
    #bl_w5 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_5:0"])
    #bl_w6 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_6:0"])
    #bl_w7 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_7:0"])
    #bl_w8 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_8:0"])
    bl_rand_map = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_5"]["residual_sign_5"]["means:0"]
    
    #w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask, bl_w3*bl_pruning_mask, bl_w4*bl_pruning_mask, bl_w5*bl_pruning_mask, bl_w6*bl_pruning_mask, bl_w7*bl_pruning_mask, bl_w8*bl_pruning_mask]
    w_lut = [bl_w1]
    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map])
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
    
    bl_w1 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_1:0"])
    bl_w2 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_2:0"])
    bl_w3 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_3:0"])
    bl_w4 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_4:0"])
    bl_w5 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_5:0"])
    bl_w6 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_6:0"])
    bl_w7 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_7:0"])
    bl_w8 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_8:0"])
    bl_w9 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_9:0"])
    bl_w10 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_10:0"])
    bl_w11 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_11:0"])
    bl_w12 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_12:0"])
    bl_w13 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_13:0"])
    bl_w14 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_14:0"])
    bl_w15 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_15:0"])
    bl_w16 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_16:0"])
    bl_w17 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_17:0"])
    bl_w18 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_18:0"])
    bl_w19 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_19:0"])
    bl_w20 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_20:0"])
    bl_w21 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_21:0"])
    bl_w22 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_22:0"])
    bl_w23 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_23:0"])
    bl_w24 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_24:0"])
    bl_w25 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_25:0"])
    bl_w26 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_26:0"])
    bl_w27 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_27:0"])
    bl_w28 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_28:0"])
    bl_w29 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_29:0"])
    bl_w30 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_30:0"])
    bl_w31 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_31:0"])
    bl_w32 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_32:0"])
    bl_w33 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_33:0"])
    bl_w34 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_34:0"])
    bl_w35 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_35:0"])
    bl_w36 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_36:0"])
    bl_w37 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_37:0"])
    bl_w38 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_38:0"])
    bl_w39 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_39:0"])
    bl_w40 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_40:0"])
    bl_w41 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_41:0"])
    bl_w42 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_42:0"])
    bl_w43 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_43:0"])
    bl_w44 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_44:0"])
    bl_w45 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_45:0"])
    bl_w46 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_46:0"])
    bl_w47 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_47:0"])
    bl_w48 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_48:0"])
    bl_w49 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_49:0"])
    bl_w50 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_50:0"])
    bl_w51 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_51:0"])
    bl_w52 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_52:0"])
    bl_w53 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_53:0"])
    bl_w54 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_54:0"])
    bl_w55 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_55:0"])
    bl_w56 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_56:0"])
    bl_w57 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_57:0"])
    bl_w58 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_58:0"])
    bl_w59 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_59:0"])
    bl_w60 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_60:0"])
    bl_w61 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_61:0"])
    bl_w62 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_62:0"])
    bl_w63 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_63:0"])
    bl_w64 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_64:0"])
    bl_w65 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_65:0"])
    bl_w66 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_66:0"])
    bl_w67 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_67:0"])
    bl_w68 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_68:0"])
    bl_w69 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_69:0"])
    bl_w70 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_70:0"])
    bl_w71 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_71:0"])
    bl_w72 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_72:0"])
    bl_w73 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_73:0"])
    bl_w74 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_74:0"])
    bl_w75 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_75:0"])
    bl_w76 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_76:0"])
    bl_w77 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_77:0"])
    bl_w78 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_78:0"])
    bl_w79 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_79:0"])
    bl_w80 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_80:0"])
    bl_w81 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_81:0"])
    bl_w82 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_82:0"])
    bl_w83 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_83:0"])
    bl_w84 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_84:0"])
    bl_w85 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_85:0"])
    bl_w86 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_86:0"])
    bl_w87 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_87:0"])
    bl_w88 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_88:0"])
    bl_w89 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_89:0"])
    bl_w90 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_90:0"])
    bl_w91 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_91:0"])
    bl_w92 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_92:0"])
    bl_w93 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_93:0"])
    bl_w94 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_94:0"])
    bl_w95 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_95:0"])
    bl_w96 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_96:0"])
    bl_w97 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_97:0"])
    bl_w98 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_98:0"])
    bl_w99 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_99:0"])
    bl_w100 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_100:0"])
    bl_w101 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_101:0"])
    bl_w102 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_102:0"])
    bl_w103 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_103:0"])
    bl_w104 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_104:0"])
    bl_w105 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_105:0"])
    bl_w106 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_106:0"])
    bl_w107 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_107:0"])
    bl_w108 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_108:0"])
    bl_w109 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_109:0"])
    bl_w110 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_110:0"])
    bl_w111 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_111:0"])
    bl_w112 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_112:0"])
    bl_w113 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_113:0"])
    bl_w114 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_114:0"])
    bl_w115 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_115:0"])
    bl_w116 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_116:0"])
    bl_w117 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_117:0"])
    bl_w118 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_118:0"])
    bl_w119 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_119:0"])
    bl_w120 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_120:0"])
    bl_w121 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_121:0"])
    bl_w122 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_122:0"])
    bl_w123 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_123:0"])
    bl_w124 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_124:0"])
    bl_w125 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_125:0"])
    bl_w126 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_126:0"])
    bl_w127 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_127:0"])
    bl_w128 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_128:0"])
    bl_w129 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_129:0"])
    bl_w130 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_130:0"])
    bl_w131 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_131:0"])
    bl_w132 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_132:0"])
    bl_w133 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_133:0"])
    bl_w134 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_134:0"])
    bl_w135 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_135:0"])
    bl_w136 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_136:0"])
    bl_w137 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_137:0"])
    bl_w138 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_138:0"])
    bl_w139 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_139:0"])
    bl_w140 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_140:0"])
    bl_w141 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_141:0"])
    bl_w142 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_142:0"])
    bl_w143 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_143:0"])
    bl_w144 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_144:0"])
    bl_w145 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_145:0"])
    bl_w146 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_146:0"])
    bl_w147 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_147:0"])
    bl_w148 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_148:0"])
    bl_w149 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_149:0"])
    bl_w150 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_150:0"])
    bl_w151 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_151:0"])
    bl_w152 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_152:0"])
    bl_w153 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_153:0"])
    bl_w154 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_154:0"])
    bl_w155 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_155:0"])
    bl_w156 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_156:0"])
    bl_w157 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_157:0"])
    bl_w158 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_158:0"])
    bl_w159 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_159:0"])
    bl_w160 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_160:0"])
    bl_w161 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_161:0"])
    bl_w162 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_162:0"])
    bl_w163 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_163:0"])
    bl_w164 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_164:0"])
    bl_w165 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_165:0"])
    bl_w166 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_166:0"])
    bl_w167 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_167:0"])
    bl_w168 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_168:0"])
    bl_w169 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_169:0"])
    bl_w170 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_170:0"])
    bl_w171 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_171:0"])
    bl_w172 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_172:0"])
    bl_w173 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_173:0"])
    bl_w174 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_174:0"])
    bl_w175 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_175:0"])
    bl_w176 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_176:0"])
    bl_w177 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_177:0"])
    bl_w178 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_178:0"])
    bl_w179 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_179:0"])
    bl_w180 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_180:0"])
    bl_w181 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_181:0"])
    bl_w182 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_182:0"])
    bl_w183 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_183:0"])
    bl_w184 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_184:0"])
    bl_w185 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_185:0"])
    bl_w186 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_186:0"])
    bl_w187 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_187:0"])
    bl_w188 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_188:0"])
    bl_w189 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_189:0"])
    bl_w190 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_190:0"])
    bl_w191 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_191:0"])
    bl_w192 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_192:0"])
    bl_w193 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_193:0"])
    bl_w194 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_194:0"])
    bl_w195 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_195:0"])
    bl_w196 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_196:0"])
    bl_w197 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_197:0"])
    bl_w198 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_198:0"])
    bl_w199 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_199:0"])
    bl_w200 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_200:0"])
    bl_w201 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_201:0"])
    bl_w202 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_202:0"])
    bl_w203 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_203:0"])
    bl_w204 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_204:0"])
    bl_w205 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_205:0"])
    bl_w206 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_206:0"])
    bl_w207 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_207:0"])
    bl_w208 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_208:0"])
    bl_w209 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_209:0"])
    bl_w210 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_210:0"])
    bl_w211 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_211:0"])
    bl_w212 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_212:0"])
    bl_w213 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_213:0"])
    bl_w214 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_214:0"])
    bl_w215 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_215:0"])
    bl_w216 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_216:0"])
    bl_w217 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_217:0"])
    bl_w218 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_218:0"])
    bl_w219 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_219:0"])
    bl_w220 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_220:0"])
    bl_w221 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_221:0"])
    bl_w222 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_222:0"])
    bl_w223 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_223:0"])
    bl_w224 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_224:0"])
    bl_w225 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_225:0"])
    bl_w226 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_226:0"])
    bl_w227 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_227:0"])
    bl_w228 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_228:0"])
    bl_w229 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_229:0"])
    bl_w230 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_230:0"])
    bl_w231 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_231:0"])
    bl_w232 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_232:0"])
    bl_w233 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_233:0"])
    bl_w234 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_234:0"])
    bl_w235 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_235:0"])
    bl_w236 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_236:0"])
    bl_w237 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_237:0"])
    bl_w238 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_238:0"])
    bl_w239 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_239:0"])
    bl_w240 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_240:0"])
    bl_w241 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_241:0"])
    bl_w242 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_242:0"])
    bl_w243 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_243:0"])
    bl_w244 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_244:0"])
    bl_w245 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_245:0"])
    bl_w246 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_246:0"])
    bl_w247 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_247:0"])
    bl_w248 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_248:0"])
    bl_w249 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_249:0"])
    bl_w250 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_250:0"])
    bl_w251 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_251:0"])
    bl_w252 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_252:0"])
    bl_w253 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_253:0"])
    bl_w254 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_254:0"])
    bl_w255 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_255:0"])
    bl_w256 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable_256:0"])
        
        
    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_2:0"])
    bl_rand_map_3 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_3:0"])
    bl_rand_map_4 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_4:0"])
    bl_rand_map_5 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_5:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_6"]["residual_sign_6"]["means:0"]

    w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask,  bl_w3*bl_pruning_mask,  bl_w4*bl_pruning_mask,  bl_w5*bl_pruning_mask,  bl_w6*bl_pruning_mask,  bl_w7*bl_pruning_mask,  bl_w8*bl_pruning_mask,  bl_w9*bl_pruning_mask,  bl_w10*bl_pruning_mask, bl_w11*bl_pruning_mask, bl_w12*bl_pruning_mask, bl_w13*bl_pruning_mask, bl_w14*bl_pruning_mask, bl_w15*bl_pruning_mask, bl_w16*bl_pruning_mask, bl_w17*bl_pruning_mask, bl_w18*bl_pruning_mask, bl_w19*bl_pruning_mask, bl_w20*bl_pruning_mask, bl_w21*bl_pruning_mask, bl_w22*bl_pruning_mask, bl_w23*bl_pruning_mask, bl_w24*bl_pruning_mask, bl_w25*bl_pruning_mask, bl_w26*bl_pruning_mask, bl_w27*bl_pruning_mask, bl_w28*bl_pruning_mask, bl_w29*bl_pruning_mask, bl_w30*bl_pruning_mask, bl_w31*bl_pruning_mask, bl_w32*bl_pruning_mask, bl_w33*bl_pruning_mask, bl_w34*bl_pruning_mask, bl_w35*bl_pruning_mask, bl_w36*bl_pruning_mask, bl_w37*bl_pruning_mask, bl_w38*bl_pruning_mask, bl_w39*bl_pruning_mask, bl_w40*bl_pruning_mask, bl_w41*bl_pruning_mask, bl_w42*bl_pruning_mask, bl_w43*bl_pruning_mask, bl_w44*bl_pruning_mask, bl_w45*bl_pruning_mask, bl_w46*bl_pruning_mask, bl_w47*bl_pruning_mask, bl_w48*bl_pruning_mask, bl_w49*bl_pruning_mask, bl_w50*bl_pruning_mask, bl_w51*bl_pruning_mask, bl_w52*bl_pruning_mask, bl_w53*bl_pruning_mask, bl_w54*bl_pruning_mask, bl_w55*bl_pruning_mask, bl_w56*bl_pruning_mask, bl_w57*bl_pruning_mask, bl_w58*bl_pruning_mask, bl_w59*bl_pruning_mask, bl_w60*bl_pruning_mask, bl_w61*bl_pruning_mask, bl_w62*bl_pruning_mask, bl_w63*bl_pruning_mask, bl_w64*bl_pruning_mask, bl_w65*bl_pruning_mask, bl_w66*bl_pruning_mask, bl_w67*bl_pruning_mask, bl_w68*bl_pruning_mask, bl_w69*bl_pruning_mask, bl_w70*bl_pruning_mask, bl_w71*bl_pruning_mask, bl_w72*bl_pruning_mask, bl_w73*bl_pruning_mask, bl_w74*bl_pruning_mask, bl_w75*bl_pruning_mask, bl_w76*bl_pruning_mask, bl_w77*bl_pruning_mask, bl_w78*bl_pruning_mask, bl_w79*bl_pruning_mask, bl_w80*bl_pruning_mask, bl_w81*bl_pruning_mask, bl_w82*bl_pruning_mask, bl_w83*bl_pruning_mask, bl_w84*bl_pruning_mask, bl_w85*bl_pruning_mask, bl_w86*bl_pruning_mask, bl_w87*bl_pruning_mask, bl_w88*bl_pruning_mask, bl_w89*bl_pruning_mask, bl_w90*bl_pruning_mask, bl_w91*bl_pruning_mask, bl_w92*bl_pruning_mask, bl_w93*bl_pruning_mask, bl_w94*bl_pruning_mask, bl_w95*bl_pruning_mask, bl_w96*bl_pruning_mask, bl_w97*bl_pruning_mask, bl_w98*bl_pruning_mask, bl_w99*bl_pruning_mask, bl_w100*bl_pruning_mask, bl_w101*bl_pruning_mask, bl_w102*bl_pruning_mask, bl_w103*bl_pruning_mask, bl_w104*bl_pruning_mask, bl_w105*bl_pruning_mask, bl_w106*bl_pruning_mask, bl_w107*bl_pruning_mask, bl_w108*bl_pruning_mask, bl_w109*bl_pruning_mask, bl_w110*bl_pruning_mask, bl_w111*bl_pruning_mask, bl_w112*bl_pruning_mask, bl_w113*bl_pruning_mask, bl_w114*bl_pruning_mask, bl_w115*bl_pruning_mask, bl_w116*bl_pruning_mask, bl_w117*bl_pruning_mask, bl_w118*bl_pruning_mask, bl_w119*bl_pruning_mask, bl_w120*bl_pruning_mask, bl_w121*bl_pruning_mask, bl_w122*bl_pruning_mask, bl_w123*bl_pruning_mask, bl_w124*bl_pruning_mask, bl_w125*bl_pruning_mask, bl_w126*bl_pruning_mask, bl_w127*bl_pruning_mask, bl_w128*bl_pruning_mask,bl_w129*bl_pruning_mask,bl_w130*bl_pruning_mask,bl_w131*bl_pruning_mask,bl_w132*bl_pruning_mask,bl_w133*bl_pruning_mask,bl_w134*bl_pruning_mask,bl_w135*bl_pruning_mask,bl_w136*bl_pruning_mask,bl_w137*bl_pruning_mask,bl_w138*bl_pruning_mask,bl_w139*bl_pruning_mask,bl_w140*bl_pruning_mask,bl_w141*bl_pruning_mask,bl_w142*bl_pruning_mask,bl_w143*bl_pruning_mask,bl_w144*bl_pruning_mask,bl_w145*bl_pruning_mask,bl_w146*bl_pruning_mask,bl_w147*bl_pruning_mask,bl_w148*bl_pruning_mask,bl_w149*bl_pruning_mask,bl_w150*bl_pruning_mask,bl_w151*bl_pruning_mask,bl_w152*bl_pruning_mask,bl_w153*bl_pruning_mask,bl_w154*bl_pruning_mask,bl_w155*bl_pruning_mask,bl_w156*bl_pruning_mask,bl_w157*bl_pruning_mask,bl_w158*bl_pruning_mask,bl_w159*bl_pruning_mask,bl_w160*bl_pruning_mask,bl_w161*bl_pruning_mask,bl_w162*bl_pruning_mask,bl_w163*bl_pruning_mask,bl_w164*bl_pruning_mask,bl_w165*bl_pruning_mask,bl_w166*bl_pruning_mask,bl_w167*bl_pruning_mask,bl_w168*bl_pruning_mask,bl_w169*bl_pruning_mask,bl_w170*bl_pruning_mask,bl_w171*bl_pruning_mask,bl_w172*bl_pruning_mask,bl_w173*bl_pruning_mask,bl_w174*bl_pruning_mask,bl_w175*bl_pruning_mask,bl_w176*bl_pruning_mask,bl_w177*bl_pruning_mask,bl_w178*bl_pruning_mask,bl_w179*bl_pruning_mask,bl_w180*bl_pruning_mask,bl_w181*bl_pruning_mask,bl_w182*bl_pruning_mask,bl_w183*bl_pruning_mask,bl_w184*bl_pruning_mask,bl_w185*bl_pruning_mask,bl_w186*bl_pruning_mask,bl_w187*bl_pruning_mask,bl_w188*bl_pruning_mask,bl_w189*bl_pruning_mask,bl_w190*bl_pruning_mask,bl_w191*bl_pruning_mask,bl_w192*bl_pruning_mask,bl_w193*bl_pruning_mask,bl_w194*bl_pruning_mask,bl_w195*bl_pruning_mask,bl_w196*bl_pruning_mask,bl_w197*bl_pruning_mask,bl_w198*bl_pruning_mask,bl_w199*bl_pruning_mask,bl_w200*bl_pruning_mask,bl_w201*bl_pruning_mask,bl_w202*bl_pruning_mask,bl_w203*bl_pruning_mask,bl_w204*bl_pruning_mask,bl_w205*bl_pruning_mask,bl_w206*bl_pruning_mask,bl_w207*bl_pruning_mask,bl_w208*bl_pruning_mask,bl_w209*bl_pruning_mask,bl_w210*bl_pruning_mask,bl_w211*bl_pruning_mask,bl_w212*bl_pruning_mask,bl_w213*bl_pruning_mask,bl_w214*bl_pruning_mask,bl_w215*bl_pruning_mask,bl_w216*bl_pruning_mask,bl_w217*bl_pruning_mask,bl_w218*bl_pruning_mask,bl_w219*bl_pruning_mask,bl_w220*bl_pruning_mask,bl_w221*bl_pruning_mask,bl_w222*bl_pruning_mask,bl_w223*bl_pruning_mask,bl_w224*bl_pruning_mask,bl_w225*bl_pruning_mask,bl_w226*bl_pruning_mask,bl_w227*bl_pruning_mask,bl_w228*bl_pruning_mask,bl_w229*bl_pruning_mask,bl_w230*bl_pruning_mask,bl_w231*bl_pruning_mask,bl_w232*bl_pruning_mask,bl_w233*bl_pruning_mask,bl_w234*bl_pruning_mask,bl_w235*bl_pruning_mask,bl_w236*bl_pruning_mask,bl_w237*bl_pruning_mask,bl_w238*bl_pruning_mask,bl_w239*bl_pruning_mask,bl_w240*bl_pruning_mask,bl_w241*bl_pruning_mask,bl_w242*bl_pruning_mask,bl_w243*bl_pruning_mask,bl_w244*bl_pruning_mask,bl_w245*bl_pruning_mask,bl_w246*bl_pruning_mask,bl_w247*bl_pruning_mask,bl_w248*bl_pruning_mask,bl_w249*bl_pruning_mask,bl_w250*bl_pruning_mask,bl_w251*bl_pruning_mask,bl_w252*bl_pruning_mask,bl_w253*bl_pruning_mask,bl_w254*bl_pruning_mask,bl_w255*bl_pruning_mask,bl_w256*bl_pruning_mask]

    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    bl_rand_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2, bl_rand_map_3, bl_rand_map_4, bl_rand_map_5]
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map])
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
    
    bl_w1 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_1:0"])
    #bl_w2 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_2:0"])
    #bl_w3 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_3:0"])
    #bl_w4 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_4:0"])
    #bl_w5 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_5:0"])
    #bl_w6 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_6:0"])
    #bl_w7 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_7:0"])
    #bl_w8 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_8:0"])
    bl_rand_map = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["rand_map:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_7"]["residual_sign_7"]["means:0"]
    
    #w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask, bl_w3*bl_pruning_mask, bl_w4*bl_pruning_mask, bl_w5*bl_pruning_mask, bl_w6*bl_pruning_mask, bl_w7*bl_pruning_mask, bl_w8*bl_pruning_mask]
    w_lut = [bl_w1]
    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map])
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
    
    bl_w1 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"])
    #bl_w2 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_2:0"])
    #bl_w3 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_3:0"])
    #bl_w4 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_4:0"])
    #bl_w5 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_5:0"])
    #bl_w6 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_6:0"])
    #bl_w7 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_7:0"])
    #bl_w8 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_8:0"])
    bl_rand_map = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_8"]["residual_sign_8"]["means:0"]
    
    #w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask, bl_w3*bl_pruning_mask, bl_w4*bl_pruning_mask, bl_w5*bl_pruning_mask, bl_w6*bl_pruning_mask, bl_w7*bl_pruning_mask, bl_w8*bl_pruning_mask]
    w_lut = [bl_w1]
    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map])
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
    
    bl_w1 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"])
    #bl_w2 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_2:0"])
    #bl_w3 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_3:0"])
    #bl_w4 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_4:0"])
    #bl_w5 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_5:0"])
    #bl_w6 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_6:0"])
    #bl_w7 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_7:0"])
    #bl_w8 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_8:0"])
    bl_rand_map = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["moving_variance:0"])+batch_norm_eps)
    
    #bl_means = bl["model_weights"]["residual_sign_9"]["residual_sign_9"]["means:0"]
    
    #w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask, bl_w3*bl_pruning_mask, bl_w4*bl_pruning_mask, bl_w5*bl_pruning_mask, bl_w6*bl_pruning_mask, bl_w7*bl_pruning_mask, bl_w8*bl_pruning_mask]
    w_lut = [bl_w1]
    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map])
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
    weights[0][0] = SignNumpy(weights[0][0])

    for i in range(1,9):
        if i==5:
            for j in range(256):
                weights[i][j] = SignNumpy(weights[i][j])
        else:
            for j in range(1):
                weights[i][j] = SignNumpy(weights[i][j])

    # write header file
    with open('../src/weights.h', 'w') as f:
        f.write('#pragma once\n')
    with open('../src/weights.h', 'a') as f:
        f.write('//Generated weights for CIFAR-10\n')

    for layer_id in range(9):
        # generate weights
        if layer_id!=5:  # first layer: fxp inputs and binary weights
            weights_per_act = 1
        else:
            weights_per_act = 256 # weights_per_act = #_of_bits_per_act x 2 ^ #_of_lut_inputs

        dims = np.shape(weights[layer_id][0])
        if len(dims)==2:
            layer_type = "fc"
            word_length = dims[0]
            nfilters = dims[1]
        elif len(dims)==4:
            layer_type = "conv"
            word_length = dims[0]*dims[1]*dims[2]
            nfilters = dims[3]

#        for weight_id in range(weights_per_act):
#            mat = weights[layer_id][weight_id]
#            if layer_type=="fc":
#                mat_flat = mat.transpose(1,0).flatten()
#            elif layer_type=="conv":
#                mat_flat = mat.transpose(3,0,1,2).flatten()
#            else:
#                print("unknown weight format!")
#
#            with open('../src/weights.h', 'a') as f:
#                f.write('//Array shape: {}\n'.format(dims))
#                fold = (word_length-1)/32 + 1
#                f.write("const ap_uint<32> " + "weights_" + layer_type + str(layer_id+1) + "_" + str(weight_id+1) + "["+str(nfilters*fold) + "] = {")
#                bin_append = 0
#                for i, ele in enumerate(mat_flat):
#                    #bin_append = (bin_append << 1) | (int(ele) # left-first bit-push
#                    bin_append = bin_append | (int(ele) << (i % word_length)) # right-first bit-push
#                    if (i % word_length == (word_length - 1)):
#                        mask = 0xFFFFFFFF
#                        for i_32b in range(fold):
#                            #word = (bin_append>>(32*(fold-i_32b-1))) & mask # Big-endian: left-first word-push
#                            word = (bin_append>>(32*i_32b)) & mask # Little-endian: right-first word-push
#                            hex_word = '%X' % word
#                            if i_32b!=0:
#                                f.write(', ')
#                            f.write('0x' + hex_word)
#                        bin_append = 0
#                        if i != nfilters*word_length-1:
#                            f.write(', ')
#                f.write('};\n')

        if layer_id==5:
            # generate verilog source file for LUTARRAY: Vivado HLS will take forever
            with open('../src/LUTARRAY_b0_' + str(layer_id) + '.v', 'w') as v0:
                v0.write('`timescale 1 ns / 1 ps\n\n')
                v0.write('module LUTARRAY_b0 (\n        in_V,\n        in_1_V,\n        in_2_V,\n        in_3_V,\n        in_4_V,\n        in_5_V,\n        in_6_V')
                for tm in range(nfilters):
                    v0.write(',\n        ap_return_' + str(tm))
                v0.write(');\n\n')
            with open('../src/LUTARRAY_b1_' + str(layer_id) + '.v', 'w') as v1:
                v1.write('`timescale 1 ns / 1 ps\n\n')
                v1.write('module LUTARRAY_b1 (\n        in_V,\n        in_1_V,\n        in_2_V,\n        in_3_V,\n        in_4_V,\n        in_5_V,\n        in_6_V')
                for tm in range(nfilters):
                    v1.write(',\n        ap_return_' + str(tm))
                v1.write(');\n\n')

            mat_flat = []
            for weight_id in range(weights_per_act):
                mat = weights[layer_id][weight_id]
                pm = pruning_masks[layer_id]#.transpose(3,0,1,2).flatten()
                if layer_type=="fc":
                    mat = mat.transpose(1,0)
                    pm_flat = pm.transpose(1,0)
                elif layer_type=="conv":
                    mat = mat.transpose(3,0,1,2).reshape((nfilters, -1))
                    pm_flat = pm.transpose(3,0,1,2).reshape((nfilters, -1))
                else:
                    print("unknown weight format!")
                mat_flat.extend([mat])        
            with open('../src/LUTARRAY_b0_' + str(layer_id) + '.v', 'a') as v0:
                v0.write('\n\n')
                v0.write('input  [' + str(word_length-1) + ':0] in_V;\n')
                v0.write('input  [' + str(word_length-1) + ':0] in_1_V;\n')
                v0.write('input  [' + str(word_length-1) + ':0] in_2_V;\n')
                v0.write('input  [' + str(word_length-1) + ':0] in_3_V;\n')
                v0.write('input  [' + str(word_length-1) + ':0] in_4_V;\n')
                v0.write('input  [' + str(word_length-1) + ':0] in_5_V;\n')
                v0.write('input  [' + str(word_length-1) + ':0] in_6_V;\n')
                for tm in range(nfilters):
                    v0.write('output  [' + str(word_length-1) + ':0] ap_return_' + str(tm) + ';\n')
                for tm in range(nfilters):
                    for ti, ele in enumerate(pm_flat[tm]):
                        if ele==1:
                            v0.write('wire tmp_' + str(tm) + '_' + str(ti) + ';\n')
                            v0.write('assign tmp_' + str(tm) + '_' + str(ti) + ' = ')
                            v0.write('(' + str(int(mat_flat[128][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[129][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[130][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[131][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[132][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[133][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[134][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[135][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[136][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[137][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[138][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[139][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[140][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[141][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[142][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[143][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[144][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[145][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[146][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[147][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[148][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[149][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[150][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[151][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[152][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[153][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[154][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[155][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[156][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[157][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[158][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[159][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[160][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[161][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[162][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[163][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[164][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[165][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[166][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[167][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[168][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[169][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[170][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[171][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[172][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[173][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[174][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[175][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[176][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[177][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[178][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[179][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[180][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[181][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[182][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[183][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[184][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[185][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[186][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[187][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[188][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[189][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[190][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[191][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[192][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[193][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[194][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[195][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[196][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[197][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[198][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[199][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[200][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[201][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[202][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[203][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[204][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[205][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[206][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[207][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[208][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[209][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[210][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[211][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[212][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[213][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[214][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[215][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[216][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[217][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[218][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[219][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[220][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[221][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[222][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[223][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[224][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[225][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[226][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[227][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[228][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[229][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[230][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[231][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[232][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[233][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[234][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[235][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[236][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[237][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[238][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[239][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[240][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[241][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[242][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[243][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[244][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[245][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[246][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[247][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[248][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[249][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[250][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[251][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[252][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[253][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[254][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[255][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']);\n')

                    v0.write('assign ap_return_' + str(tm) + ' = {')
                    for ti, ele in enumerate(pm_flat[tm]):
                        if ele == 0:
                            v0.write("1'b0")
                        elif ele == 1:
                            v0.write('tmp_' + str(tm) + '_' + str(ti))
                        else:
                            print("pruning mask elements must be binary!")
                        if ti != word_length-1:
                            v0.write(',')
                        else:
                            v0.write('};\n')
                v0.write('endmodule')
            with open('../src/LUTARRAY_b1_' + str(layer_id) + '.v', 'a') as v1:
                v1.write('\n\n')
                v1.write('input  [' + str(word_length-1) + ':0] in_V;\n')
                v1.write('input  [' + str(word_length-1) + ':0] in_1_V;\n')
                v1.write('input  [' + str(word_length-1) + ':0] in_2_V;\n')
                v1.write('input  [' + str(word_length-1) + ':0] in_3_V;\n')
                v1.write('input  [' + str(word_length-1) + ':0] in_4_V;\n')
                v1.write('input  [' + str(word_length-1) + ':0] in_5_V;\n')
                v1.write('input  [' + str(word_length-1) + ':0] in_6_V;\n')
                for tm in range(nfilters):
                    v1.write('output  [' + str(word_length-1) + ':0] ap_return_' + str(tm) + ';\n')
                for tm in range(nfilters):
                    for ti, ele in enumerate(pm_flat[tm]):
                        if ele==1:
                            v1.write('wire tmp_' + str(tm) + '_' + str(ti) + ';\n')
                            v1.write('assign tmp_' + str(tm) + '_' + str(ti) + ' = ')
                            v1.write('(' + str(int(mat_flat[0][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[1][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[2][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[3][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[4][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[5][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[6][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[7][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[8][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[9][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[10][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[11][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[12][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[13][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[14][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[15][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[16][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[17][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[18][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[19][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[20][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[21][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[22][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[23][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[24][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[25][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[26][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[27][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[28][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[29][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[30][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[31][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[32][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[33][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[34][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[35][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[36][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[37][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[38][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[39][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[40][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[41][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[42][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[43][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[44][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[45][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[46][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[47][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[48][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[49][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[50][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[51][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[52][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[53][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[54][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[55][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[56][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[57][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[58][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[59][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[60][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[61][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[62][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[63][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[64][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[65][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[66][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[67][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[68][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[69][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[70][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[71][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[72][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[73][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[74][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[75][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[76][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[77][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[78][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[79][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[80][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[81][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[82][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[83][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[84][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[85][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[86][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[87][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[88][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[89][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[90][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[91][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[92][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[93][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[94][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[95][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[96][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[97][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[98][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[99][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[100][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[101][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[102][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[103][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[104][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[105][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[106][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[107][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[108][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[109][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[110][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[111][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[112][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[113][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[114][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[115][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[116][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[117][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[118][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[119][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[120][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[121][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[122][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[123][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[124][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[125][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[126][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & in_6_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[127][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + '] & ~in_6_V[' + str(ti) + ']);\n')

                    v1.write('assign ap_return_' + str(tm) + ' = {')
                    for ti, ele in enumerate(pm_flat[tm]):
                        if ele == 0:
                            v1.write("1'b0")
                        elif ele == 1:
                            v1.write('tmp_' + str(tm) + '_' + str(ti))
                        else:
                            print("pruning mask elements must be binary!")
                        if ti != word_length-1:
                            v1.write(',')
                        else:
                            v1.write('};\n')
                v1.write('endmodule')

        # generate pruning mask (first layer only)
        if layer_id==0:
            pruning_mask_flat = pruning_masks[layer_id].transpose(3,0,1,2).flatten()
            with open('../src/weights.h', 'a') as f:
                fold = (word_length-1)/32 + 1
                f.write("const ap_uint<32> " + "pruning_mask_" + layer_type + str(layer_id+1) + "_" + str(1) + "["+str(nfilters*fold) + "] = {")
                bin_append = 0
                for i, ele in enumerate(pruning_mask_flat):
                    #bin_append = (bin_append << 1) | (int(ele) # left-first bit-push
                    bin_append = bin_append | (int(ele) << (i % word_length)) # right-first bit-push
                    if (i % word_length == (word_length - 1)):
                        mask = 0xFFFFFFFF
                        for i_32b in range(fold):
                            #word = (bin_append>>(32*(fold-i_32b-1))) & mask # Big-endian: left-first word-push
                            word = (bin_append>>(32*i_32b)) & mask # Little-endian: right-first word-push
                            hex_word = '%X' % word
                            if i_32b!=0:
                                f.write(', ')
                            f.write('0x' + hex_word)
                        bin_append = 0
                        if i != nfilters*word_length-1:
                            f.write(', ')
                f.write('};\n')
        # generate threshold
        if layer_id!=8: # the last layer does not need threshold
            use_popcount = not(layer_id==0)
            next_means_b0 = abs(means[layer_id][0])
            print(next_means_b0)
            next_means_b1 = abs(means[layer_id][1])
            print(next_means_b1)
            if layer_type=="conv":
                fanin = np.sum(pruning_masks[layer_id].reshape(-1,dims[3]),axis=0)
            elif layer_type=="fc":
                fanin = np.sum(pruning_masks[layer_id],axis=0)
            if layer_id!=0:
                fanin = fanin * abs(gammas[layer_id] * means[layer_id-1][0]) + fanin * abs(gammas[layer_id] * means[layer_id-1][1])
            thresholds = np.array(makeBNComplex(0, fanin, bn_betas[layer_id], bn_gammas[layer_id], bn_means[layer_id], bn_inv_stds[layer_id], usePopCount=use_popcount))
            next_means_bn_b0 = np.array(makeBNComplex(next_means_b0, fanin, bn_betas[layer_id], bn_gammas[layer_id], bn_means[layer_id], bn_inv_stds[layer_id], usePopCount=use_popcount)) - thresholds

            with open('../src/weights.h', 'a') as f:
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
#            with open('../src/weights.h', 'a') as f:
#                next_means_b0 = abs(means[layer_id][0])
#                next_means_b1 = abs(means[layer_id][1])
#                f.write("const ap_fixed<24, 16> " + "next_layer_means_" + layer_type + str(layer_id+1) + "[2] = {")
#                f.write(str(next_means_b0))
#                f.write(','+ str(next_means_b1))
#                f.write('};\n')


        # generate random map
        for j in range(6):
            with open('../src/weights.h', 'a') as f:
                rand_map = rand_maps[layer_id][j].flatten().astype(np.uint32)
                f.write("const unsigned int " + "rand_map_" + str(j) + "_" + layer_type + str(layer_id+1) + "["+str(len(rand_map))+"] = {")
                for i, ele in enumerate(rand_map):
                    if i == 0:
                        f.write(str(ele))
                    else:
                        f.write(','+ str(ele))
                f.write('};\n')
        # generate alpha
        with open('../src/weights.h', 'a') as f:
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




