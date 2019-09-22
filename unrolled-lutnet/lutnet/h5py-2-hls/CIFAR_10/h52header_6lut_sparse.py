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

    bl = h5py.File("pretrained_network_6lut.h5", 'r')
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

    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_2:0"])
    bl_rand_map_3 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_3:0"])
    bl_rand_map_4 = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["rand_map_4:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_6"]["binary_conv_6"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_6"]["batch_normalization_6"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_6"]["residual_sign_6"]["means:0"]

    w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask,  bl_w3*bl_pruning_mask,  bl_w4*bl_pruning_mask,  bl_w5*bl_pruning_mask,  bl_w6*bl_pruning_mask,  bl_w7*bl_pruning_mask,  bl_w8*bl_pruning_mask,  bl_w9*bl_pruning_mask,  bl_w10*bl_pruning_mask, bl_w11*bl_pruning_mask, bl_w12*bl_pruning_mask, bl_w13*bl_pruning_mask, bl_w14*bl_pruning_mask, bl_w15*bl_pruning_mask, bl_w16*bl_pruning_mask, bl_w17*bl_pruning_mask, bl_w18*bl_pruning_mask, bl_w19*bl_pruning_mask, bl_w20*bl_pruning_mask, bl_w21*bl_pruning_mask, bl_w22*bl_pruning_mask, bl_w23*bl_pruning_mask, bl_w24*bl_pruning_mask, bl_w25*bl_pruning_mask, bl_w26*bl_pruning_mask, bl_w27*bl_pruning_mask, bl_w28*bl_pruning_mask, bl_w29*bl_pruning_mask, bl_w30*bl_pruning_mask, bl_w31*bl_pruning_mask, bl_w32*bl_pruning_mask, bl_w33*bl_pruning_mask, bl_w34*bl_pruning_mask, bl_w35*bl_pruning_mask, bl_w36*bl_pruning_mask, bl_w37*bl_pruning_mask, bl_w38*bl_pruning_mask, bl_w39*bl_pruning_mask, bl_w40*bl_pruning_mask, bl_w41*bl_pruning_mask, bl_w42*bl_pruning_mask, bl_w43*bl_pruning_mask, bl_w44*bl_pruning_mask, bl_w45*bl_pruning_mask, bl_w46*bl_pruning_mask, bl_w47*bl_pruning_mask, bl_w48*bl_pruning_mask, bl_w49*bl_pruning_mask, bl_w50*bl_pruning_mask, bl_w51*bl_pruning_mask, bl_w52*bl_pruning_mask, bl_w53*bl_pruning_mask, bl_w54*bl_pruning_mask, bl_w55*bl_pruning_mask, bl_w56*bl_pruning_mask, bl_w57*bl_pruning_mask, bl_w58*bl_pruning_mask, bl_w59*bl_pruning_mask, bl_w60*bl_pruning_mask, bl_w61*bl_pruning_mask, bl_w62*bl_pruning_mask, bl_w63*bl_pruning_mask, bl_w64*bl_pruning_mask, bl_w65*bl_pruning_mask, bl_w66*bl_pruning_mask, bl_w67*bl_pruning_mask, bl_w68*bl_pruning_mask, bl_w69*bl_pruning_mask, bl_w70*bl_pruning_mask, bl_w71*bl_pruning_mask, bl_w72*bl_pruning_mask, bl_w73*bl_pruning_mask, bl_w74*bl_pruning_mask, bl_w75*bl_pruning_mask, bl_w76*bl_pruning_mask, bl_w77*bl_pruning_mask, bl_w78*bl_pruning_mask, bl_w79*bl_pruning_mask, bl_w80*bl_pruning_mask, bl_w81*bl_pruning_mask, bl_w82*bl_pruning_mask, bl_w83*bl_pruning_mask, bl_w84*bl_pruning_mask, bl_w85*bl_pruning_mask, bl_w86*bl_pruning_mask, bl_w87*bl_pruning_mask, bl_w88*bl_pruning_mask, bl_w89*bl_pruning_mask, bl_w90*bl_pruning_mask, bl_w91*bl_pruning_mask, bl_w92*bl_pruning_mask, bl_w93*bl_pruning_mask, bl_w94*bl_pruning_mask, bl_w95*bl_pruning_mask, bl_w96*bl_pruning_mask, bl_w97*bl_pruning_mask, bl_w98*bl_pruning_mask, bl_w99*bl_pruning_mask, bl_w100*bl_pruning_mask, bl_w101*bl_pruning_mask, bl_w102*bl_pruning_mask, bl_w103*bl_pruning_mask, bl_w104*bl_pruning_mask, bl_w105*bl_pruning_mask, bl_w106*bl_pruning_mask, bl_w107*bl_pruning_mask, bl_w108*bl_pruning_mask, bl_w109*bl_pruning_mask, bl_w110*bl_pruning_mask, bl_w111*bl_pruning_mask, bl_w112*bl_pruning_mask, bl_w113*bl_pruning_mask, bl_w114*bl_pruning_mask, bl_w115*bl_pruning_mask, bl_w116*bl_pruning_mask, bl_w117*bl_pruning_mask, bl_w118*bl_pruning_mask, bl_w119*bl_pruning_mask, bl_w120*bl_pruning_mask, bl_w121*bl_pruning_mask, bl_w122*bl_pruning_mask, bl_w123*bl_pruning_mask, bl_w124*bl_pruning_mask, bl_w125*bl_pruning_mask, bl_w126*bl_pruning_mask, bl_w127*bl_pruning_mask, bl_w128*bl_pruning_mask]
    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    bl_rand_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2, bl_rand_map_3, bl_rand_map_4]
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
            for j in range(128):
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
            weights_per_act = 128 # weights_per_act = #_of_bits_per_act x 2 ^ #_of_lut_inputs

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
                v0.write('module LUTARRAY_b0 (\n        in_V,\n        in_1_V,\n        in_2_V,\n        in_3_V,\n        in_4_V,\n        in_5_V')
                for tm in range(nfilters):
                    v0.write(',\n        ap_return_' + str(tm))
                v0.write(');\n\n')
            with open('../src/LUTARRAY_b1_' + str(layer_id) + '.v', 'w') as v1:
                v1.write('`timescale 1 ns / 1 ps\n\n')
                v1.write('module LUTARRAY_b1 (\n        in_V,\n        in_1_V,\n        in_2_V,\n        in_3_V,\n        in_4_V,\n        in_5_V')
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
                for tm in range(nfilters):
                    v0.write('output  [' + str(word_length-1) + ':0] ap_return_' + str(tm) + ';\n')
                for tm in range(nfilters):
                    for ti, ele in enumerate(pm_flat[tm]):
                        if ele==1:
                            v0.write('wire tmp_' + str(tm) + '_' + str(ti) + ';\n')
                            v0.write('assign tmp_' + str(tm) + '_' + str(ti) + ' = ')
                            v0.write('(' + str(int(mat_flat[64][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[65][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[66][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[67][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[68][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[69][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[70][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[71][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[72][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[73][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[74][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[75][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[76][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[77][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[78][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[79][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[80][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[81][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[82][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[83][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[84][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[85][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[86][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[87][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[88][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[89][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[90][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[91][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[92][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[93][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[94][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[95][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[96][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[97][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[98][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[99][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[100][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[101][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[102][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[103][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[104][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[105][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[106][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[107][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[108][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[109][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[110][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[111][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[112][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[113][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[114][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[115][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[116][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[117][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[118][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[119][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[120][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[121][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[122][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[123][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[124][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[125][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[126][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[127][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']);\n')
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
                for tm in range(nfilters):
                    v1.write('output  [' + str(word_length-1) + ':0] ap_return_' + str(tm) + ';\n')
                for tm in range(nfilters):
                    for ti, ele in enumerate(pm_flat[tm]):
                        if ele==1:
                            v1.write('wire tmp_' + str(tm) + '_' + str(ti) + ';\n')
                            v1.write('assign tmp_' + str(tm) + '_' + str(ti) + ' = ')
                            v1.write('(' + str(int(mat_flat[0][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[1][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[2][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[3][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[4][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[5][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[6][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[7][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[8][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[9][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[10][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[11][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[12][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[13][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[14][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[15][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[16][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[17][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[18][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[19][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[20][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[21][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[22][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[23][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[24][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[25][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[26][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[27][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[28][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[29][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[30][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[31][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[32][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[33][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[34][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[35][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[36][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[37][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[38][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[39][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[40][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[41][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[42][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[43][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[44][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[45][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[46][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[47][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[48][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[49][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[50][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[51][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[52][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[53][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[54][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[55][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[56][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[57][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[58][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[59][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[60][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[61][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[62][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & in_5_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[63][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + '] & ~in_4_V[' + str(ti) + '] & ~in_5_V[' + str(ti) + ']);\n')
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
        for j in range(5):
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




