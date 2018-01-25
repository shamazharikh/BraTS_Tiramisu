import tensorflow as tf
import numpy as np
from functools import reduce
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary
import time
from scipy.ndimage.measurements import label
import os
class DimensionError(Exception):
    pass


class MissingInput(Exception):
    pass


class DeepAPI(object):
    def __init__(self, model_name, memory_fraction=None, *args, **kwargs):
        self._model_name = model_name
        if memory_fraction==None:
            self._sess = tf.Session()
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
            self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    def __del__(self):
        self._sess.close()
        tf.reset_default_graph()

    def _get_wasserstein_mat(self, n_classes):
        return np.triu(np.ones((n_classes,
                    n_classes)))[:,:-1].astype(np.float32)

    def wasserstein_ordered_loss(self, softmax, labels):
        diff = softmax - labels
        n_classes = int(labels.shape[-1])
        self._wass_mat = self._get_wasserstein_mat(n_classes)
        sum_mat = tf.square(tf.multiply(diff, self._wass_mat[:, 0]))
        for i in range(1, self._wass_mat.shape[1]):
            sum_mat = sum_mat + tf.square(tf.multiply(diff,
                                                      self._wass_mat[:, i]))
        return tf.reduce_mean(sum_mat)

    def _get_fans(self, shape):
        if len(shape) >= 4:
            receptive_field_size = reduce(lambda x, y: x * y, shape[:-2])
            fan_in = receptive_field_size * shape[-2]
            fan_out = receptive_field_size * shape[-1]
            return fan_in, fan_out
        elif len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
            return fan_in, fan_out
        return None

    def batch_norm_wrapper(self,
                           input_tensor,
                           param_dict,
                           layer_name='batch_norm',
                           scale_offset=True,
                           decay=0.999):
        '''
        V v slight improvement of r2rt.com implementation
        '''
        epsilon = 1e-4
        shape=[int(input_tensor.shape[-1])]
        scale = tf.Variable(tf.ones(shape), name='scale') if scale_offset else None
        beta = tf.Variable(tf.zeros(shape), name='beta') if scale_offset else None
        pop_mean = tf.Variable(tf.zeros(shape),
                               trainable=False, name='pop_mean')
        pop_var = tf.Variable(tf.ones(shape),
                              trainable=False, name='pop_var')

        for x in [scale, beta, pop_mean, pop_var]:
            if x is not None:
                name = x.name
                param_dict[name] = x
        axes=[ i for i in range(len(input_tensor.shape)-1)]
        batch_mean, batch_var = tf.nn.moments(input_tensor, axes)
        condition1 = tf.cond(self._is_training,
                             lambda: tf.assign(pop_mean, pop_mean * decay\
                                            + batch_mean * (1 - decay)),
                             lambda: tf.assign(pop_mean, pop_mean))
        condition2 = tf.cond(self._is_training,
                             lambda: tf.assign(pop_var, pop_var * decay\
                                            + batch_var * (1 - decay)),
                             lambda: tf.assign(pop_var, pop_var))
        with tf.control_dependencies([condition1,condition2]):
            return tf.nn.batch_normalization(input_tensor,
                                             pop_mean,
                                             pop_var,
                                             beta,
                                             scale,
                                             epsilon)

    def batch_scale_offset(self,
                           input_tensor,
                           param_dict,
                           layer_name='batch_norm',
                           axis=1):
        shape = [i if i == axis else 1 for i in range(len(input_tensor.shape))]
        scale = tf.Variable(tf.ones(shape), name=layer_name+'scale')
        beta = tf.Variable(tf.zeros(shape), name=layer_name+'beta')
        for x in [scale, beta]:
            name = x.name
            param_dict[name] = x
        return input_tensor * scale + beta

    def fully_connected_layer(self,
                              input_tensor,
                              weight=None,
                              bias=None,
                              output_dim=None,
                              layer_name='FC',
                              param_dict=None,
                              relu=True,
                              dropout=True,
                              batch_norm=False,
                              reuse=False):
        if not len(input_tensor.shape) == 2:
            raise DimensionError('Input to fully_connected_layer must be\
                length 2 like [None, x] but here it is {}'.format(
                                                    input_tensor.shape))
        if output_dim is None:
            raise MissingInput('Output dimension should be specified \
                for fully_connected_layer')
        if weight is None and bias is None:

                with tf.variable_scope(self._model_name + layer_name, reuse=reuse):
                    input_dim = int(input_tensor.shape[-1])
                    weight = self.glorot_uniform_init((input_dim, output_dim), varname='weight')
                    bias = self.glorot_uniform_init((output_dim,), varname='bias')
                    for x in [weight, bias]:
                        name = x.name
                        param_dict[name] = x

        if batch_norm:
            with tf.name_scope(self._model_name + layer_name) as scope:
                input_tensor = self.batch_norm_wrapper(input_tensor,
                                                       param_dict)
        if relu:
            input_tensor = tf.nn.relu(input_tensor)
        input_tensor = tf.matmul(input_tensor, weight) + bias
        if dropout:
            input_tensor = tf.nn.dropout(input_tensor, self._dropout_prob)
        return input_tensor

    def soft_dice_loss(self,logits,labels, index=-1, weights=None):
        axis = list(range(len(logits.shape)))
        if index<0:
            index = len(logits.shape) + index
        axis=axis.remove(index)
        logits=tf.nn.softmax(logits)
        num = 2*tf.reduce_sum(logits*labels, axis=axis)
        denom = tf.reduce_sum(tf.pow(logits,2), axis=axis) + tf.reduce_sum(labels, axis=axis)
        score = num/denom
        if weights is not None:
            weights = tf.constant(np.array(weights, dtype=np.float32))
            score = weights*score
        return tf.reduce_mean(score)

    def weighted_softmax_cross_entropy_with_logits(self,
                                                    logits,
                                                    labels,
                                                    weights=None,
                                                    axis=-1):
        loss_map = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, dim=axis)
        if weights is not None:
            weights = np.array(weights, dtype=np.float32)
            shape = [1]*len(logits.shape)
            shape[axis] = -1
            weights = weights.reshape(shape)
            weight = tf.constant(weights)
            weight_map = tf.reduce_sum(tf.multiply(labels, weight), axis=axis)
            loss_map = tf.multiply(loss_map, weight_map)
        return tf.reduce_mean(loss_map)

    def _get_glorot_limit(self, fan_in, fan_out):
        return np.sqrt(6.0 / (fan_in + fan_out))

    def glorot_uniform_init(self, shape, varname=None):
        fans = self._get_fans(shape)
        if fans is None:
            return tf.get_variable(name=varname, initializer=tf.zeros(shape, dtype=tf.float32))
        else:
            limit = self._get_glorot_limit(*fans)
            return tf.get_variable(
                initializer=tf.random_uniform(
                    shape, minval=-limit, maxval=limit, dtype=tf.float32),
                name=varname)

    def upconv2D_layer(self,
                       input_tensor,
                       weight=None,
                       bias=None,
                       output_shape=None,
                       batch_size=1,
                       filter_size=2,
                       n_filters=None,
                       layer_name='2dupconv',
                       param_dict=None,
                       strides=[1, 2, 2, 1],
                       padding='SAME',
                       reuse=False):
        if not len(input_tensor.shape) == 4:
            raise DimensionError('Input to 2dupconv_layer must be\
                length 4 like [None, w, h, c] but here it is {}'.format(
                                                    input_tensor.shape))
        if weight is None and bias is None:
            with tf.variable_scope(self._model_name + layer_name, reuse=reuse):
                input_dim = int(input_tensor.shape[-1])
                weight = self.glorot_uniform_init((filter_size,
                                               filter_size,
                                               n_filters,
                                               input_dim),varname='weight')
                bias = self.glorot_uniform_init((n_filters,), varname='bias')
                for x in [weight, bias]:
                    name = x.name
                    param_dict[name] = x
        return tf.nn.conv2d_transpose(input_tensor, weight, [
            batch_size, output_shape[0], output_shape[1], n_filters], strides, padding=padding) + bias

    def upconv3D_layer(self,
                       input_tensor,
                       weight=None,
                       bias=None,
                       output_shape=None,
                       batch_size=1,
                       filter_size=2,
                       n_filters=None,
                       layer_name='3dupconv',
                           padding='SAME',
                       param_dict=None,
                       strides=[1, 2, 2, 2, 1]):
        if not len(input_tensor.shape) == 5:
            raise DimensionError('Input to fully_connected_layer must be\
                length 5 like [None, w, h, b, d] but here it is {}'.format(
                                                    input_tensor.shape))
        if weight is None and bias is None:
            with tf.name_scope(self._model_name + layer_name) as scope:
                input_dim = int(input_tensor.shape[-1])
                weight = self.glorot_uniform_init((filter_size,
                                               filter_size,
                                               filter_size,
                                               n_filters,
                                               input_dim),
                                               varname='weight')
                bias = self.glorot_uniform_init((n_filters,),varname='bias')
                for x in [weight, bias]:
                    name = x.name
                    param_dict[name] = x
        return tf.nn.conv3d_transpose(input_tensor,
                                          weight,
                                          [batch_size, output_shape[0], output_shape[1], output_shape[2], n_filters],
                                          strides,
                                          padding=padding) + bias

    def maxpool_layer(self,
                      input_tensor,
                      kernel_size=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1],
                      padding='VALID'):
        return tf.nn.max_pool(input_tensor, kernel_size, strides, padding)

    def maxpool3D_layer(self,
                      input_tensor,
                      kernel_size=[1, 2, 2, 2, 1],
                      strides=[1, 2, 2, 2, 1],
                      padding='VALID',
                      data_format='NCDHW'):
        return tf.nn.max_pool3d(input_tensor, kernel_size, strides, padding)

    def avgpool_layer(self,
                      input_tensor,
                      kernel_size=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1],
                      padding='VALID'):
        return tf.nn.avg_pool(input_tensor, kernel_size, strides, padding)

    def avgpool3D_layer(self,
                      input_tensor,
                      kernel_size=[1, 2, 2, 2, 1],
                      strides=[1, 2, 2, 2, 1],
                      padding='VALID',
                      data_format='NCDHW'):
        return tf.nn.avg_pool3d(input_tensor, kernel_size, strides, padding)


    def conv2D_atrous_layer(self,
                            input_tensor,
                            dilation_rate,
                            weight=None,
                            bias=None,
                            filter_size=None,
                            n_filters=None,
                            layer_name='2dconv_atrous',
                            param_dict=None,
                            padding='SAME',
                            relu=False,
                            elu=False,
                            dropout=False,
                            batch_norm=False):
        if not len(input_tensor.shape) == 4:
            raise DimensionError('Input to 2D_convolutional_layer must be\
                rank 4 like [batch_size, height, width, channels] but here it is {}'.format(
                                                    input_tensor.shape))
        if n_filters is None:
            raise MissingInput('Number of filters should be specified \
                for convolutional_layer')
        if filter_size is None:
            raise MissingInput('Filter size should be specified \
                for convolutional_layer')
        if weight is None and bias is None:
            param_dict = param_dict or self._params
            with tf.name_scope(self._model_name + layer_name) as scope:
                input_dim = int(input_tensor.shape[-1])
                weight = self.glorot_uniform_init((filter_size, filter_size, input_dim, n_filters), varname='weight')
                bias = self.glorot_uniform_init((n_filters,), varname='bias')
                for x in [weight, bias]:
                    name = x.name
                    param_dict[name] = x
        if batch_norm:
            with tf.name_scope(self._model_name + layer_name) as scope:
                input_tensor = self.batch_norm_wrapper(input_tensor,
                                                       param_dict)
        if relu:
            input_tensor = tf.nn.relu(input_tensor)
        if elu:
            input_tensor = tf.nn.elu(input_tensor)
        input_tensor = tf.nn.atrous_conv2d(
            input_tensor, weight, rate=dilation_rate, padding=padding) + bias
        if dropout:
            input_tensor = tf.nn.dropout(input_tensor, self._dropout_prob)
        return input_tensor

    def conv2D_layer(self,
                     input_tensor,
                     weight=None,
                     bias=None,
                     bias_type = 'Normal',
                     filter_size=None,
                     n_filters=None,
                     layer_name='2dconv',
                     param_dict=None,
                     padding='SAME',
                     strides=(1,1),
                     relu=False,
                     elu=False,
                     dropout=False,
                     batch_norm=False):

        if not len(input_tensor.shape) == 4:
            raise DimensionError('Input to 2D_convolutional_layer must be\
                rank 4 like [batch_size, channels, height, width] but here it is {}'.format(
                                                    input_tensor.shape))
        if n_filters is None:
            raise MissingInput('Number of filters should be specified \
                for convolutional_layer')
        if filter_size is None:
            raise MissingInput('Filter size should be specified \
                for convolutional_layer')
        if weight is None:
            with tf.name_scope(self._model_name + layer_name) as scope:
                input_dim = int(input_tensor.shape[-1])
                weight = self.glorot_uniform_init((filter_size, filter_size, input_dim, n_filters), varname='weight')
                name = weight.name
                param_dict[name] = weight
        if batch_norm:
            with tf.name_scope(self._model_name + layer_name) as scope:
                input_tensor = self.batch_norm_wrapper(input_tensor,
                                                       param_dict)
        if relu:
            input_tensor = tf.nn.relu(input_tensor)
        if elu:
            input_tensor = tf.nn.elu(input_tensor)
        input_tensor = tf.nn.conv2d(
            input_tensor, weight, strides=[1, strides[0], strides[1], 1], padding=padding)

        if bias is None:
            with tf.name_scope(self._model_name + layer_name) as scope:

                if bias_type == 'Normal':
                    shape = (int(input_tensor.shape[-1]),)
                    bias = self.glorot_uniform_init(shape, varname='bias')
                elif bias_type == 'Full':
                    bias = self.glorot_uniform_init((map(int,input_tensor.shape[1:])), varname='bias')
                name = bias.name
                param_dict[name] = bias
        input_tensor = input_tensor + bias
        if dropout:
            input_tensor = tf.nn.dropout(input_tensor, self._dropout_prob)
        return input_tensor

    def conv3D_layer(self,
                     input_tensor,
                     weight=None,
                     bias=None,
                     n_filters=None,
                     filter_size=None,
                     layer_name='3dconv',
                     param_dict=None,
                     padding='SAME',
                     strides=(1,1,1),
                     relu=False,
                     elu=False,
                     dropout=True,
                     batch_norm=False):
        if not len(input_tensor.shape) == 5:
            raise DimensionError('Input to 3D_convolutional_layer must be\
                rank 5 like [batch_size, height, width, depth, channels] but here it is {}'.format(
                                                    input_tensor.shape))
        if n_filters is None:
            raise MissingInput('Number of filters should be specified \
                for convolutional_layer')
        if filter_size is None:
            raise MissingInput('Filter size should be specified \
                for convolutional_layer')
        if weight is None and bias is None:
            with tf.name_scope(self._model_name + layer_name) as scope:
                input_dim = int(input_tensor.shape[-1])
                weight = self.glorot_uniform_init((filter_size, filter_size, filter_size, input_dim, n_filters))
                bias = self.glorot_uniform_init((n_filters,))
                for x in [weight, bias]:
                    name = x.name
                    param_dict[name] = x
        if batch_norm:
            with tf.name_scope(self._model_name + layer_name) as scope:
                input_tensor = self.batch_norm_wrapper(input_tensor,
                                                       param_dict)
        if relu:
            input_tensor = tf.nn.relu(input_tensor)
        if elu:
            input_tensor = tf.nn.elu(input_tensor)
        input_tensor = tf.nn.conv3d(
            input_tensor, weight, strides=[1, strides[0], strides[1], strides[2], 1], padding=padding, data_format="NDHWC") + bias
        if dropout:
            input_tensor = tf.nn.dropout(input_tensor, self._dropout_prob)
        return input_tensor

    def _class_dice(self, predict, truth, class_index, dice=False):
        intersection = np.sum((truth == class_index)[predict == class_index])
        denom = np.sum(truth == class_index) + \
                        np.sum(predict == class_index) + 1e-06
        if dice:
            return 2 * intersection / denom
        return np.array([2 * intersection, denom])

    def _dice_score(self,
                    predict,
                    truth,
                    classes = {'WT':[1,2,3],
                               'TC':[1,3],
                               'AT':[3]},
                    dice=True):
        dices = {}
        predict = np.round(predict).astype(np.uint8)
        truth = np.round(truth).astype(np.uint8)
        for key in classes.keys():
            copy_predict = np.copy(predict)
            copy_truth = np.copy(truth)
            class_truth = np.zeros_like(truth)
            class_predict = np.zeros_like(predict)
            for l in classes[key]:
                class_truth[copy_truth==l]=1
                class_predict[copy_predict==l]=1
            num = 2*np.sum(class_predict[class_truth==1])
            denom = np.sum(class_predict) + np.sum(class_truth) + 1e-09
            if dice:
                dices[key] = num/denom
            else:
                dices[key] = np.array([num, denom])
        return dices

    def old_dice_score(self,
                    predict,
                    truth,
                    dice=True):

        dices = {}
        class1 = np.sum((truth==1).astype('int'))
        class2 = np.sum((truth==2).astype('int'))
        class3 = np.sum((truth==3).astype('int'))
        tot = max(1, class1+class2+class3)
        class1_p = 100*class1/tot
        class2_p = 100*class2/tot
        class3_p = 100*class3/tot
        classtc = class1+class3
        classtc_p = 100*classtc/tot
        WT_truth = np.copy(truth)
        WT_truth[WT_truth>0] = 1
        WT_predict = np.copy(predict)
        WT_predict[WT_predict>0] = 1
        WT_num = np.sum(WT_predict[WT_truth==1])*2.0
        WT_denom = (np.sum(WT_predict)+np.sum(WT_truth)+1e-9)
        WT_dice = WT_num/WT_denom
        del WT_truth
        del WT_predict
        TC_truth = np.copy(truth)
        TC_truth[TC_truth==2] = 0
        TC_truth[TC_truth>0] = 1
        TC_predict = np.copy(predict)
        TC_predict[TC_predict==2] = 0
        TC_predict[TC_predict>0] = 1
        TC_num = np.sum(TC_predict[TC_truth==1])*2.0
        TC_denom = (np.sum(TC_predict)+np.sum(TC_truth)+1e-9)
        TC_dice = TC_num/TC_denom
        del TC_truth
        del TC_predict
        AT_truth = np.copy(truth)
        AT_truth[AT_truth<3] = 0
        AT_truth[AT_truth>0] = 1
        AT_predict = np.copy(predict)
        AT_predict[AT_predict<3]=0
        AT_predict[AT_predict>0] = 1
        AT_num = np.sum(AT_predict[AT_truth==1])*2.0
        AT_denom = (np.sum(AT_predict)+np.sum(AT_truth)+1e-9)
        AT_dice = AT_num/AT_denom
        del AT_truth
        del AT_predict
        if dice:
            return {'WT':WT_dice, 'TC':TC_dice, 'AT':AT_dice}
        else:
            return {'WT':np.array([WT_num,WT_denom]),
                    'TC':np.array([TC_num,TC_denom]),
                    'AT':np.array([AT_num,AT_denom])}
    def tumour_dice_score(self, predict, truth, dice=True):
        WT_truth = np.copy(truth)
        WT_truth[WT_truth<2] = 0
        WT_truth[WT_truth>0] = 1
        WT_predict = np.copy(predict)
        WT_predict[WT_predict<2]=0
        WT_predict[WT_predict>0] = 1
        WT_num = np.sum(WT_predict[WT_truth==1])*2.0
        WT_denom = (np.sum(WT_predict)+np.sum(WT_truth)+1e-9)
        WT_dice = WT_num/WT_denom
        del WT_truth
        del WT_predict
        if dice:
            return {'WT': WT_dice}
        else:
            return {'WT': np.array([WT_num,WT_denom])}

    def one_hot_encoding(self, x, labels=[0, 1, 2, 3]):
        return np.stack([1 * (x == l) for l in labels], axis=-1)

    def _save_weights(self, savepath):
        folder_dir = os.path.split(savepath)[0]
        score_string = '_'.join('{}:{:.2f}'.format(x, self._best_score[x]) for x in self._best_score.keys())
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        self._saver.save(self._sess, savepath.format(score_string))

    def _restore_weights(self, savepath):
        print('Restoring Parameters')
        start = time.time()
        score_string = '_'.join('{}:{:.2f}'.format(x, self._best_score[x]) for x in self._best_score.keys())
        self._saver.restore(self._sess, savepath.format(score_string))
        end = time.time()
        print('Restored in :', end-start)

    def CRF(
        self,
        image,
        softmax,
        iterations = 1,
        sdims_g=(1, 1),
        sdims_b=(3, 3),
        schan=0.5):
        softmax = softmax.transpose(2,0,1)
        n_classes = softmax.shape[0]
        unary = unary_from_softmax(softmax)
        d = dcrf.DenseCRF(image.shape[1]*image.shape[0], n_classes)
        d.setUnaryEnergy(unary)
        feats = create_pairwise_gaussian(sdims=sdims_g, shape=image.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)


        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=sdims_b, schan=schan,
                                          img=image, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(iterations)
        probabilities = np.array(Q).reshape((n_classes,image.shape[0],-1))
        labels = np.argmax(Q, axis=0).reshape(image.shape[0:2])

        return probabilities.transpose(1,2,0), labels

    def correct_labels(self, labels):
        labels[labels==3]=4
        return labels.transpose(1,0)

    def connected_components(self, voxels, threshold=12000):
        c,n = label(voxels)
        nums = np.array([np.sum(c==i) for i in range(1, n+1)])
        selected_components = nums>threshold
        selected_components[np.argmax(nums)] = True
        mask = np.zeros_like(voxels)
        print(selected_components.tolist())
        for i,select in enumerate(selected_components):
            if select:
                mask[c==(i+1)]=1
        return mask*voxels, mask
