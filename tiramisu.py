from __future__ import division


import tensorflow as tf
from deep_api import DeepAPI
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import nibabel as nib
import pdb
import pandas as pd
import shutil

class Tiramisu(DeepAPI):
    def __init__(self,
                 input_height=512,
                 input_width=512,
                 input_channels=1,
                 filters=3,
                 output_filters=1,
                 n_classes=8,
                 growth=16,
                 saved='saved/brats_small_tiramisu_{}.npz',
                 best_score={'WT':0.84,'TC':0.61,'AT':0.60},
                 short_residual=False,
                 long_residual=False,
                 activation='relu',
                 class_weights=None,
                 layer_depths=[2,2,2],
                 bottle_neck=8,
                 batch_size=1,
                 memory_fraction=0.8):
        super(Tiramisu, self).__init__('tiramisu', memory_fraction=memory_fraction)
        self._input_height = input_height
        self._input_width = input_width
        self._input_channels = input_channels
        self._filters = filters
        self._output_filters = output_filters
        self._n_classes = n_classes
        self._saved = saved
        self._best_score = best_score
        self._batch_size = batch_size
        self._layer_depths = layer_depths
        self._short_residual = short_residual
        self._long_residual = long_residual
        self._growth = growth
        self._bottleneck = bottle_neck
        self._relu = True if activation=='relu' else False
        self._elu = True if activation=='elu' else False
        self._X = tf.placeholder(
            tf.float32, [batch_size, input_height, input_width, input_channels])
        self._Y = tf.placeholder(tf.float32,
                                 [batch_size, input_height, input_width, n_classes])
        self._threshold = tf.placeholder(tf.float32, [])
        self._is_training = tf.placeholder(tf.bool, [])
        self._dropout_prob = tf.placeholder(tf.float32, [])
        self._params = {}
        if class_weights is None:
            self._class_weights = [0.1] + [10.] * (self._n_classes - 1)
        else:
            self._class_weights = np.array(class_weights)
        self._params['global_step'] = tf.Variable(
            0, trainable=False, name='global_step')
        self._build_model()
        self._saver = tf.train.Saver(self._params)

    def _dense_block(self,
                    input_tensor,
                    depth,
                    growth,
                    param_dict,
                    block_name='DenseBlock',
                    strides=(1,1),
                    filter_size=3):
        """
        Creating dense block as given in Tiramisu paper by Bengio
        """
        outputs = []
        for i in range(depth):
            output_tensor = self.conv2D_layer(input_tensor,
                                              filter_size=filter_size,
                                              n_filters=growth,
                                              relu=self._relu,
                                              elu=self._elu,
                                              batch_norm=True,
                                              dropout=False,
                                              param_dict=param_dict,
                                              strides=strides,
                                              layer_name=block_name+'conv'+str(i+1))
            outputs.append(output_tensor)
            if not i == (depth - 1):
                input_tensor = tf.concat([input_tensor,output_tensor], axis=-1)
        return tf.concat(outputs, axis=-1)

    def _transition_down(self,
                         input_tensor,
                         param_dict,
                         block_name='TransitionDown'):
        input_tensor = self.conv2D_layer(input_tensor,
                                    filter_size=1,
                                    n_filters=int(input_tensor.shape[-1]),
                                    relu=self._relu,
                                    elu=self._elu,
                                    batch_norm=True,
                                    dropout=False,
                                    layer_name=block_name,
                                    param_dict=param_dict)
        return self.maxpool_layer(input_tensor)

    def _transition_up(self,
                       input_tensor,
                       output_shape,
                       param_dict,
                       batch_size=1,
                       block_name='TransitionUp',
                       padding='SAME'):
        return self.upconv2D_layer(input_tensor,
                                   filter_size=3,
                                   n_filters=int(input_tensor.shape[-1]),
                                   output_shape=output_shape,
                                   batch_size=batch_size,
                                   param_dict=param_dict,
                                   layer_name=block_name,
                                   padding=padding)

    def _regression_block(self,
                          input_tensor,
                          param_dict,
                          n_reg=1,
                          n_conv=2,
                          block_name='RegressionBlock'):
        for i in range(n_conv):
            input_tensor = self.conv2D_layer(input_tensor,
                                         filter_size=3,
                                         n_filters=int(int(input_tensor.shape[-1])/2),
                                         relu=True,
                                         batch_norm=True,
                                         dropout=True,
                                         layer_name=block_name+'conv'+str(i),
                                         param_dict=param_dict,
                                         strides=(1,1),
                                         padding='VALID')
        input_tensor = tf.reduce_sum(input_tensor, axis=[1,2])
        return self.fully_connected_layer(input_tensor,
                                                  output_dim = n_reg,
                                                  relu=True,
                                                  batch_norm=True,
                                                  dropout=True,
                                                  layer_name=block_name+'Output',
                                                  param_dict=param_dict)

    def visualize_tensor(self, input_tensor):
        input_tensor = tf.argmax(input_tensor, axis=-1)
        input_tensor = tf.expand_dims(input_tensor, axis=-1)
        input_tensor = (255*input_tensor)/(self._n_classes-1)
        input_tensor = tf.round(input_tensor)
        input_tensor = tf.cast(input_tensor, tf.uint8)
        return input_tensor

    def _build_model(self):
        param_dict = self._params
        stack = self.conv2D_layer(self._X,
                                  filter_size=self._filters,
                                  n_filters=48,
                                  relu=self._relu,
                                  elu=self._elu,
                                  batch_norm=True,
                                  layer_name='ConvLayer',
                                  param_dict=param_dict)
        combine_layers = []
        layer_sizes = []
        for i,depth in enumerate(self._layer_depths):
            dense_block = self._dense_block(stack,
                                         depth=depth,
                                         growth=self._growth,
                                         param_dict=param_dict,
                                         block_name='DenseBlock{}'.format(i),
                                         filter_size=self._filters)
            print(dense_block.shape)
            layer_sizes.append(int(dense_block.shape[2]))
            if self._short_residual:
                bypass = self.conv2D_layer(stack,
                                          filter_size=1,
                                          n_filters=int(dense_block.shape[-1]),
                                          relu=False,
                                          batch_norm=True,
                                          dropout=False,
                                          layer_name='Bypass{}'.format(i),
                                          param_dict=param_dict,
                                          strides=(1,1),
                                          padding='SAME')
                combine = dense_block + bypass
            else:
                combine = tf.concat([stack, dense_block], axis=-1)
            combine_layers.append(combine)
            stack = self._transition_down(combine,param_dict=param_dict,
                                        block_name='TransitionDown{}'.format(i))
            print(stack.shape)
        combine_layers = combine_layers[::-1]
        layer_sizes = layer_sizes[::-1]
        layer_depths = self._layer_depths[::-1]

        stack = self._dense_block(stack,
                                    depth=self._bottleneck,
                                    growth=16,
                                    param_dict=param_dict,
                                    block_name='BottleNeck',
                                    filter_size=self._filters)
        print(stack.shape)
        # volumes = self._regression_block(stack,
                                        # param_dict=param_dict,
                                        # n_reg=self._n_classes,
                                        # block_name='RegeressionBlock')

        for i, (combine, layer_size, depth) in enumerate(zip(combine_layers, layer_sizes, layer_depths)):
            padding = 'SAME' if 2*int(stack.shape[2])==layer_size else 'VALID'
            transition_up = self._transition_up(stack,
                                             output_shape=(layer_size,layer_size),
                                             param_dict=param_dict,
                                             batch_size=self._batch_size,
                                             block_name='TransitionUp{}'.format(i),
                                             padding=padding)
            print(transition_up.shape)
            if self._long_residual:
                bypass = self.conv2D_layer(combine,
                                          filter_size=1,
                                          n_filters=int(transition_up.shape[-1]),
                                          relu=False,
                                          batch_norm=True,
                                          dropout=False,
                                          layer_name='BypassUp{}'.format(i),
                                          param_dict=param_dict,
                                          strides=(1,1),
                                          padding='SAME')
                combine_up = bypass + transition_up
            else:
                combine_up = tf.concat([combine, transition_up], axis=-1)
            stack = self._dense_block(combine_up,
                                            depth=depth,
                                            growth=self._growth,
                                            param_dict=param_dict,
                                            block_name='DenseBlockUp{}'.format(i),
                                            filter_size=self._filters)
            print(stack.shape)

        self._logits = self.conv2D_layer(stack,
                                   filter_size=self._output_filters,
                                   n_filters=self._n_classes,
                                   dropout=True,
                                   relu=self._relu,
                                   elu=self._elu,
                                   batch_norm=False,
                                   layer_name='OutputConv',
                                   param_dict=param_dict)

        self._soft_prob = tf.nn.softmax(self._logits, dim=-1)
        # self._dice_loss = self.soft_dice_loss(self._logits,
        #                                       self._Y,
        #                                       index=-1,
        #                                       weights=self._class_weights)
        output_vis = self.visualize_tensor(self._soft_prob)
        gt_vis = self.visualize_tensor(self._Y)
        self._entropy_loss = self.weighted_softmax_cross_entropy_with_logits(
                                                            self._logits,
                                                            self._Y,
                                                            self._class_weights,
                                                            axis=-1)
        learning_rate = tf.train.exponential_decay(5e-4,
                                                   self._params['global_step'],
                                                   1000,
                                                   0.90,
                                                   staircase=True)
        self._cost = self._entropy_loss
        self._cost_summary = tf.summary.scalar('Cost', self._cost)

        self._train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                beta1=0.99,
                                                beta2=0.995).minimize(self._cost, global_step = self._params['global_step'])
        self._predictions = tf.stop_gradients(tf.argmax(self._logits, -1))
        self._output_summary = tf.summary.image('Output', output_vis)
        self._input_summary = tf.summary.image('Ground Truth', gt_vis)
        self._image_summary = tf.summary.merge([self._output_summary, self._input_summary])
        folder_dir = os.path.split(self._saved)[0]
        folder_dir = os.path.split(folder_dir)[0]
        self.log_dir = os.path.join(folder_dir,'logs')
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir)
        else:
            os.makedirs(self.log_dir)
        self._summary_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())
        ################
        #Reinforcement Learning nodes

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                beta1=0.99,
                                                beta2=0.995) 
        self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, labels=self._predictions)
        self.pg_loss = tf.reduce_mean(self.cross_entropy_loss)
        self.gradients = self.optimizer.compute_gradients(self.pg_loss)
        self.discounted_rewards = self.stop_gradients(self.soft_dice_loss(logits=self._logits, labels=self._Y) - self._threshold)
        for i, (grad, var) in enumerate(self.gradients):
            if grad is not None:
                self.gradients[i] = (grad * self.discounted_rewards, var)
        self._reinforce_op = self.optimizer.apply_gradients(self.gradients)
        
    def train_net(self,
                  data,
                  validation=None,
                  plot=True,
                  restore=False,
                  n_epochs=5):
        self._sess.run(tf.global_variables_initializer())
        # best_score = {'TC':0.0, 'AT':0.0, 'WT':0.0}
        if restore:
            self._restore_weights_new(self._saved.format(self._best_score_string))
        for n in range(n_epochs):
            history = []
            patients = {}
            print('Epoch number : {}'.format(n))
            for i, batch in enumerate(tqdm(data())):

                cost_summary, _, pred = self._sess.run(
                    [self._cost_summary, self._train_op, self._predictions],
                    feed_dict={
                        self._X: batch['X'],
                        self._Y: batch['Y'],
                        self._dropout_prob: 0.5,
                        self._is_training:True
                    })
                self._summary_writer.add_summary(cost_summary, (n+1)*(i+1))
                if (i+1)%20 == 0:
                    image_summary, = self._sess.run([self._image_summary], feed_dict={
                        self._X: batch['X'],
                        self._Y:batch['Y'],
                        self._dropout_prob: 1.0,
                        self._is_training:False
                    })
                    self._summary_writer.add_summary(image_summary, i)
                for split_i,split_name in enumerate(batch['name']):
                    results = self._dice_score(pred[split_i],
                                               np.argmax(batch['Y'][split_i], axis=-1),
                                               classes = {'WT':[2,3,4],
                                                          'TC':[2,4],
                                                          'AT':[4]},
                                               dice=False)
                    if not split_name in patients.keys():
                        patients[split_name]={x :np.array([0.0,0.0]) for x in results.keys()}
                    for key in patients[split_name].keys():
                        patients[split_name][key]+=results[key]

            overall_results = {x :0.0 for x in results.keys()}
            for key in overall_results.keys():
                overall_results[key] = np.mean([patients[patient_name][key][0]/patients[patient_name][key][1] for patient_name in patients.keys()])
            print("Training Dice Scores :{}".format(overall_results))
            if validation is None:
                self._save_weights(self._saved)
            elif (n) % 1 == 0:
                print("Validating Model")
                scores = self.score(validation, restore=False)
                print("Validation Score\nBefore CRF:{}\nAfter CRF:{}".format(scores['overall'], scores['crf']))
                score = scores['overall']
                if np.all([score[key]>=self._best_score[key] for key in self._best_score.keys()]):
                    self._best_score = score
                    self._save_weights(self._saved)
                    
              
    def reinforce(self,
                  data,
                  validation=None,
                  restore=False,
                  plot=True,
                  n_epochs=5):
        self._sess.run(tf.global_variables_initializer())
        # best_score = {'TC':0.0, 'AT':0.0, 'WT':0.0}
        if restore:
            self._restore_weights_new(self._saved.format(self._best_score_string))
        for n in range(n_epochs):
            history = []
            patients = {}
            print('Epoch number : {}'.format(n))
            for i, batch in enumerate(tqdm(data())):

                cost_summary, _, pred = self._sess.run(
                    [self._reinforce_cost_summary, self._reinforce_op, self._predictions],
                    feed_dict={
                        self._X: batch['X'],
                        self._Y: batch['Y'],
                        self._dropout_prob: 0.5,
                        self._is_training:True,
                        self._threshold:0.75
                    })
                self._summary_writer.add_summary(cost_summary, (n+1)*(i+1))
                if (i+1)%20 == 0:
                    image_summary, = self._sess.run([self._image_summary], feed_dict={
                        self._X: batch['X'],
                        self._Y:batch['Y'],
                        self._dropout_prob: 1.0,
                        self._is_training:False
                    })
                    self._summary_writer.add_summary(image_summary, i)
                for split_i,split_name in enumerate(batch['name']):
                    results = self._dice_score(pred[split_i],
                                               np.argmax(batch['Y'][split_i], axis=-1),
                                               classes = {'WT':[2,3,4],
                                                          'TC':[2,4],
                                                          'AT':[4]},
                                               dice=False)
                    if not split_name in patients.keys():
                        patients[split_name]={x :np.array([0.0,0.0]) for x in results.keys()}
                    for key in patients[split_name].keys():
                        patients[split_name][key]+=results[key]

            overall_results = {x :0.0 for x in results.keys()}
            for key in overall_results.keys():
                overall_results[key] = np.mean([patients[patient_name][key][0]/patients[patient_name][key][1] for patient_name in patients.keys()])
            print("Training Dice Scores :{}".format(overall_results))
            if validation is None:
                self._save_weights(self._saved)
            elif (n) % 1 == 0:
                print("Validating Model")
                scores = self.score(validation, restore=False)
                print("Validation Score\nBefore CRF:{}\nAfter CRF:{}".format(scores['overall'], scores['crf']))
                score = scores['overall']
                if np.all([score[key]>=self._best_score[key] for key in self._best_score.keys()]):
                    self._best_score = score
                    self._save_weights(self._saved)
    
    def score(self,
              data,
              restore=True,
              **kwargs):
        if restore:
            self._sess.run(tf.global_variables_initializer())
            self._restore_weights_new(self._saved)
        patients = {}
        crf_patients = {}
        for j, batch in enumerate(tqdm(data())):
            pred, probs = self._sess.run([self._predictions, self._soft_prob],
                                           feed_dict={
                                               self._X: batch['X'],
                                               self._dropout_prob: 1.0,
                                               self._is_training:False
                                           })


            for i,n in enumerate(batch['name']):
                #crf_prob, crf_labels = self.CRF(batch['X'][i,:,:,:], probs[i,:,:,:], **kwargs)
                #crf_results = self._dice_score(crf_labels,np.argmax(batch['Y'][i], axis=1),dice=False)
                results = self._dice_score(pred[i],
                                           np.argmax(batch['Y'][i], axis=-1),
                                           classes = {'WT':[2,3,4],
                                                      'TC':[2,4],
                                                      'AT':[4]},
                                           dice=False)
                if not n in patients.keys():
                    patients[n]={x :np.array([0.0,0.0]) for x in results.keys()}
                #    crf_patients[n]={x :np.array([0.0,0.0]) for x in results.keys()}
                for key in patients[n].keys():
                    patients[n][key]+=results[key]
                #    crf_patients[n][key]+=crf_results[key]
        overall_results = {x :0.0 for x in results.keys()}
        #crf_overall_results = {x :0.0 for x in results.keys()}
        for key in overall_results.keys():
            overall_results[key] = np.mean([patients[n][key][0]/patients[n][key][1] for n in patients.keys()])
            #crf_overall_results[key] = np.mean([crf_patients[n][key][0]/crf_patients[n][key][1] for n in crf_patients.keys()])
        return {'overall':overall_results,
                'crf': None}
    
    def hard_mine(self,
                  data,
                  validation,
                  **kwargs):
        self._sess.run(tf.global_variables_initializer())
        model_loc = self._saved.format(self._best_score_string)
        model_name = os.path.split(model_loc)[-1]
        self._restore_weights(self.saved)
        hard_mined_slices = {'patients':[], 'slice':[]}
        for j, batch in enumerate(tqdm(data(skip=False))):
            pred, probs = self._sess.run([self._predictions, self._soft_prob],
                                           feed_dict={
                                               self._X: batch['X'],
                                               self._dropout_prob: 1.0,
                                               self._is_training:False
                                           })
            for i, index in enumerate(batch['index']):
                dice_score = self._dice_score(pred[i],np.argmax(batch['Y'][i], axis=-1), dice=False)
                for key, item in dice_score.items():
                    if item[0]/item[1] < 0.8 and item[1]>10:
                        cost, = self._sess.run([self._train_op],
                                        feed_dict={
                                            self._X: batch['X'][[i]],
                                            self._Y: batch['Y'][[i]],
                                            self._dropout_prob: 0.8,
                                            self._is_training: True
                                        })
                        hard_mined_slices['patients'].append(batch['name'][i])
                        hard_mined_slices['slice'].append(index)
            if (j+1)%2000==0:
                data = pd.DataFrame.from_dict(hard_mined_slices)
                data.to_csv('{}_hard_slices.csv'.format(model_name))
                print("Validating Model")
                score, crf_score = self.score(validation, restore=False)
                print("Validation Score\nBefore CRF:{}\nAfter CRF:{}".format(score, crf_score))
                # count = 0
                # for key in score.keys():
                #     count += 1*(best_score[key]<score[key])
                score_string = '_'.join('{}:{:.2f}'.format(x, score[x]) for x in score.keys())
                self._save_weights(self._saved)
                self._best_score_string = score_string


    def predict(self,
                data,
                fileformat='pred',
                **kwargs):
        self._sess.run(tf.global_variables_initializer())
        model_loc = self._saved.format(self._best_score_string)
        model_name = os.path.split(model_loc)[-1]
        self._restore_weights_new(model_loc)
        patients = {}

        for j, batch in enumerate(tqdm(data())):
            pred, probs = self._sess.run([self._predictions, self._soft_prob],
                                           feed_dict={
                                               self._X: batch['X'],
                                               self._dropout_prob: 1.0,
                                               self._is_training:False
                                           })
            for i,n in enumerate(batch['name']):
                crf_prob, crf_labels = self.CRF(batch['X'][i,:,:,:], probs[i,:,:,:], **kwargs)
                if n not in patients.keys():
                    patients[n]={'probs':[], 'labels':[], 'affine':batch['affine'][i]}
                patients[n]['labels'].append(self.correct_labels(crf_labels))
                patients[n]['probs'].append(crf_prob)
            for n in patients.keys():
                if len(patients[n]['labels'])==155:
                    vox=np.stack(patients[n]['labels'], axis=-1)
                    cleaned_vox, mask = self.connected_components(vox)
                    probs = np.stack(patients[n]['probs'], axis=-2).astype(np.float16)
                    #tqdm.write(vox.shape)
                    if fileformat=='pred' or fileformat=='both':
                        nii_img = nib.Nifti1Image(cleaned_vox, patients[n]['affine'])
                        nii_img.set_data_dtype(np.uint8)
                        if model_name+'_val' not in os.listdir('results/'):
                            os.mkdir(os.path.join('results',model_name+'_val'))
                        nib.save(nii_img,'results/{}/{}.nii.gz'.format(model_name+'_val',n))
                    if fileformat=='prob' or fileformat=='both':
                        if model_name not in os.listdir('probabilities/'):
                            os.mkdir(os.path.join('probabilities',model_name))
                        np.savez('probabilities/{}/{}.npz'.format(model_name,n), probs=probs, affine=patients[n]['affine'])
                    del patients[n]
