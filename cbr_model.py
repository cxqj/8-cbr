import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

import vs_multilayer
from dataset import TestingDataSet
from dataset import TrainingDataSet


class CBR_Model(object):
    def __init__(self, batch_size, ctx_num, unit_size, unit_feature_size, action_class_num, lr, lambda_reg, train_clip_path, background_path, test_clip_path, train_flow_feature_dir, train_appr_feature_dir, test_flow_feature_dir, test_appr_feature_dir):
        
        self.batch_size = batch_size
        self.test_batch_size = 1  # 测试的时候batch = 1个clip,训练的时候batcch = 128
        self.middle_layer_size = 1000
        self.vs_lr = lr
        self.lambda_reg = lambda_reg  # 1.0
        self.action_class_num = action_class_num  # 20
        self.visual_feature_dim = unit_feature_size*3  # 4096*3
        self.train_set = TrainingDataSet(train_flow_feature_dir, train_appr_feature_dir, train_clip_path, background_path, batch_size, ctx_num, unit_size, unit_feature_size, action_class_num)
        self.test_set = TestingDataSet(test_flow_feature_dir, test_appr_feature_dir, test_clip_path, self.test_batch_size, unit_size)
   
    	    
    def fill_feed_dict_train(self):
        image_batch, label_batch, offset_batch, one_hot_label_batch = self.train_set.next_batch()
        input_feed = {
                self.visual_featmap_ph_train: image_batch,  # 加载的inter特征和上下文特征
                self.label_ph: label_batch,
                self.offset_ph: offset_batch,
                self.one_hot_label_ph: one_hot_label_batch
        }
        return input_feed

    # 计算总的loss和回归loss  
    # vs_multilayer由两个全连接层构成，将该clip对应的特征输入，生成用于分类和回归的特征
    def compute_loss_reg(self, visual_feature, offsets, labels, one_hot_labels):

        cls_reg_vec = vs_multilayer.vs_multilayer(visual_feature, "CBR", middle_layer_dim=self.middle_layer_size, output_layer_dim=(self.action_class_num+1)*3)
        cls_reg_vec = tf.reshape(cls_reg_vec, [self.batch_size, (self.action_class_num+1)*3]) # [128,21*3]
        cls_score_vec = cls_reg_vec[:, :self.action_class_num+1]
        start_offset_pred = cls_reg_vec[:, self.action_class_num+1:(self.action_class_num+1)*2]
        end_offset_pred = cls_reg_vec[:, (self.action_class_num+1)*2:]

        #classification loss
        loss_cls_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(cls_score_vec, labels)
        loss_cls = tf.reduce_mean(loss_cls_vec)
        
        
        # regression loss
        pick_start_offset_pred = []
        pick_end_offset_pred = []
        # 选取第K个sampel的属于某个类别的回归值。参考论文中的回归计算
        for k in range(self.batch_size):   # 选取第K个sample的回归预测值
            pick_start_offset_pred.append(start_offset_pred[k, labels[k]])
            pick_end_offset_pred.append(end_offset_pred[k, labels[k]])
        pick_start_offset_pred = tf.reshape(tf.stack(pick_start_offset_pred),[self.batch_size, 1])
        pick_end_offset_pred = tf.reshape(tf.stack(pick_end_offset_pred), [self.batch_size, 1])
        labels_1 = tf.to_float(tf.not_equal(labels,0))   # 选取对应的类别的回归值,labels中保存的是该sample属于哪个类别
        label_tmp = tf.to_float(tf.reshape(labels_1, [self.batch_size, 1]))
        label_for_reg = tf.concat(1, [label_tmp, label_tmp])  # 按列进行拼接 [128,2]
        offset_pred = tf.concat(1,(pick_start_offset_pred, pick_end_offset_pred)) # [128,2]
        loss_reg = tf.reduce_mean(tf.mul(tf.abs(tf.sub(offset_pred, offsets)), label_for_reg))
        
        
        loss = tf.add(tf.mul(self.lambda_reg, loss_reg), loss_cls)
        return loss, loss_reg

    # 创建placeholder
    def init_placeholder(self):
        visual_featmap_ph_train = tf.placeholder(tf.float32, shape=(self.batch_size, self.visual_feature_dim)) # (128,12288)
        label_ph = tf.placeholder(tf.int32, shape=(self.batch_size)) # (128,)
        offset_ph = tf.placeholder(tf.float32, shape=(self.batch_size, 2)) # (128,2)
        one_hot_label_ph = tf.placeholder(tf.float32, shape=(self.batch_size, self.action_class_num+1)) # (128,21)
        visual_featmap_ph_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.visual_feature_dim)) # (1,12288)

        return visual_featmap_ph_train, visual_featmap_ph_test, label_ph, offset_ph, one_hot_label_ph
    

    # 测试，输出结果
    def eval(self, visual_feature_test):
        sim_score = vs_multilayer.vs_multilayer(visual_feature_test, "CBR", middle_layer_dim=self.middle_layer_size, output_layer_dim=(self.action_class_num+1)*3, dropout=False, reuse=True)
        sim_score = tf.reshape(sim_score, [(self.action_class_num+1)*3])
        return sim_score


    def get_variables_by_name(self, name_list): # name_list : ['CBR']
        v_list = tf.trainable_variables() #tf.trainable_variables返回的是需要训练的变量列表
        v_dict = {}
        for name in name_list:
            v_dict[name] = []
        # 遍历名为name的变量列表，将其添加到v_dict
        for v in v_list:
            for name in name_list:
                if name in v.name: v_dict[name].append(v)

        for name in name_list:
            print "Variables of <"+name+">"
            for v in v_dict[name]:
                print "    "+v.name
        return v_dict


    # 训练，现货区可训练变量，再构造优化器，最终最小化loss
    def training(self, loss):
        v_dict = self.get_variables_by_name(["CBR"])  # 获取可训练变量列表
        vs_optimizer = tf.train.AdamOptimizer(self.vs_lr, name='vs_adam')
        vs_train_op = vs_optimizer.minimize(loss, var_list=v_dict["CBR"])
        return vs_train_op


    def construct_model(self):
        #construct the network:
        self.visual_featmap_ph_train, self.visual_featmap_ph_test, self.label_ph, self.offset_ph, self.one_hot_label_ph = self.init_placeholder()
        visual_featmap_ph_train_norm = tf.nn.l2_normalize(self.visual_featmap_ph_train, dim=1)  # 按行进行L2归一化
        visual_featmap_ph_test_norm = tf.nn.l2_normalize(self.visual_featmap_ph_test, dim=1)
        # 计算分类loss和回归Loss，返回最终的总的loss和回归loss
        self.loss,loss_reg = self.compute_loss_reg(visual_featmap_ph_train_norm, self.offset_ph, self.label_ph, self.one_hot_label_ph)
        self.vs_train_op = self.training(self.loss)
        vs_eval_op = self.eval(visual_featmap_ph_test_norm)
        return self.loss, self.vs_train_op, vs_eval_op, loss_reg  #返回总的loss，训练操作，测试操作，
    


