
import numpy as np
from math import sqrt
import os
import random
import pickle


# 已经修改了类别标签从1-20
cat_index_dict={
"Background":0,
"BaseballPitch":1,
"BasketballDunk":2,
"Billiards":3,
"CleanAndJerk":4,
"CliffDiving":5,
"CricketBowling":6,
"CricketShot":7,
"Diving":8,
"FrisbeeCatch":9,
"GolfSwing":10,
"HammerThrow":11,
"HighJump":12,
"JavelinThrow":13,
"LongJump":14,
"PoleVault":15,
"Shotput":16,
"SoccerPenalty":17,
"TennisSwing":18,
"ThrowDiscus":19,
"VolleyballSpiking":20
}


class TrainingDataSet(object):
    def __init__(self, flow_feat_dir, appr_feat_dir, clip_gt_path, background_path, batch_size, ctx_num, unit_size, unit_feature_size, action_class_num):
        #it_path: image_token_file path
        self.batch_size = batch_size # 128
        print "Reading training data list from "+clip_gt_path+" and "+background_path
        self.ctx_num = ctx_num  # 2
        self.visual_feature_dim = unit_feature_size*3  # 4096 *3 是因为有两个两下文向量
        self.unit_feature_size = unit_feature_size # 4096
        self.flow_feat_dir = flow_feat_dir
        self.appr_feat_dir = appr_feat_dir
        self.training_samples = []   # 将读取的标注文件的结果存入次列表，列表中每个元素对应标注文件的一行
        self.unit_size = unit_size  #16
        self.action_class_num = action_class_num
        # 读取正样本提议存入training_sample列表中
        with open(clip_gt_path) as f:
            for l in f:
                movie_name = l.rstrip().split(" ")[0]
                clip_start = float(l.rstrip().split(" ")[1])
                clip_end = float(l.rstrip().split(" ")[2])
                gt_start = float(l.rstrip().split(" ")[3])
                gt_end = float(l.rstrip().split(" ")[4])
                # 这两步是干嘛的
                round_gt_start = np.round(gt_start/self.unit_size)*self.unit_size+1
                round_gt_end = np.round(gt_end/self.unit_size)*self.unit_size+1
                category = l.rstrip().split(" ")[5]  # 动作类别
                cat_index = cat_index_dict[category] # 动作类比对应的索引
                one_hot_label = np.zeros([self.action_class_num+1],dtype=np.float32)
                one_hot_label[cat_index] = 1.0
                self.training_samples.append((movie_name, clip_start, clip_end, gt_start, gt_end, round_gt_start, round_gt_end, cat_index, one_hot_label))
            
        print str(len(self.training_samples))+" training samples are read"
        
        positive_num = len(self.training_samples)*1.0
        
        # 读取负样本提议存入training_sample列表中
        with open(background_path) as f:
            for l in f:
                # control the number of background samples
                if random.random()>1.0*positive_num/self.action_class_num/279584: continue
                movie_name = l.rstrip().split(" ")[0]
                clip_start = float(l.rstrip().split(" ")[1])
                clip_end = float(l.rstrip().split(" ")[2])
                one_hot_label = np.zeros([self.action_class_num+1], dtype=np.float32)
                self.training_samples.append((movie_name, clip_start, clip_end, 0, 0, 0, 0, 0, one_hot_label))
        self.num_samples = len(self.training_samples)
        print str(len(self.training_samples))+" training samples are read"

    # 计算回归的坐标偏移
    def calculate_regoffset(self, clip_start, clip_end, round_gt_start, round_gt_end):
        start_offset = (round_gt_start-clip_start)/self.unit_size
        end_offset = (round_gt_end-clip_end)/self.unit_size
        return start_offset, end_offset
    
    # start = clip_start ,一个clip中又包含多个unit，每次加载一个unit的特征
    def get_pooling_feature(self, flow_feat_dir, appr_feat_dir, movie_name, start, end):
        swin_step = self.unit_size  # 在特征图上的窗口大小
        all_feat = np.zeros([0, self.unit_feature_size], dtype=np.float32)
        current_pos = start
        while current_pos<end:
            swin_start = current_pos  # 指示每个unit的起始帧
            swin_end = swin_start+swin_step
            # 加载一个unit的appearance和denseflow特征
            flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            feat = np.hstack((flow_feat, appr_feat))
            all_feat = np.vstack((all_feat, feat))
            current_pos+=swin_step
        pool_feat = np.mean(all_feat, axis=0)  # 求一个clip(包含多个unit)特征的均值
        return pool_feat
    
    def get_left_context_feature(self, flow_feat_dir, appr_feat_dir, movie_name, start, end):
        swin_step = self.unit_size
        all_feat = np.zeros([0, self.unit_feature_size], dtype=np.float32)
        count = 0
        current_pos = start
        context_ext = False
        while  count<self.ctx_num:
            swin_start = current_pos-swin_step
            swin_end = current_pos
            if os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
                flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                feat = np.hstack((flow_feat,appr_feat))
                all_feat = np.vstack((all_feat,feat))
                context_ext = True
            current_pos-=swin_step
            count+=1
        if context_ext:
            pool_feat = np.mean(all_feat,axis=0)
        else:
            pool_feat = np.zeros([self.unit_feature_size], dtype=np.float32)
        return pool_feat
    
    def get_right_context_feature(self, flow_feat_dir, appr_feat_dir, movie_name, start, end):
        swin_step = self.unit_size
        all_feat = np.zeros([0, self.unit_feature_size], dtype=np.float32)
        count = 0
        current_pos = end
        context_ext = False
        while  count<self.ctx_num:
            swin_start = current_pos
            swin_end = current_pos+swin_step
            if os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
                flow_feat = np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                appr_feat = np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                feat = np.hstack((flow_feat,appr_feat))
                all_feat = np.vstack((all_feat,feat))
                context_ext = True
            current_pos+=swin_step
            count+=1
        if context_ext:
            pool_feat = np.mean(all_feat,axis=0)
        else:
            pool_feat = np.zeros([self.unit_feature_size],dtype=np.float32)
        return pool_feat

    def next_batch(self):

        random_batch_index = random.sample(range(self.num_samples), self.batch_size)  # num_samples为总的clip的数量，每个batch=128.获取每个batch的索引
        image_batch = np.zeros([self.batch_size, self.visual_feature_dim])  # visual_feature_dim = 4096*3
        label_batch = np.zeros([self.batch_size], dtype=np.int32)
        offset_batch = np.zeros([self.batch_size,2], dtype=np.float32)
        one_hot_label_batch = np.zeros([self.batch_size, self.action_class_num+1], dtype=np.float32)
        index = 0
        # 获取一个batch的数据
        while index < self.batch_size:
            k = random_batch_index[index]
            # 获取该索引对应clip对应的在training_samples中的信息
            movie_name = self.training_samples[k][0]
            # 正样本
            if self.training_samples[k][7]!=0:  # cat_index != 0  
                clip_start = self.training_samples[k][1]
                clip_end = self.training_samples[k][2]
                round_gt_start = self.training_samples[k][5]
                round_gt_end = self.training_samples[k][6]
                start_offset, end_offset = self.calculate_regoffset(clip_start, clip_end, round_gt_start, round_gt_end)
                # 获取该clip的inter,left,right信息
                featmap = self.get_pooling_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end)
                left_feat = self.get_left_context_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end)
                right_feat = self.get_right_context_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end)
                image_batch[index,:] = np.hstack((left_feat, featmap, right_feat))  # 上下文concat后的向量
                label_batch[index] = self.training_samples[k][7]  # cat_index
                one_hot_label_batch[index,:] = self.training_samples[k][8]
                offset_batch[index,0] = start_offset
                offset_batch[index,1] = end_offset
                index+=1
            # background
            else:    
                clip_start = self.training_samples[k][1]
                clip_end = self.training_samples[k][2]
                left_feat = self.get_left_context_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end)
                right_feat = self.get_right_context_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end)
                featmap = self.get_pooling_feature(self.flow_feat_dir, self.appr_feat_dir, movie_name, clip_start, clip_end)
                image_batch[index,:] = np.hstack((left_feat, featmap, right_feat))
                label_batch[index] = 0
                one_hot_label_batch[index,:] = self.training_samples[k][8]
                offset_batch[index,0] = 0
                offset_batch[index,1] = 0
                index+=1  
        return image_batch, label_batch, offset_batch, one_hot_label_batch  # 返回一个batch(128个clip)对应的信息


class TestingDataSet(object):
    def __init__(self, flow_feat_dir, appr_feat_dir, test_clip_path, batch_size, unit_size):
        self.batch_size = batch_size
        self.flow_feat_dir = flow_feat_dir
        self.appr_feat_dir = appr_feat_dir
        print "Reading testing data list from "+test_clip_path
        self.test_samples = []
        self.unit_size = unit_size
        with open(test_clip_path) as f:
            for l in f:
                movie_name = l.rstrip().split(" ")[0]
                clip_start = float(l.rstrip().split(" ")[1])
                clip_end = float(l.rstrip().split(" ")[2])
                round_start = np.round(clip_start/self.unit_size)*self.unit_size+1
                round_end = np.round(clip_end/self.unit_size)*self.unit_size+1
                self.test_samples.append((movie_name, round_start, round_end))
        self.num_samples = len(self.test_samples)
        print "test clips number: "+str(len(self.test_samples))
        



