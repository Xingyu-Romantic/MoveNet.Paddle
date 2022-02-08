import paddle
import numpy as np



_img_size = 192
_feature_map_size = _img_size//4

_center_weight_path = 'lib/data/center_weight_origin.npy'


class JointBoneLoss(paddle.nn.Layer):
    def __init__(self, joint_num):
        super(JointBoneLoss, self).__init__()
        id_i, id_j = [], []
        for i in range(joint_num):
            for j in range(i+1, joint_num):
                id_i.append(i)
                id_j.append(j)
        self.id_i = id_i
        self.id_j = id_j

        # self.id_i = [0,1,2,3,4,5,2]
        # self.id_j = [1,2,3,4,5,6,4]


    def forward(self, joint_out, joint_gt):
        J = paddle.linalg.norm(joint_out[:,self.id_i,:] - joint_out[:,self.id_j,:], p=2, axis=-1, keepdim=False)
        Y = paddle.linalg.norm(joint_gt[:,self.id_i,:] - joint_gt[:,self.id_j,:], p=2, axis=-1, keepdim=False)
        loss = paddle.abs(J-Y)
        # loss = loss.mean()
        loss = paddle.sum(loss)/joint_out.shape[0]/len(self.id_i)
        return loss

class MovenetLoss(paddle.nn.Layer):
    def __init__(self, use_target_weight=False, target_weight=[1]):
        super(MovenetLoss, self).__init__()
        self.mse = paddle.nn.MSELoss('mean')
        self.use_target_weight = use_target_weight
        self.target_weight=target_weight

        self.center_weight = paddle.to_tensor(np.load(_center_weight_path))
        self.make_center_w = False

        # self.range_weight_x = paddle.from_numpy(np.array([[x for x in range(48)] for _ in range(48)]))
        # self.range_weight_y = self.range_weight_x.T 

        self.boneloss = JointBoneLoss(17)


    def l1(self, pre, target,kps_mask):
        # print("1 ",pre.shape, pre.device)
        # print("2 ",target.shape, target.device)
        # b

        # return paddle.mean(paddle.abs(pre - target)*kps_mask)
        return paddle.sum(paddle.abs(pre - target)*kps_mask)/ (kps_mask.sum() + 1e-4)

    def l2_loss(self, pre, target):
        loss = (pre - target) 
        loss = (loss * loss) / 2 / pre.shape[0]

        return loss.sum()



    def centernetfocalLoss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        Arguments:
          pred (batch x c x h x w)
          gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = paddle.pow(1 - gt, 4)

        loss = 0

        pos_loss = paddle.log(pred) * paddle.pow(1 - pred, 2) * pos_inds
        neg_loss = paddle.log(1 - pred) * paddle.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


    def myMSEwithWeight(self, pre, target):
        #target 0-1
        # pre = paddle.sigmoid(pre)
        # print(paddle.max(pre), paddle.min(pre))
        # b
        loss = paddle.pow((pre-target),2)
        # loss = paddle.abs(pre-target)

        #weight_mask = (target+0.1)/1.1
        weight_mask = target*8+1
        # weight_mask = paddle.pow(target,2)*8+1

        #gamma from focal loss
        #gamma = paddle.pow(paddle.abs(target-pre), 2)

        loss = loss*weight_mask#*gamma

        loss = paddle.sum(loss)/target.shape[0]/target.shape[1]

        # bg_loss = self.bgLoss(pre, target)
        return loss

    def heatmapL1(self, pre, target):
        #target 0-1
        # pre = paddle.sigmoid(pre)
        # print(paddle.max(pre), paddle.min(pre))
        # b
        loss = paddle.abs(pre-target)
        

        #weight_mask = (target+0.1)/1.1
        weight_mask = target*4+1

        #gamma from focal loss
        #gamma = paddle.pow(paddle.abs(target-pre), 2)

        loss = loss*weight_mask#*gamma

        loss = paddle.sum(loss)/target.shape[0]/target.shape[1]
        return loss


    ###############
    def boneLoss(self, pred, target):
        #[64, 7, 48, 48]
        def _Frobenius(mat1, mat2):
            return paddle.pow(paddle.sum(paddle.pow(mat1-mat2,2)),0.5)
            # return paddle.sum(paddle.pow(mat1-mat2,2))


        _bone_idx = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[2,4]]

        loss = 0
        for bone_id in _bone_idx:
            bone_pre = pred[:,bone_id[0],:,:]-pred[:,bone_id[1],:,:]
            bone_gt = target[:,bone_id[0],:,:]-target[:,bone_id[1],:,:]

            f = _Frobenius(bone_pre,bone_gt)
            loss+=f

        loss = loss/len(_bone_idx)/pred.shape[0]
        return loss


    def bgLoss(self, pre, target):
        ##[64, 7, 48, 48]

        bg_pre = paddle.sum(pre, axis=1)
        bg_pre = 1-paddle.clamp(bg_pre, 0, 1)

        bg_gt = paddle.sum(target, axis=1)
        bg_gt = 1-paddle.clamp(bg_gt, 0, 1)

        #weight_mask = (1-bg_gt)*4+1

        loss = paddle.sum(paddle.pow((bg_pre-bg_gt),2))/pre.shape[0]

        return loss

    def heatmapLoss(self, pred, target, batch_size):
        #[64, 7, 48, 48]
        # print(pred.shape, target.shape)

        # heatmaps_pred = pred.reshape((batch_size, pred.shape[1], -1)).split(1, 1)
        # #对tensor在某一dim维度下，根据指定的大小split_size=int，或者list(int)来分割数据，返回tuple元组
        # #print(len(heatmaps_pred), heatmaps_pred[0].shape)#7 paddle.Size([64, 1, 48*48]
        # heatmaps_gt = target.reshape((batch_size, pred.shape[1], -1)).split(1, 1)

        # loss = 0

        # for idx in range(pred.shape[1]):
        #     heatmap_pred = heatmaps_pred[idx].squeeze()#[64, 40*40]
        #     heatmap_gt = heatmaps_gt[idx].squeeze()
        #     if self.use_target_weight:
        #         loss += self.centernetfocalLoss(
        #                         heatmap_pred.mul(self.target_weight[idx//2]),
        #                         heatmap_gt.mul(self.target_weight[idx//2])
        #                     )
        #     else:

        #         loss += self.centernetfocalLoss(heatmap_pred, heatmap_gt)
        # loss /= pred.shape[1]

        return self.myMSEwithWeight(pred,target) 
             

    def centerLoss(self, pred, target, batch_size):
        # heatmaps_pred = pred.reshape((batch_size, -1))
        # heatmaps_gt = target.reshape((batch_size, -1))
        return self.myMSEwithWeight(pred, target) 


    def regsLoss(self, pred, target, cx0, cy0,  kps_mask, batch_size, num_joints):
        #[64, 14, 48, 48]
        # print(target.shape, cx0.shape, cy0.shape)#paddle.Size([64, 14, 48, 48]) paddle.Size([64]) paddle.Size([64])
        
        _dim0 = paddle.cast(paddle.arange(0,batch_size), dtype=paddle.int64)
        _dim1 = paddle.zeros([batch_size], dtype=paddle.int64)

        #print("regsLoss: " , cx0,cy0)
        # print(target.shape)#paddle.Size([1, 14, 48, 48])
        # print(paddle.max(target[0][2]), paddle.min(target[0][2]))
        # print(paddle.max(target[0][3]), paddle.min(target[0][3]))

        # cv2.imwrite("t.jpg", target[0][2].cpu().numpy()*255)
        loss = 0
        for idx in range(num_joints):

            gt_x = target[_dim0,_dim1+idx*2,cy0,cx0]
            gt_y = target[_dim0,_dim1+idx*2+1,cy0,cx0]
            
            
            pre_x = pred[_dim0,_dim1+idx*2,cy0,cx0]
            pre_y = pred[_dim0,_dim1+idx*2+1,cy0,cx0]

            # print(paddle.max(target[_dim0,_dim1+idx*2,:,:]),paddle.min(target[_dim0,_dim1+idx*2,:,:]))
            # print(gt_x,pre_x)                                       
            # print(gt_y,pre_y)
            

            # print(kps_mask[:,idx])
            # print(gt_x,pre_x)
            # print(self.l1(gt_x,pre_x,kps_mask[:,idx]))
            # print('---')
            # 

            loss+=self.l1(gt_x,pre_x,kps_mask[:,idx])
            loss+=self.l1(gt_y,pre_y,kps_mask[:,idx])
        #b
        # offset_x_pre = paddle.clip(pre_x,0,_feature_map_size-1).long()
        # offset_y_pre = paddle.clip(pre_y,0,_feature_map_size-1).long()
        # offset_x_gt = paddle.clip(gt_x+cx0,0,_feature_map_size-1).long()
        # offset_y_gt = paddle.clip(gt_y+cy0,0,_feature_map_size-1).long()

        return loss / num_joints


    def offsetLoss(self, pred, target,  cx0, cy0, regs, kps_mask, batch_size, num_joints):
        _dim0 = paddle.cast(paddle.arange(0,batch_size), dtype=paddle.int64)
        _dim1 = paddle.zeros([batch_size], dtype=paddle.int64)
        loss = 0
        # print(gt_y,gt_x)
        for idx in range(num_joints):
            gt_x = paddle.cast(regs[_dim0,_dim1+idx*2,cy0,cx0],dtype=paddle.float32)+cx0
            gt_y = paddle.cast(regs[_dim0,_dim1+idx*2+1,cy0,cx0], dtype=paddle.float32)+cy0

            gt_x[gt_x>47]=47
            gt_x[gt_x<0]=0
            gt_y[gt_y>47]=47
            gt_y[gt_y<0]=0

            gt_x = paddle.cast(gt_x, dtype=paddle.int64)
            gt_y = paddle.cast(gt_y, dtype=paddle.int64)

            gt_offset_x = target[_dim0,_dim1+idx*2,gt_y,gt_x]
            gt_offset_y = target[_dim0,_dim1+idx*2+1,gt_y,gt_x]

            pre_offset_x = pred[_dim0,_dim1+idx*2,gt_y,gt_x]
            pre_offset_y = pred[_dim0,_dim1+idx*2+1,gt_y,gt_x]

            # print(gt_offset_x, paddle.max(target[_dim0,_dim1+idx*2,...]),paddle.min(target[_dim0,_dim1+idx*2,...]))
            # print(gt_offset_y, paddle.max(target[_dim0,_dim1+idx*2+1,...]),paddle.min(target[_dim0,_dim1+idx*2+1,...]))
            loss+=self.l1(gt_offset_x,pre_offset_x,kps_mask[:,idx])
            loss+=self.l1(gt_offset_y,pre_offset_y,kps_mask[:,idx])
        #     print(gt_y,gt_x)    
        # b
        return loss / num_joints

        """
        0.0 0.5
        0.0 0.75
        0.75 0.25
        0.0 0.75
        0.0 0.5
        """


    def maxPointPth(self, heatmap, center=True):
        #pypaddle version
        # n,1,h,w
        # 计算center heatmap的最大值得到中心点
        if center:
            heatmap = heatmap*self.center_weight[:heatmap.shape[0],...]
            #加权取最靠近中间的

        n,c,h,w = heatmap.shape
        heatmap = heatmap.reshape((n, -1)) #64, 48x48
        # print(heatmap[0])
        max_id = paddle.argmax(heatmap, 1)#64, 1
        # max_v,max_id = paddle.max(heatmap, axis = 1)#64, 1
        # max_v,max_id
        # print(max_v)
        # print("max_i: ",max_i)

        # mask0 = paddle.zeros(max_v.shape).to(heatmap.device)
        # mask1 = paddle.ones(max_v.shape).to(heatmap.device)
        # mask = paddle.where(paddle.gt(max_v,th), mask1, mask0)
        # print(mask)
        # b
        y = max_id//w
        x = max_id%w

        return x,y


    def forward(self, output, target, kps_mask):
        batch_size = output[0].shape[0]
        num_joints = output[0].shape[1]
        #print("output: ", [x.shape for x in output])
        #[64, 7, 48, 48] [64, 1, 48, 48] [64, 14, 48, 48] [64, 14, 48, 48]
        # print("target: ", [x.shape for x in target])#[64, 36, 48, 48]
        #print(weights.shape)# [14,]
        heatmaps = target[:,:17,:,:]
        centers = target[:,17:18,:,:]
        regs = target[:,18:52,:,:]
        offsets = target[:,52:,:,:]


        heatmap_loss = self.heatmapLoss(output[0], heatmaps, batch_size)

        # bg_loss = self.bgLoss(output[0], heatmaps)
        #bone_loss = self.boneloss(output[0], heatmaps)
        bone_loss = self.boneLoss(output[0], heatmaps)
        #print(heatmap_loss)
        center_loss = self.centerLoss(output[1], centers, batch_size)


        if not self.make_center_w:
            self.center_weight = paddle.reshape(self.center_weight,(1,1,48,48))
            self.center_weight = self.center_weight.tile((output[1].shape[0],output[1].shape[1],1,1))
            # print(self.center_weight.shape)
            # b
            self.center_weight = self.center_weight
            self.make_center_w = True
            self.center_weight.stop_gradient = True

            # self.range_weight_x = self.range_weight_x.to(target.device)
            # self.range_weight_y = self.range_weight_y.to(target.device)
            # self.range_weight_x.requires_grad_(False)
            # self.range_weight_y.requires_grad_(False)
        #print(self.center_weight)


        cx0, cy0 = self.maxPointPth(centers)
        # cx1, cy1 = self.maxPointPth(pre_centers)
        cx0 = paddle.cast(paddle.clip(cx0,0,_feature_map_size-1), dtype=paddle.int64)
        cy0 = paddle.cast(paddle.clip(cy0,0,_feature_map_size-1), dtype=paddle.int64)
        # cx1 = paddle.clip(cx1,0,_feature_map_size-1).long()
        # cy1 = paddle.clip(cy1,0,_feature_map_size-1).long()

        # print(cx0, cy0)
        # bbb
        # cv2.imwrite("_centers.jpg", centers[0][0].cpu().numpy()*255)
        # b

        regs_loss = self.regsLoss(output[2], regs, cx0, cy0, kps_mask,batch_size, num_joints)
        offset_loss = self.offsetLoss(output[3], offsets, 
                            cx0, cy0,regs,
                            kps_mask,batch_size, num_joints)
        
        # total_loss = heatmap_loss+center_loss+0.1*regs_loss+offset_loss
        # print(heatmap_loss,center_loss,regs_loss,offset_loss)
        # b
        
        """
        
        """ 
        # boneloss = self.boneLoss(output[3], offsets, 
        #                     cx0, cy0,regs,
        #                     kps_mask,batch_size, num_joints)


        return [heatmap_loss,bone_loss,center_loss,regs_loss,offset_loss]

movenetLoss = MovenetLoss(use_target_weight=False)


def calculate_loss(predict, label):
    loss = movenetLoss(predict, label)
    return loss 


