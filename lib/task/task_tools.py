
import paddle
import paddle.optimizer as optim
import numpy as np

from lib.utils.utils import maxPoint,extract_keypoints



_range_weight_x = np.array([[x for x in range(48)] for _ in range(48)])
_range_weight_y = _range_weight_x.T
# _reg_weight = np.load("lib/data/my_weight_reg.npy") # 99x99



def getSchedu(schedu, learning_rate):
    if 'default' in schedu:
        factor = float(schedu.strip().split('-')[1])
        patience = int(schedu.strip().split('-')[2])
        scheduler = optim.lr.ReduceOnPlateau(learning_rate, 
                    mode='max', factor=factor, patience=patience,min_lr=0.000001)
    elif 'step' in schedu:
        step_size = int(schedu.strip().split('-')[1])
        gamma = int(schedu.strip().split('-')[2])
        scheduler = optim.lr.StepDecay(learning_rate, step_size=step_size, gamma=gamma, last_epoch=-1)
    elif 'SGDR' in schedu: 
        T_0 = int(schedu.strip().split('-')[1])
        T_mult = int(schedu.strip().split('-')[2])
        scheduler = optim.lr.CosineAnnealingDecay(learning_rate,
                                                            T_max=T_mult)
    elif 'MultiStepLR' in schedu:
        milestones = [int(x) for x in schedu.strip().split('-')[1].split(',')]
        gamma = float(schedu.strip().split('-')[2])
        scheduler = optim.lr.MultiStepDecay(learning_rate, 
                                                milestones=milestones, 
                                                gamma=gamma)

    else:
        raise Exception("Unknow schedu.")

    return scheduler

def getOptimizer(optims, model, scheduler, weight_decay, clip_gradient):
    if clip_gradient:
        clip = paddle.nn.ClipGradByValue(min=-clip_gradient, max=clip_gradient)
    else:
        clip = None
    if optims=='Adam':
        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=scheduler, weight_decay=weight_decay, grad_clip=clip)
    elif optims=='SGD':
        optimizer = optim.Momentum(parameters=model.parameters(), learning_rate=scheduler, momentum=0.9, weight_decay=weight_decay, grad_clip=clip)
    else:
        raise Exception("Unknow optims.")
    return optimizer







# def sigmoid(x):
#     # TODO: Implement sigmoid function
#     return 1/(1 + np.exp(-x))

def movenetDecode(data, kps_mask=None,mode='output', num_joints = 17, 
                img_size=192, hm_th=0.1):
    ##data [64, 7, 48, 48] [64, 1, 48, 48] [64, 14, 48, 48] [64, 14, 48, 48]
    #kps_mask [n, 7]


    if mode == 'output':
        batch_size = data[0].shape[0]

        heatmaps = data[0].detach().numpy()

        heatmaps[heatmaps < hm_th] = 0

        centers = data[1].detach().numpy()


        regs = data[2].detach().numpy()
        offsets = data[3].detach().numpy()

        
        cx,cy = maxPoint(centers)
        # cx,cy = extract_keypoints(centers[0])
        #print("movenetDecode 119 cx,cy: ",cx,cy)

        dim0 = np.arange(batch_size,dtype=np.int32).reshape(batch_size,1)
        dim1 = np.zeros((batch_size,1),dtype=np.int32)

        res = []
        for n in range(num_joints):
            #nchw!!!!!!!!!!!!!!!!!

            reg_x_origin = (regs[dim0,dim1+n*2,cy,cx]+0.5).astype(np.int32)
            reg_y_origin = (regs[dim0,dim1+n*2+1,cy,cx]+0.5).astype(np.int32)
            # print(reg_x_origin,reg_y_origin)
            reg_x = reg_x_origin+cx
            reg_y = reg_y_origin+cy
            # print(reg_x, reg_y)

            ### for post process
            reg_x = np.reshape(reg_x, (reg_x.shape[0],1,1))
            reg_y = np.reshape(reg_y, (reg_y.shape[0],1,1))
            # print(reg_x.shape,reg_x,reg_y)
            reg_x = reg_x.repeat(48,1).repeat(48,2)
            reg_y = reg_y.repeat(48,1).repeat(48,2)
            #print(reg_x.repeat(48,1).repeat(48,2).shape)
            #bb


            #### 根据center得到关键点回归位置，然后加权heatmap
            range_weight_x = np.reshape(_range_weight_x,(1,48,48)).repeat(reg_x.shape[0],0)
            range_weight_y = np.reshape(_range_weight_y,(1,48,48)).repeat(reg_x.shape[0],0)
            tmp_reg_x = (range_weight_x-reg_x)**2
            tmp_reg_y = (range_weight_y-reg_y)**2
            # print(tmp_reg_x.shape, _range_weight_x.shape, reg_x.shape)
            tmp_reg = (tmp_reg_x+tmp_reg_y)**0.5+1.8#origin 1.8
            #print(tmp_reg.shape,heatmaps[:,n,...].shape)(1, 48, 48)
            # print(heatmaps[:,n,...][0][19:25,19:25])
            # cv2.imwrite("t.jpg",heatmaps[:,n,...][0]*255)
            # print(tmp_reg[0][19:25,19:25])
            tmp_reg = heatmaps[:,n,...]/tmp_reg
            # print(tmp_reg[0][19:25,19:25])

            

            # reg_cx = max(0,min(47,reg_x[0][0][0]))
            # reg_cy = max(0,min(47,reg_y[0][0][0]))
            # _reg_weight_part = _reg_weight[49-reg_cy:49-reg_cy+48, 49-reg_cx:49-reg_cx+48]
            # if _reg_weight_part.shape[0]!=48 or _reg_weight_part.shape[1]!=48:
            #     print(_reg_weight_part.shape)
            #     print(reg_cy,reg_cx)
            #     bbb
            # # print(_reg_weight_part[reg_cy,reg_cx])
            # #keep reg_cx reg_cy to 1
            # tmp_reg = heatmaps[:,n,...]*_reg_weight_part

            # b


            # if n==1:
            #     cv2.imwrite('output/predict/t3.jpg', cv2.resize(tmp_reg[0]*2550,(192,192)))
            tmp_reg = tmp_reg[:,np.newaxis,:,:]
            reg_x,reg_y = maxPoint(tmp_reg, center=False)
            
            # # print(reg_x, reg_y)
            reg_x[reg_x>47] = 47
            reg_x[reg_x<0] = 0
            reg_y[reg_y>47] = 47
            reg_y[reg_y<0] = 0

            score = heatmaps[dim0,dim1+n,reg_y,reg_x]
            # print(score)
            offset_x = offsets[dim0,dim1+n*2,reg_y,reg_x]#*img_size//4
            offset_y = offsets[dim0,dim1+n*2+1,reg_y,reg_x]#*img_size//4
            # print(offset_x,offset_y)
            res_x = (reg_x+offset_x)/(img_size//4)
            res_y = (reg_y+offset_y)/(img_size//4)
            # print(res_x,res_y)

            res_x[score<hm_th] = -1
            res_y[score<hm_th] = -1


            res.extend([res_x, res_y])
            # b
                
        res = np.concatenate(res,axis=1) #bs*14


      
    elif mode == 'label':
        kps_mask = kps_mask.detach().cpu().numpy()

        data = data.detach().cpu().numpy()
        # print(data.shape)
        batch_size = data.shape[0]
        
        heatmaps = data[:,:17,:,:]
        centers = data[:,17:18,:,:]
        regs = data[:,18:52,:,:]
        offsets = data[:,52:,:,:]

        # cv2.imwrite(os.path.join("_centers.jpg"), centers[0][0]*255)
        # cv2.imwrite(os.path.join("_heatmaps0.jpg"), heatmaps[0][0]*255)
        # cv2.imwrite(os.path.join("_regs0.jpg"), regs[0][0]**2*255)
        # cv2.imwrite(os.path.join("_offsets0.jpg"), offsets[0][1]**2*1000)
        # bb
        # print("movenetDecode centers  ",centers)
        # print(centers[0,0,26:30,30:33])
        # print(_center_weight[20:26,20:26])
        # t = centers*_center_weight
        # print(t[0,0,20:26,20:26])
        cx,cy = maxPoint(centers)
        # cx,cy = extract_keypoints(centers[0])
        # print("decode maxPoint " ,cx, cy)
        # print(regs[0][0][22:26,22:26])
        # print(regs[0][1][22:26,22:26])
        
        # print(cx.shape,cy.shape)#(64, 1) (64, 1)
        # print(cx.squeeze().shape)#(64,)
        # print(regs.shape)#64, 14, 48, 48
        dim0 = np.arange(batch_size,dtype=np.int32).reshape(batch_size,1)
        dim1 = np.zeros((batch_size,1),dtype=np.int32)

        res = []
        for n in range(num_joints):
            #nchw!!!!!!!!!!!!!!!!!
            #print(regs[dim0,dim1,cx,cy].shape) #64,1

            reg_x_origin = (regs[dim0,dim1+n*2,cy,cx]+0.5).astype(np.int32)
            reg_y_origin = (regs[dim0,dim1+n*2+1,cy,cx]+0.5).astype(np.int32)

            # print(reg_x_origin,reg_y_origin)
            # print(np.max(regs[dim0,dim1+n*2,:,:]),np.min(regs[dim0,dim1+n*2,:,:]))

            # print(regs[dim0,dim1+n*2,22:26,22:26])
            # b
            # print(kps_mask.shape, kps_mask[:,n].shape,kps_mask[:,n])
            # bb

            # print(reg_x_origin,reg_y_origin)

            reg_x = reg_x_origin+cx
            reg_y = reg_y_origin+cy

            # print(reg_x, reg_y)
            reg_x[reg_x>47] = 47
            reg_x[reg_x<0] = 0
            reg_y[reg_y>47] = 47
            reg_y[reg_y<0] = 0

            offset_x = offsets[dim0,dim1+n*2,reg_y,reg_x]#*img_size//4
            offset_y = offsets[dim0,dim1+n*2+1,reg_y,reg_x]#*img_size//4
            # print(offset_x,offset_y)
            res_x = (reg_x+offset_x)/(img_size//4)
            res_y = (reg_y+offset_y)/(img_size//4)

            #不存在的点设为-1 后续不参与acc计算
            res_x[kps_mask[:,n]==0] = -1
            res_y[kps_mask[:,n]==0] = -1
            #print(res_x,res_y)
            #print()
            res.extend([res_x, res_y])
            # b
                
        res = np.concatenate(res,axis=1) #bs*14
        # print(res.shape)
        #b
    return res



