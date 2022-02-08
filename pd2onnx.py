import os
import random
import pandas as pd   
import paddle
from paddle.static import InputSpec

from lib import init, Data, MoveNet, Task

from config import cfg





def main(cfg):

    init(cfg)


    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    


    run_task = Task(cfg, model)
    run_task.modelLoad('output/e74_valacc0.76914.pth')


    run_task.model.eval()

    #data type nchw
    input_spec = InputSpec([1, 3, 192, 192], 'float32', 'x')
    save_path = 'output/pose.onnx'
    # output_names = [ "output1","output2","output3","output4" ]

    paddle.onnx.export(run_task.model, save_path, input_spec = [input_spec], opset_version=11)



if __name__ == '__main__':
    main(cfg)









