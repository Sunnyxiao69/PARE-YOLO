import warnings
from ultralytics import RTDETR
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = RTDETR("ultralytics/cfg/models/rt-detr/1.yaml")
    # model.load('') # loading pretrain weights
    model.train(data="dataset/data.yaml",
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                workers=16,
                device='0',
                #resume="/home/zhy0311/zhy/xp/ultralytics-main/runs/train/exp-v12-LLVIP/weights/last.pt", # last.pt path
                project='runs/train',
                name='exp-',
                )