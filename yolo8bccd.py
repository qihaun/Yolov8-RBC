from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(data='BCCD.yaml',
            workers=0,
            epochs=200,
            batch=8,
            imgsz=800,
            lr0=0.0006,
            box=8.5,  #train3加入且早停由150改为50,train4改为8.5早停改为100预训练轮次由20变为10
            cos_lr=True,
            lrf=0.01,
            weight_decay=0.001,#train5由0.0005改为0.001
            warmup_epochs=10,
            augment=True,
            patience=30,
            device=0)
#from ultralytics import YOLO

#model = YOLO('runs/detect/train13/weights/best.pt')

#model.train(data='BCCD.yaml',workers=0,epochs=200,batch=8,imgsz=640,lr0=0.0003,cos_lr=True,lrf=0.01,weight_decay=0.0005,augment=True,patience=100,device=0)
#train3加入box=10且早停由150改为50,train4改为8.5早停改为100预训练轮次由20变为10
#train5 weight_decay由0.0005改为0.001早停改为30
#train6改为n模型（变化不大召回率降低）
#train7 imgsz由640改为800改回s模型