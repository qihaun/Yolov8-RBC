from ultralytics import YOLO

model = YOLO("runs/detect/train7/weights/best.pt")
results = model.val(
    data="BCCD.yaml",
    workers=0,
    split="test",
    batch=8,
    imgsz=640,
    augment=True,
    save_json=True,
    save_txt=True,
    show=True,
    save=True,
    device=0,
)
print("\n" + "=" * 50)
print("【核心评估指标汇总】")
print(f"精确率 (Precision): {results.box.mp:.4f}")
print(f"召回率 (Recall): {results.box.mr:.4f}")
print(f"mAP@0.5: {results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {results.box.map:.4f}")
print("=" * 50 + "\n")

# 4. 可选：打印每个类别的详细指标
print("【每个类别的详细指标】")
class_names = results.names
for i, class_name in enumerate(class_names):
    p = results.box.p[i]  # 该类别的精确率
    r = results.box.r[i]  # 该类别的召回率
    map50 = results.box.ap50[i]  # 该类别的mAP@0.5
    print(f"{class_name}: Precision={p:.4f}, Recall={r:.4f}, mAP@0.5={map50:.4f}")
