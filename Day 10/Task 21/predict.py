from ultralytics import Yolo

model = Yolo('best.pt')

results = model('test.png')

print(results)