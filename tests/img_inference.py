import cv2
from yolo_detector import YOLODetector

MODEL_PATH = './model/best.pt'
IMG_PATH = './test/img4.jpg'


frame = cv2.imread(IMG_PATH)
h, w = frame.shape[:2]

model = YOLODetector(MODEL_PATH)
results = model.inference(frame)

dimple_centers, inner_centers, boxes, classes = model.extract_center(results)
angle, line = model.compute_angle(dimple_centers, inner_centers)

vis_frame = model.visualize(results, frame, line=line, export=True)
model.export_result(results, frame)

screen_width = 1920
screen_height = 1080

img_h, img_w = vis_frame.shape[:2]
scale = min(screen_width / img_w, screen_height / img_h)
new_w = int(img_w * scale)
new_h = int(img_h * scale)
resized = cv2.resize(vis_frame, (new_w, new_h))

top = (screen_height - new_h) // 2
bottom = screen_height - new_h - top
left = (screen_width - new_w) // 2
right = screen_width - new_w - left
letterbox = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('Result', letterbox)
cv2.waitKey(0)
