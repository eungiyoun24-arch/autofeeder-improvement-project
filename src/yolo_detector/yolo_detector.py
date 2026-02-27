import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

LINE_THICKNESS = 5

class YOLODetector:
    def __init__(self, path, conf_threshold=0.6):
        self.model = YOLO(path)
        self.conf_threshold = conf_threshold
    
    
    def inference(self, img):
        results = self.model(img)[0]
        return results
    
    
    def extract_center(self, results):
        classes = results.boxes.cls.numpy()
        boxes = results.boxes.xyxy.numpy().astype(np.int32)
        
        inner_center = []
        dimple_center = []
        
        for cls, bbox in zip(classes, boxes):
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            if cls == 0: 
                dimple_center.append((cx, cy))
            elif cls == 1:
                inner_center.append((cx, cy))
                
        return dimple_center, inner_center, boxes, classes

    
    def compute_angle(self, dimple_centers, inner_centers):
        if len(dimple_centers) == 0 or len(inner_centers) == 0:
            return None, None

        inner_c = np.array(inner_centers[0])
        line = []

        if len(dimple_centers) == 1:
            dimple_c = np.array(dimple_centers[0])
            
            dx = dimple_c[0] - inner_c[0]
            dy = -(midpoint[1] - inner_c[1]) # y-axis is downward in opencv
            angle = np.arctan2(dy, dx)
            
            line.append((tuple(inner_c), tuple(dimple_c)))
        else:
            p1, p2 = np.array(dimple_centers[0]), np.array(dimple_centers[1])
            midpoint = (p1 + p2) / 2
            
            dx = midpoint[0] - inner_c[0]
            dy = -(midpoint[1] - inner_c[1]) # y-axis is downward in opencv
            angle = np.arctan2(dy, dx)
            
            line.append((tuple(inner_c), tuple(midpoint.astype(int))))

        return angle, line
    
    
    def visualize(self, results, img, line=None, export=False, export_path='./result/img/output.png'):
        img = img.copy()
        classes = results.boxes.cls.numpy()
        boxes = results.boxes.xyxy.numpy().astype(np.int32)

        for cls, bbox in zip(classes, boxes):
            if cls in [0,1]:  # ellipse for dimple/inner
                center = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)
                axes = ((bbox[2]-bbox[0])//2, (bbox[3]-bbox[1])//2)
                cv2.ellipse(img, center, axes, 0, 0, 360, (0,255,0), LINE_THICKNESS)
                
                img_h, img_w = img.shape[:2]
                base_width = 640 
                scale = img_w / base_width * 0.5

                label = {0: "Dimple", 1: "Inner Center"}[cls]
                lbl_margin = int(3 * scale)  
                font_scale = 1.0 * scale
                thickness = max(1, int(2 * scale))

                label_size = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)
                lbl_w, lbl_h = label_size[0]
                lbl_w += 2 * lbl_margin
                lbl_h += 2 * lbl_margin

                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+lbl_w, bbox[1]-lbl_h), color=(0, 0, 255), thickness=-1)
                cv2.putText(img, label, (bbox[0]+lbl_margin, bbox[1]-lbl_margin),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(255, 255, 255), thickness=thickness)
                
            else:  # rectangle for others
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), LINE_THICKNESS)
                
                # label = "Lobe" 
                # lbl_margin = 3 
                # label_size = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2) 
                # lbl_w, lbl_h = label_size[0]
                # lbl_w += 2* lbl_margin 
                # lbl_h += 2*lbl_margin 
                # img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+lbl_w, bbox[1]-lbl_h), color=(0, 0, 255), thickness=-1) 
                # cv2.putText(img, label, (bbox[0]+ lbl_margin, bbox[1]-lbl_margin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255 ), thickness=1)
                
                # Label scaling with the size of the img
                img_h, img_w = img.shape[:2]
                base_width = 640 
                scale = img_w / base_width  

                label = "Lobe"
                lbl_margin = int(3 * scale)  
                font_scale = 1.0 * scale
                thickness = max(1, int(2 * scale))

                label_size = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)
                lbl_w, lbl_h = label_size[0]
                lbl_w += 2 * lbl_margin
                lbl_h += 2 * lbl_margin

                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+lbl_w, bbox[1]-lbl_h), color=(0, 0, 255), thickness=-1)
                cv2.putText(img, label, (bbox[0]+lbl_margin, bbox[1]-lbl_margin),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(255, 255, 255), thickness=thickness)

        if line:
            for start, end in line:
                cv2.line(img, start, end, (255,0,0), LINE_THICKNESS)
                
        if export:
            folder = os.path.dirname(export_path)
            base_name, ext = os.path.splitext(os.path.basename(export_path))
            if folder and not os.path.exists(folder):
                print("The result folder is created.")
                os.makedirs(folder, exist_ok=True)

            save_path = export_path
            counter = 1
            while os.path.exists(save_path):
                save_path = os.path.join(folder, f"{base_name}{counter}{ext}")
                counter += 1

            cv2.imwrite(save_path, img)
            print(f"Image saved to {save_path}")

        return img
    
    
    def export_result(self, results, img, csv_path="./result/csv/results.csv"):
        dimple_centers, inner_centers, boxes, classes = self.extract_center(results)
        angle, _ = self.compute_angle(dimple_centers, inner_centers)
        
        w, h = img.shape[:2]
        img_center = np.array([w//2, h//2])
        
        inner_center_val = inner_centers[0] - img_center if inner_centers else None
        dimple0_val = np.array(dimple_centers[0]) - img_center if len(dimple_centers) > 0 else None
        dimple1_val = np.array(dimple_centers[1]) - img_center if len(dimple_centers) > 0 else None
        angle_deg = np.degrees(angle) if angle is not None else None

        folder = os.path.dirname(csv_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        
        row = {
            "inner_center_x": float(inner_center_val[0]) if inner_center_val is not None else '-',
            "inner_center_y": float(inner_center_val[1]) if inner_center_val is not None else '-',
            "dimple0_center_x": float(dimple0_val[0]) if dimple0_val is not None else '-',
            "dimple0_center_y": float(dimple0_val[1]) if dimple0_val is not None else '-',
            "dimple1_center_x": float(dimple1_val[0]) if dimple1_val is not None else '-',
            "dimple1_center_y": float(dimple1_val[1]) if dimple1_val is not None else '-',
            "angle_deg": float(angle_deg) if angle_deg is not None else '-',
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df.to_csv(csv_path, index=False)
        print(f"Result exported to {csv_path}")