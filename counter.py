from ultralytics import YOLO
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

class RideShareCounter():
    def __init__(self, model):
        self.model = model
        self.res = 0
        self.img_arr = np.ndarray()

    def update_result(self, arr: np.ndarray):
        res = self.model(arr)
        self.res = res
        self.img_arr = arr
    
    def get_num_class(
            self, 
            target_class: int = 0,
            class_name: str = 'Person',
            show: bool = False,
            google_colab: bool = False,
            color: tuple = (0, 255, 0)
    ) -> int:
        boxes = self.res[0].boxes
        
        counter = 0
        for box in boxes:
            if box.cls == target_class:
                counter += 1
                
                # バウンディングボックスを描画
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(self.img_arr, (x1, y1), (x2, y2), color, 1)
                cv2.putText(self.img_arr, class_name, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        if show:
            if google_colab:
                try:
                    cv2_imshow(self.img_arr)  # cv2.imshowの代わりにcv2_imshowを使用
                except Exception:
                    print('Google Colabを使用していない場合は`google_colab`をTrueに設定してください')
            else:
                try:
                    cv2_imshow(self.img_arr)
                except Exception:
                    print('Google Colabを使用している場合は`google_colab`をFalseに設定してください')

        return counter
    
    def save_img(self, img_path: str)
        cv2.imwrite(img_path, self.img_arr)
        print(f"画像が {img_path} に保存されました")