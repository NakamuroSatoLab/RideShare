from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt

class RideShareCounter():
    def __init__(self, model):
        self.model = model
        self.res = 0
        self.img_arr = np.empty(0)

    # 解析
    def update_result(self, arr: np.ndarray):
        # 解析結果（result）の取得
        res = self.model(arr)
        self.res = res
        # クラスに画像の一時保存
        self.img_arr = arr
    
    # 指定したオブジェクトの数を返す
    def get_num_class(
            self, 
            target_class: int = 0,
            class_name: str = 'Person',
            show: bool = False,
            google_colab: bool = False,
            color: tuple = (0, 255, 0)
    ) -> int:
        # 任意のオブジェクトを取得
        boxes = self.res[0].boxes
        
        counter = 0
        for box in boxes:
            # オブジェクトが検出したい対象であれば
            if box.cls == target_class:
                counter += 1
                
                # バウンディングボックスを描画（boxの座標を取得）
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 画像に描画
                cv2.rectangle(self.img_arr, (x1, y1), (x2, y2), color, 1)
                # ラベル（テキスト）を描画
                cv2.putText(self.img_arr, class_name, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        # 検出数を返す
        return counter
    
    def save_img(self, img_path: str):
        cv2.imwrite(img_path, self.img_arr)
        print(f"画像が {img_path} に保存されました。")

    def show_img(self, waitkey: int = 0):
        cv2.imshow('Detected Objects', self.img_arr)
        cv2.waitKey(waitkey)
