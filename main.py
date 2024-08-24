from ultralytics import YOLO
import random
from get_frame import get_frame
from tqdm import tqdm
import time

# ここにリンクをリスト形式で貼る
urls = ['https://www.youtube.com/watch?v=CO_ZjH6N7RE']

for i in tqdm(range(1000)):
    url = random.choice(urls)
    img_path = f'./img/fig{i}.png'
    arr = get_frame(url, img_path=img_path)
    time.sleep(60)