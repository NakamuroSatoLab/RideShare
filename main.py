import os
import time
import datetime
import sqlite3
from ultralytics import YOLO

from get_frame import get_frame
from counter import RideShareCounter

# ここにリンクをリスト形式で貼る
urls = [
    ['https://www.youtube.com/watch?v=CO_ZjH6N7RE', '京都駅八条口']
]

# 実行時間の設定
sleep_time = 600
start_time_hour = 6
end_time_hour = 23

# 画像を保存する閾値
threshold = 10

# 画像の保存先
save_dir = './img'
if not os.path.exists(save_dir):
            os.makedirs(save_dir)

# データベースの設定
dbname = 'database.db'
conn = sqlite3.connect(dbname)
cursor = conn.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS live_log (date DATE, camera TEXT, num_people INTEGER, num_cars INTEGER)")
conn.commit()

# モデルの設定
model = YOLO('yolov8n.pt')
rsc = RideShareCounter(model)

print('処理を実行します。中断するにはCtrl+Cを押してください。')
try:
    while True:
        # 現在時刻の更新
        now = datetime.datetime.now()
        start_time = now.replace(hour=start_time_hour, minute=0, second=0, microsecond=0)
        end_time = now.replace(hour=end_time_hour, minute=59, second=59, microsecond=0)

        if start_time <= now <= end_time:
            for streaming in urls:
                url = streaming[0]
                camera = streaming[1]

                dt_now = now.strftime('%Y-%m-%d %H:%M:%S')
                img_path = f'{save_dir}/{camera}_{dt_now}.png'

                # 画像の取得
                arr = get_frame(url)
                # 推論の実行
                rsc.update_result(arr)

                # オブジェクトのカウント
                num_people = rsc.get_num_class(target_class=0, class_name='person', show=False, color=(255, 0, 0))
                num_cars = rsc.get_num_class(target_class=2, class_name='car', show=True, color=(0, 0, 255))
                
                # 結果の格納
                cursor.execute("INSERT INTO live_log(date, camera, num_people, num_cars) VALUES (?, ?, ?, ?)", (dt_now, camera, num_people, num_cars))
                conn.commit()

                print(f"時刻 : {dt_now}   場所 : {camera}")
                print(f"待機人数 : {num_people}   待機車両数 : {num_cars}")
                
                if num_people >= threshold:
                    rsc.save_img(img_path)

        time.sleep(sleep_time)

except KeyboardInterrupt:
    print('プログラムが中断されました。')

finally:
    cursor.close()
    conn.close()