from ultralytics import YOLO
import yt_dlp
import cv2
import time

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
ydl_options = {'simulate' : True,'ignoreerrors':True, 'no_warnings': True}

def process_frame(frame, model):
    # モデルでフレームを処理する
    results = model(frame)
    return results

# Instanceを作成(前処理と後処理を明示してる？)
with yt_dlp.YoutubeDL(ydl_options) as ydl:
    info = ydl.extract_info('https://www.youtube.com/watch?v=CO_ZjH6N7RE',download = False)
    # print(json.dumps(ydl.sanitize_info(info)))
    # print(info)
    video_url = info['url']
    print(video_url)

# オブジェクトの作成
cap = cv2.VideoCapture(video_url)
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1/fps)
while True:
    # オブジェクトの作成
    cap = cv2.VideoCapture(video_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(100/fps)
    # 動画を1フレームずつ読み込む
    ret, frame = cap.read()
    if not ret:
        break

    # フレームをモデルで処理する
    results = process_frame(frame, model)
    
    # 処理結果の表示（例: バウンディングボックスの描画など）  
    # モデルの結果を描画するメソッドを使用する
    # results.render()

    # 一コマを表示する→分析用にここで処理を行う
    cv2.imshow('frame', frame)
    time.sleep(wait_time)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
