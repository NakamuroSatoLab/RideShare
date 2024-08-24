import cv2
import numpy as np
from yt_dlp import YoutubeDL

ydl_opts = {
    'format': 'bv',
    'live_from_start': False,
    'quiet': True,
}

def _get_stream_url(url: str) -> str:
    with YoutubeDL(ydl_opts) as ydl:
        # 例外処理
        try:
            info_dict = ydl.extract_info(url, download=False)
            if not info_dict.get('is_live'):
                raise ValueError("指定されたURLはライブ配信ではありません。")
            return str(info_dict['url'])
            
        except Exception as e:
            print(f"ストリームURLの取得中にエラーが発生しました: {e}")
            return None

def get_frame(url: str, max_attempts: int = 10, img_path: str = None) -> np.ndarray:
    # ストリームURLの取得
    stream_url = _get_stream_url(url)
    if stream_url is None:
        return None

    # キャプチャの取得
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("ビデオストリームを開くことができませんでした。")
        return

    # フレームを取得
    for attempt in range(max_attempts):
        ret, frame = cap.read()
        if ret:
            break
    else:  # max_attemptsまでにフレームを取得できなかった場合の処理
        print("フレームを取得できませんでした。")
        cap.release()
        cv2.destroyAllWindows()
        return

    if img_path:  # フレームを画像として保存する場合
        cv2.imwrite(img_path, frame)
        print(f"フレームが {img_path} に保存されました。")

    cap.release()
    cv2.destroyAllWindows()
    return frame
