from flask import Flask, jsonify
from flask_cors import CORS
import cv2
import os

app = Flask(__name__)
CORS(app)

@app.route('/get_shape', methods=['GET'])
def get_shape():
    image_files = ['input1.jpg', 'input2.jpg', 'input3.jpg']
    all_shapes = []

    for img_path in image_files:
        if not os.path.exists(img_path):
            print(f"경고: {img_path} 파일을 찾을 수 없습니다.")
            continue

        # 1. 이미지를 흑백으로 불러와서 블러 처리 (잔잔한 노이즈 제거)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 2. 부드러운 외곽선 추출
        edges = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        points = []
        h, w = img.shape
        
        # 3. 곡선을 그대로 살리되, 점을 촘촘하게(3픽셀 간격) 추출하여 물길(Path) 형성
        for cnt in contours:
            if cv2.contourArea(cnt) > 200: # 너무 작은 먼지 같은 윤곽선 제외
                for i, p in enumerate(cnt):
                    if i % 3 == 0: # 3번째 점마다 하나씩 추출 (곡선 유지)
                        nx = p[0][0] / w
                        ny = p[0][1] / h
                        points.append({"x": round(nx, 3), "y": round(ny, 3)})
        
        if points:
            all_shapes.append(points)

    return jsonify(all_shapes)

if __name__ == '__main__':
    app.run(port=5000, debug=True)