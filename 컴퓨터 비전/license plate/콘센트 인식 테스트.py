import torch
import cv2
import numpy as np

# 미리 학습된 YOLOv5s 모델 로드 (COCO 데이터셋 기준)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 테스트 이미지 경로
image_path = r'C:\Users\USER\Desktop\내 실습폴더\PythonStudy\my\콘센트 인식 테스트.png'
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

# 모델 추론 실행
results = model(img)

# 결과 확인: Pandas DataFrame 형식으로 출력 (x1, y1, x2, y2, confidence, class, name)
detections = results.pandas().xyxy[0]
print("검출 결과:")
print(detections)

# 검출된 객체가 있다면 결과 이미지에 박스를 그려서 출력
if not detections.empty:
    print("객체 검출됨!")
    annotated_img = np.squeeze(results.render())
    cv2.imshow("Detected Objects", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("검출된 객체가 없습니다.")
