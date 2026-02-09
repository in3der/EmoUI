import dlib
import cv2
import numpy as np
from facial_analysis import FacialImageProcessing
from tensorflow.keras.models import load_model

## 모델 경로
landmark_model = "../models/shape_predictor_68_face_landmarks.dat"
base_model_path = "../models/mobilenet_7.h5"

## 얼굴 표정과 id 딕셔너리 정의
idx_to_class = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}

## 모델 로드
landmark_detector = dlib.shape_predictor(landmark_model)
base_model = load_model(base_model_path)

# 이미지 전처리
imgProcessing = FacialImageProcessing(False)

# 이미지 사이즈
INPUT_SIZE = (224, 224)

# landmark detection using dlib
def lanmark(image, face):
    # 얼굴에서 68개 점 찾기
    landmarks = landmark_detector(image, face)

    # create list to contain landmarks
    landmark_list = []

    # append (x, y) in landmark_list
    for p in landmarks.parts():
        landmark_list.append([p.x, p.y])
        cv2.circle(image, (p.x, p.y), 2, (255, 255, 255), -1)



### extract frames
def get_iou(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou



## main ##
#video_path = "./DataSets/PlayVideos/2026.02.05 은정초등학교 B-003.mp4"
video_path = "./DataSets/PlayVideos/Sample1-009.mp4"


# webcam or video file open
try:
    cap = cv2.VideoCapture(video_path)
    #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture(0)

except Exception as e:
        print(str(e))

if cap.isOpened:
    print('camera is opened width: {0}, height: {1}'.format(cap.get(3), cap.get(4)))

#--- output video setting ---#
output_path = "fer_output_video3.mp4"

# 비디오 정보 (첫 프레임 기준으로 가져오는 게 안전)
ret, image = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")

h, w = image.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback

# 코덱 설정 (mp4용)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# 첫 프레임 저장
writer.write(image)
#---------------------------#

while(cap.isOpened()):
    ret, image = cap.read()
    # 카메라가 작동한다면..
    if ret:
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 영상 이미지를 rgb 형식으로 변환

        bounding_boxes, points = imgProcessing.detect_faces(frame)  # 프레임에서 찾은 얼굴 전처리해서 바운딩 박스와 포인트 값으로 반환
        # 행렬을 전치
        points = points.T

        # loop as the number of face, one loop per one face
        for bbox, p in zip(bounding_boxes, points):
            box = bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            face_img = frame[y1:y2, x1:x2, :]

            # 얼굴이 잡히는데 정해진 사이즈보다 클 경우 예외처리
            try:
                face_img = cv2.resize(face_img, INPUT_SIZE)

                inp = face_img.astype(np.float32)
                inp[..., 0] -= 103.939
                inp[..., 1] -= 116.779
                inp[..., 2] -= 123.68
                inp = np.expand_dims(inp, axis=0)
                scores = base_model.predict(inp)[0]

                # emotion 잡기
                emotion = idx_to_class[np.argmax(scores)]

                # bbox 를 그리기 위해 dlib 타입으로 변환
                face = dlib.rectangle(x1, y1, x2, y2)

                # landmark detection
                lanmark(image, face)

                # draw bounding box and show emotion text on original image
                cv2.rectangle(image, (face.left() - 5, face.top() - 5), (face.right() + 5, face.bottom() + 5),
                              (0, 186, 255), 3)

                cv2.putText(image, emotion, (int(face.left()) + 10, int(face.top()) - 10), cv2.FONT_HERSHEY_COMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(str(e))

        cv2.putText(image, "Intelligent Vision Processing Lab.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow("original", image)

        writer.write(image)  #--- output video 저장 ---#

    else:
        print("error")

    # 알파벳 q 로 영상 종료
    if cv2.waitKey(25) & 0xFF == ord('q'):
        record = False
        break

cap.release()
cv2.destroyAllWindows()
writer.release()  #--- output video 저장 ---#
print(" ****** video saved at : ", output_path)