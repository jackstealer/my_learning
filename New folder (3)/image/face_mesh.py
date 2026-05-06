import cv2
import mediapipe as mp

mp_face_mesh=mp.solutions.face_mesh
mp_drawing=mp.solutions.drawing_utils
face_mesh=mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,)

cap=cv2.VideoCapture(0)
while True:
    flag,frame=cap.read()
    if not flag:
        break
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            h,w,_=frame.shape
            point=face_landmarks.landmark[33]
            x=int(point.x*w)
            y=int(point.y*h) 
            cv2.circle(frame,(x,y),5,(255,0,0),2) 

            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1,color=(255,0,255)),
            #     connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1,color=(255,0,255))
            # )
    cv2.imshow("face mesh",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()