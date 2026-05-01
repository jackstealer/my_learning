import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh=mp.solutions.face_mesh
mp_drawing=mp.solutions.drawing_utils
face_mesh=mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=7,
    refine_landmarks=True,)

script_dir = os.path.dirname(os.path.abspath(__file__))
sunglasses = cv2.imread(os.path.join(script_dir, "sunglass1.png"), cv2.IMREAD_UNCHANGED)
hat = cv2.imread(os.path.join(script_dir, "hat.png"), cv2.IMREAD_UNCHANGED)
if sunglasses is None:
    raise FileNotFoundError("sunglass1.png not found in script directory.")
if hat is None:
    raise FileNotFoundError("hat.png not found in script directory.")

def overlay_png(frame, overlay, x_offset, y_offset):
    """Blend a PNG with alpha channel onto frame at given offset, clamped to frame."""
    fh, fw = frame.shape[:2]
    oh, ow = overlay.shape[:2]
    x1c = max(0, x_offset);  y1c = max(0, y_offset)
    x2c = min(fw, x_offset+ow); y2c = min(fh, y_offset+oh)
    sx1 = x1c-x_offset; sy1 = y1c-y_offset
    sx2 = sx1+(x2c-x1c); sy2 = sy1+(y2c-y1c)
    if x2c <= x1c or y2c <= y1c:
        return
    roi = frame[y1c:y2c, x1c:x2c]
    ov_img = overlay[sy1:sy2, sx1:sx2, :3]
    msk = overlay[sy1:sy2, sx1:sx2, 3]
    msk_inv = cv2.bitwise_not(msk)
    bg = cv2.bitwise_and(roi, roi, mask=msk_inv)
    fg = cv2.bitwise_and(ov_img, ov_img, mask=msk)
    frame[y1c:y2c, x1c:x2c] = cv2.add(bg, fg)
cap=cv2.VideoCapture(0)
while True:
    flag,frame=cap.read()
    if not flag:
        break
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            # ── SUNGLASSES ─────────────────────────────────────────────
            left_eye  = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            x1, y1 = int(left_eye.x*w),  int(left_eye.y*h)
            x2, y2 = int(right_eye.x*w), int(right_eye.y*h)
            glasses_width  = int(np.hypot(x2-x1, y2-y1) * 1.8)
            glasses_height = int(glasses_width * 0.5)
            if glasses_width > 0 and glasses_height > 0:
                resized_sg = cv2.resize(sunglasses, (glasses_width, glasses_height))
                x_center = (x1+x2)//2
                y_center = (y1+y2)//2
                overlay_png(frame, resized_sg,
                            int(x_center - glasses_width/2),
                            int(y_center - glasses_height/2))

            # ── HAT ────────────────────────────────────────────────────
            # Use face-width landmarks (left cheek 234, right cheek 454)
            # and forehead top (landmark 10) to position the hat
            lm_left     = face_landmarks.landmark[234]
            lm_right    = face_landmarks.landmark[454]
            lm_forehead = face_landmarks.landmark[10]
            fl_x = int(lm_left.x  * w);  fl_y = int(lm_left.y  * h)
            fr_x = int(lm_right.x * w);  fr_y = int(lm_right.y * h)
            fh_x = int(lm_forehead.x * w); fh_y = int(lm_forehead.y * h)
            face_width  = int(np.hypot(fr_x-fl_x, fr_y-fl_y))
            hat_width   = int(face_width * 1.5)   # hat wider than face
            hat_oh, hat_ow = hat.shape[:2]
            hat_height  = int(hat_width * hat_oh / hat_ow)  # keep aspect ratio
            if hat_width > 0 and hat_height > 0:
                resized_hat = cv2.resize(hat, (hat_width, hat_height))
                hx_center   = (fl_x + fr_x) // 2
                # bottom of hat aligns with forehead landmark, extends upward
                hat_x = int(hx_center - hat_width / 2)
                hat_y = int(fh_y - hat_height + hat_height * 0.3)   # push hat lower onto head
                overlay_png(frame, resized_hat, hat_x, hat_y)
            
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