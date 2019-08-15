import hand_detection
import cv2
import time
import copy

# getting video feed from webcam
cap = cv2.VideoCapture(0)

hist = hand_detection.capture_histogram(source=0)
(prevx, prevy) = (0,0)
prev_dir = None
closed = False

bufsize = 20
buf = [0]*bufsize
bufi = 0
clr_time = time.clock()

font = cv2.FONT_HERSHEY_SIMPLEX
tx = 10 #position of text
ty = 20 #position of text

while True:
    ret, original = cap.read()
    if not ret:
        break

    frame = original.copy()
    hand_detection.detect_face(frame, block=True)
    hand = hand_detection.detect_hand(original, frame, hist)

    # to get the outline of the hand
    # min area of the hand to be detected = 10000 by default
    custom_outline = hand.draw_outline(min_area=10000, color=(0, 255, 255), thickness=2)

    # to get a quick outline of the hand
    quick_outline = hand.outline

    # draw fingertips on the outline of the hand, with radius 5 and color red,
    for fingertip in hand.fingertips:
        cv2.circle(quick_outline, fingertip, 5, (0, 0, 255), -1)

    # to get the centre of mass of the hand
    com = hand.get_center_of_mass()

    if com:
        cv2.circle(quick_outline, com, 10, (255, 0, 0), -1)
        (curx, cury) = com
        difx = curx - prevx
        dify = cury - prevy
        cur_dir = None
        if abs(difx) > 3 and abs(difx) > abs(dify): 
            if difx > 0:
                cur_dir = 'left'
                #print(cur_dir)
            else:
                cur_dir = 'right'
                #print(cur_dir)
        elif abs(dify) > 2 and abs(dify) > abs(difx): 
            if dify > 0:
                cur_dir = 'down'
                #print(cur_dir)
            else:
                cur_dir = 'up'
                #print(cur_dir)
        disp_text = cur_dir

        count = hand.count
        if count==0:
            closed = True
            if disp_text:
                disp_text += ' closed'
            else:
                disp_text = 'closed'
        
        buf[bufi%bufsize] = int(not closed and (cur_dir != None) and (prev_dir != cur_dir))
        cur_time = time.clock()
        if cur_time - clr_time > 4:
            buf = [0]*bufsize
            clr_time = time.clock()
        if sum(buf) > 4:
            if (cur_dir != None) and (not closed):
                disp_text += " waving"

        print(disp_text)
        if disp_text != None:
            cv2.putText(quick_outline, disp_text, (tx,ty), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        bufi += 1
        prev_dir = cur_dir
        (prevx, prevy) = (curx, cury)

    cv2.imshow("hand_detection", quick_outline)

    k = cv2.waitKey(5)

    # Press 'q' to exit
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
