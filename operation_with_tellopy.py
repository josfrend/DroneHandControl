from djitellopy import Tello
import mediapipe as mp
import math
import cv2
import csv

frame_width = int(680)
frame_height = int(680)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

size = (frame_width, frame_height)
cap = cv2.VideoCapture(0)

end = False

integral = 0.
previous_error = 0.
roll_gain = 0.
pitch_gain = 0.
acceleration_gain = 0.
count = 0

def hand_tracking (img):

    global roll_gain, acceleration_gain, pitch_gain, end, count

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (640,480))
    height, width, channels = image.shape

    # cv2.circle(img, (width//2, height//2), 10, (255,0,0), 5)

    
    #image = cv2.flip(image, 1)
    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    roll_gain = 0.
    acceleration_gain = 0.
    pitch_gain = 0.

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]

            distance = math.sqrt(
                (thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2 +
                (thumb_tip.z - index_finger_tip.z) ** 2)
            
            ring_distance = math.sqrt(
                (thumb_tip.x - ring_finger_tip.x) ** 2 + (thumb_tip.y - ring_finger_tip.y) ** 2 +
                (thumb_tip.z - ring_finger_tip.z) ** 2)

            # Determine if the hand is open or closed based on the distance
            is_open = distance > 0.1  # Adjust the threshold value as needed
            ring_thumb = ring_distance > 0.1

            hand_status = "Open" if is_open else "Closed"
            gesture = False if ring_thumb else True
            
            hand_position = (hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x * img.shape[1],
                                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y * img.shape[0])
            
            handedness = results.multi_handedness[
                    results.multi_hand_landmarks.index(hand_landmarks)].classification[0].label

            # Calculate the dimensions of the hand
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y

            hand_dimensions = (x_min, y_min, x_max - x_min, y_max - y_min)
            

            # Draw bounding box and hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks,
                                                        mp.solutions.hands.HAND_CONNECTIONS)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            if(handedness == "Left" and hand_status == "Closed"):
                if(hand_position[0] < width//2-20):
                    cv2.circle(img, (width-50, height//2), 10, (0,0,255), 5)
                    roll_gain = round(pid_controller(hand_position[0]-width//2+20, 0.1, 0, 0), 3)
                    # tl_flight.rc(a=-10, b=0, c=0, d=0)
                    # print("left")
                elif(hand_position[0] > width//2+20):
                    cv2.circle(img, (50, height//2), 10, (0,0,255), 5)
                    roll_gain = round(pid_controller(hand_position[0]-width//2+20, 0.1, 0, 0), 3)
                    # print("right")
                    # tl_flight.rc(a=10, b=0, c=0, d=0)

                if(hand_position[1] < height//2-20):
                    cv2.circle(img, (width//2, height-50), 10, (0,0,255), 5)
                    acceleration_gain = max(min(round(pid_controller(height//2+20 - hand_position[1], 0.1, 0, 0), 3), 20), -20)
                    # tl_flight.rc(a=0, b=0, c=10, d=0)
                    # print("up")
                elif(hand_position[1] > height//2+20):
                    # tl_flight.rc(a=0, b=0, c=-10, d=0)
                    acceleration_gain = round(pid_controller(height//2+20 - hand_position[1], 0.4, 0, 0), 3)
                    cv2.circle(img, (width//2, 50), 10, (0,0,255), 5)
                    # print("down")
                
                if(abs(75-hand_dimensions[3]) > 5 and hand_dimensions[3] < 150):
                    cv2.circle(img, (width//2,height//2), 10, (0, 255, 0), 5)
                    pitch_gain = round(pid_controller(75 - hand_dimensions[3], 0.8, 0, 0), 3)


                print(hand_dimensions[3])

                count = 0
            

            elif(handedness == "Right" and gesture == True):
                print("Other hand")
                count += 1

                if count >= 50: end = True 
            else:
                count = 0
                roll_gain= 0.
                acceleration_gain = 0.
                pitch_gain = 0.
                print("Nothing")

            

            

def pid_controller(error,kp,ki,kd):
    global integral
    global previous_error

    # Calculate proportional term
    proportional = kp * error
    
    # Calculate integral term
    integral = integral + (error * ki)
    
    # Calculate derivative term
    derivative = kd * (error - previous_error)
    
    # Calculate control output
    control_output = proportional + integral + derivative
    
    # Output saturation
    control_output = max(min(control_output, 40), -40)

    # Update previous error
    previous_error = error
    
    return control_output



if __name__ == '__main__':
    tello = Tello()
    tello.connect()
    tello.streamon()


    battery_info = tello.get_battery()

    # Set the QUAV to takeoff
    if(battery_info > 5):
        tello.takeoff()

        # # Add a delay to remain in hover
        # print("Remaning in hover")
        # time.sleep(1)
        # print("Initiating movement")

        csv_file = open("data.csv", "w", newline="")
        writer = csv.writer(csv_file)  
        header = ["Episode", "Roll Gain", "Pitch Gain","Acceleration Gain"]
        writer.writerow(header) 

        episode = 1
        while True:
            #ret, img = cap.read()
            img = tello.get_frame_read().frame

            hand_tracking(img)


            if cv2.waitKey(1) == ord("q") or end:
                break
            
            roll_gain = int(roll_gain)
            acceleration_gain = int(acceleration_gain)
            pitch_gain = int(pitch_gain)

            text = "Roll gain: "+str(roll_gain)+"\tAcceleration gain: "+str(acceleration_gain) + "\tPitch gain: "+str(pitch_gain)
            cv2.putText(img, text, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 10.0, (255,255,255), 2)

            

            # Write the data rows
            writer.writerows([str(episode),str(roll_gain), str(pitch_gain),str(acceleration_gain)])

            tello.send_rc_control(roll_gain, pitch_gain, acceleration_gain, 0)
            episode +=1

            cv2.imshow("Drone", img)
            cv2.waitKey(1)


        cv2.destroyAllWindows()
        csv_file.close()


        tello.send_rc_control(0, 0, 0, 0)

        # Set the QUAV to land
        tello.land()
    else: 
        print("Cannot operate. Drone battery soc: {0}".format(battery_info))

    # Close resources
    tello.streamoff()
    tello.end() 
