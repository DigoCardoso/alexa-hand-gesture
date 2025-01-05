import numpy as np
import cv2
import mediapipe as mp
from picamera2 import Picamera2
from keras.models import load_model
import json
import time
import urllib.request

#funcao para escrita da letra traduzida no arquivo dados.json
def reconhecimento(picam2,model,hands):

#picam2 = Picamera2()
#picam2.preview_configuration.main.size = (640,480)
#picam2.preview_configuration.main.format = "RGB888"
#picam2.preview_configuration.align()
#picam2.configure("preview")
#picam2.start()



# Load the trained LSTM model
#model = load_model('lstm_model.h5')

# Prepare the mediapipe hands module
#mp_hands = mp.solutions.hands
#hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

# Prepare MediaPipe drawing utils
#mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles

# Load labels
    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}  # Update according to your training labels
    num_classes = len(labels_dict)

    trad = 'N'
    ult_trad = 'N'
    atual = 'N'
    cont_trad = 0
    consist = False
    envio = False


# Start capturing video
    cap = cv2.VideoCapture(0)
    count=0
    start = time.time()
    end = 1.0
    while (envio != True) and (end-start < 10.0):
        frame= picam2.capture_array()
        #count += 1
        #if count % 4 != 0:
        #   continue
        frame=cv2.flip(frame,-1)
    
    # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
            # Extract landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

            # Normalize the data (if required, depending on how you trained your model)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalization example
                    data_aux.append(y - min(y_))
                print(len(data_aux))
            # Reshape the input for LSTM: (1, timesteps, features)
                if len(data_aux) == 42:  # Ensure it matches your feature count
                    input_data = np.array(data_aux).reshape((1, 6, 7))  # Adjust according to your setup
                # Make prediction
                    prediction = model.predict(input_data)
                    predicted_index = np.argmax(prediction)
                    predicted_character = labels_dict[predicted_index]
                    trad = predicted_character

                # Get the confidence values
                    confidence_values = prediction[0]
                    confidence_percentage = confidence_values[predicted_index] * 100

                # Draw the predicted character and confidence percentage
                    #cv2.putText(frame, f'{predicted_character}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
                    #cv2.putText(frame, f'Confidence: {confidence_percentage:.2f}%', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)


        if trad != "N":
            if trad == ult_trad:
                if cont_trad < 0:
                    cont_trad = cont_trad + 1
                else:
                    consist = True
                    atual = trad
            else:
                cont_trad = 0


        if consist:     
            dicionario = {
                "Letra":"{}".format(atual)
            }
            with open("dados.json","w") as file:
                json.dump(dicionario,file)
            envio = True
            print("ESCRITA REALIZADA!!")
            print("Letra {} detectada.".format(atual))


        ult_trad = trad
        end = time.time()

    # Display the frame
        #cv2.imshow('Hand Gesture Recognition', frame)

    #if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
    #    break

# Release the capture
    print("Tempo de reconhecimento: {} seg".format(end-start))
    cap.release()
    cv2.destroyAllWindows()


