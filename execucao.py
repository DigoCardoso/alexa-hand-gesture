import numpy as np
import cv2
import mediapipe as mp
from picamera2 import Picamera2
from keras.models import load_model
import json
import time
import urllib.request
from reconhecimento import reconhecimento
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)                        #Set GPIO pin numbering
pir = 23                                      #Associate pin 23 to pir
GPIO.setup(pir, GPIO.IN)                      #Set pin as input

def aciona_rotina():
    with urllib.request.urlopen('https://www.virtualsmarthome.xyz/url_routine_trigger/activate.php?trigger=c34c099f-a495-4de5-acd5-335e49e6a484&token=e70b57af-3503-42d3-8c8d-885d626315d5&response=html') as response:
       html = response.read()
    return

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the trained LSTM model
model = load_model('modelo_d.h5')

# Prepare the mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

# Prepare MediaPipe drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

time.sleep(2)

if __name__ == '__main__':
    # Create the proxy options if the data is present in cmdData
    #proxy_options = None
    #if cmdData.input_proxy_host is not None and cmdData.input_proxy_port != 0:
    #    proxy_options = http.HttpProxyOptions(
    #        host_name=cmdData.input_proxy_host,
    #        port=cmdData.input_proxy_port)

    # Create a MQTT connection from the command line data
    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint='a2qi2wn0epen4w-ats.iot.us-east-1.amazonaws.com',
        port=cmdData.input_port,
        cert_filepath=cmdData.input_cert,
        pri_key_filepath=cmdData.input_key,
        ca_filepath=cmdData.input_ca,
        on_connection_interrupted=on_connection_interrupted,
        on_connection_resumed=on_connection_resumed,
        client_id=cmdData.input_clientId,
        clean_session=False,
        keep_alive_secs=30,
        #http_proxy_options=proxy_options,
        on_connection_success=on_connection_success,
        on_connection_failure=on_connection_failure,
        on_connection_closed=on_connection_closed)

    if not cmdData.input_is_ci:
        print(f"Connecting to {cmdData.input_endpoint} with client ID '{cmdData.input_clientId}'...")
    else:
        print("Connecting to endpoint with client ID")
    connect_future = mqtt_connection.connect()

    # Future.result() waits until a result is available
    connect_future.result()
    print("Connected!")
    
    while True:
        # if button was pushed
        if GPIO.input(pir):
            reconhecimento(picam2,model,hands)
            
            try:
                with open('dados.json','r') as file:
                    message = json.lead(file)
                except Exception as e:
                    message = {"Erro":"Erro ao abrir o arquivo JSON"}
                    
            message_json = json.dumps(message)
            mqtt_connection.publish(
                topic="alexa/db",
                payload=message_json,
                qos=mqtt.QoS.AT_LEAST_ONCE)
            
            aciona_rotina()
            time.sleep(1)
        

    # Disconnect
    print("Disconnecting...")
    disconnect_future = mqtt_connection.disconnect()
    disconnect_future.result()
    print("Disconnected!")



