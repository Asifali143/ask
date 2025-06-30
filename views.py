from django.shortcuts import render
from ultralytics import YOLO
import cv2
import numpy as np
import time
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import cv2
from PIL import Image
import geocoder
import vonage
from django.core.mail import EmailMessage
import requests
from pathlib import Path
import os
from django.conf import settings
BASE_DIR = Path(__file__).resolve().parent.parent


#client = vonage.Client(key="3c68855e", secret="EdOtiRz6aIxia8XU")
#sms = vonage.Sms(client)

#weapon_model = YOLO(r'F:\2025 projects\Agadi College\Weapon detection\WebApp\WeaponProject\WeaponApp\runs\classify\train2\weights\best.pt')

#face_model = YOLO(r'F:\2025 projects\Agadi College\Weapon detection\WebApp\WeaponProject\WeaponApp\runs\classify\train3\weights\best.pt')
# Create your views here.

#model = YOLO(r'F:\2025 projects\Agadi College\Weapon detection\WebApp\WeaponProject\WeaponApp\runs\classify\train4\weights\best.pt')

weapon_model = YOLO(os.path.join(settings.BASE_DIR, 'WeaponApp/runs/classify/train2/weights/best.pt'))
face_model = YOLO(os.path.join(settings.BASE_DIR, 'WeaponApp/runs/classify/train3/weights/best.pt'))
model = YOLO(os.path.join(settings.BASE_DIR, 'WeaponApp/runs/classify/train4/weights/best.pt'))


def index(request):
    return render(request,'index.html')


def get_user_location():
    try:
        response = requests.get("http://ip-api.com/json")
        if response.status_code == 200:
            data = response.json()
            return [data['lat'], data['lon']]
        else:
            print("Location fetch failed:", response.status_code)
    except Exception as e:
        print("Location fetch exception:", e)
    return [0.0, 0.0]

def predict_result(request):
    w_flag = 0
    f_flag = 0
    predicted_class_weapon = None
    predicted_class_face = None
    location_text = "Unknown"

    if request.method == 'POST':  
        print("Entered POST method")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return render(request, 'deaf.html', {'error': "Could not open video."})

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return render(request, 'deaf.html', {'error': "Could not read frame."})

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        weapon_result = weapon_model(rgb_frame)
        face_result = face_model(rgb_frame)

        # Safe location extraction
        '''g = geocoder.ip('me')
        if g.latlng:
            location = g.latlng
            location_text = f"{location[0]}, {location[1]}"
        print(location_text)'''
        location = get_user_location()
        if location and len(location) == 2:
            location_text = f"{location[0]}, {location[1]}"
        else:
            location_text = "Unknown Location"

        # Process weapon prediction
        if weapon_result and len(weapon_result) > 0:
            try:
                names_dict_weapon = weapon_result[0].names
                probs_weapon = weapon_result[0].probs.data.tolist()
                predicted_class_weapon = names_dict_weapon[np.argmax(probs_weapon)]
                print(predicted_class_weapon)
                w_flag = 1
            except Exception as e:
                return render(request, 'predict_result.html', {'error': f"Weapon detection error: {e}"})

        # Process face prediction
        if face_result and len(face_result) > 0:
            try:
                names_dict_face = face_result[0].names
                probs_face = face_result[0].probs.data.tolist()
                predicted_class_face = names_dict_face[np.argmax(probs_face)]
                print(predicted_class_face)
                f_flag = 1
            except Exception as e:
                return render(request, 'predict_result.html', {'error': f"Face detection error: {e}"})

        # Encode image to send via email
        _, buffer = cv2.imencode('.jpg', frame)
        image_data = buffer.tobytes()

        subject = "Alert: Detection Notification"
        to_email = "snehaumatar@gmail.com"
        message_body = ""

        if w_flag == 1 and f_flag == 1:
            message_body = f"Weapon and Face Detected\n\nPerson: {predicted_class_face}\nLocation: {location_text}"
        elif w_flag == 1:
            message_body = f"Weapon Detected\n\nLocation: {location_text}"
        elif f_flag == 1:
            message_body = f"Face Detected\n\nPerson: {predicted_class_face}\nLocation: {location_text}"

        if w_flag == 1 or f_flag == 1:
            try:
                email = EmailMessage(
                    subject,
                    message_body,
                    'weapondetectioncnn@gmail.com',  # Sender email
                    [to_email],              # Receiver email
                )
                email.attach('detection.jpg', image_data, 'image/jpeg')
                email.send(fail_silently=False)
                print("Email sent successfully.")
            except Exception as e:
                print(f"Failed to send email: {e}")

        if w_flag==1:
             print("Entered Weapon Flag")
             client = vonage.Client(key="de169d06", secret="h95rZfxOXSFo2Pxi")
             sms = vonage.Sms(client)
             responseData = sms.send_message({
                        "from": "VonageAPI",
                        "to": "916361569670",  # Add country code before number
                        "text": f"Weapon detected at location: {location_text}",
                    })

             if responseData["messages"][0]["status"] == "0":
                print("Message sent successfully.")
             else:
                print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
        elif f_flag==1:
            print("Entered Face Flag")
            client = vonage.Client(key="de169d06", secret="h95rZfxOXSFo2Pxi")
            sms = vonage.Sms(client)
            responseData = sms.send_message({
                        "from": "VonageAPI",
                        "to": "916361569670",  # Add country code before number
                        "text": f"Persong recognized:{predicted_class_face} and the location is: {location_text}",
                    })

            if responseData["messages"][0]["status"] == "0":
                print("Message sent successfully.")
            else:
                print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
                # Render prediction result in HTML
        
        elif w_flag==1 and f_flag==1:
            client = vonage.Client(key="de169d06", secret="h95rZfxOXSFo2Pxi")
            sms = vonage.Sms(client)
            responseData = sms.send_message({
                        "from": "VonageAPI",
                        "to": "916361569670",  # Add country code before number
                        "text": f"Persong recognized:{predicted_class_face} with weapon and the location is: {location_text}",
                    })

            if responseData["messages"][0]["status"] == "0":
                print("Message sent successfully.")
            else:
                print(f"Message failed with error: {responseData['messages'][0]['error-text']}")

        return render(request, 'predict_result.html', {
            'prediction_weapon': predicted_class_weapon,
            'location': location_text,
            'predicted_face': predicted_class_face
        })

    else:
        return render(request, 'predict_result.html', {'error': "No object detected."})

def predict_result_upload(request):
    prediction = ""
    if request.method == 'POST' and request.FILES.get('image'):
        print("Entered POST method")

        uploaded_file = request.FILES['image']
        file_path = default_storage.save('temp.jpg', ContentFile(uploaded_file.read()))
        image_path = default_storage.path(file_path)

        img = cv2.imread(image_path)
        if img is None:
            return render(request, 'predict_result.html', {'error': "Uploaded file is not a valid image."})

        # Convert BGR to RGB for YOLO
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(rgb_img)  # Assuming your YOLO model returns a list-like object

        prediction = None
        if results and len(results) > 0:
            try:
                names_dict = results[0].names
                probs = results[0].probs.data.tolist()

                predicted_class = names_dict[np.argmax(probs)]
                print("Predicted:", predicted_class)

                # Get user location
                '''g = geocoder.ip('me')
                location = g.latlng
                location_text = f"{location[0]}, {location[1]}"
                print("Location:", location_text)'''
                location = get_user_location()
                if location and len(location) == 2:
                    location_text = f"{location[0]}, {location[1]}"
                else:
                    location_text = "Unknown Location"

                # Send SMS if high confidence
                if max(probs) > 0.5:
                    '''client = vonage.Client(key="de169d06", secret="h95rZfxOXSFo2Pxi")
                    sms = vonage.Sms(client)
                    responseData = sms.send_message({
                        "from": "VonageAPI",
                        "to": "916361569670",
                        "text": f"Weapon detected at location: {location_text}",
                    })
                    if responseData["messages"][0]["status"] == "0":
                        print("Message sent successfully.")
                    else:
                        print(f"SMS Error: {responseData['messages'][0]['error-text']}")'''

                    # Send Email with attachment
                    subject = "Alert: Detection Notification"
                    to_email = "snehaumatar@gmail.com"
                    message_body = f"Weapon detected at location: {location_text}\nPrediction: {predicted_class}"

                    email = EmailMessage(
                        subject,
                        message_body,
                        'weapondetectioncnn@gmail.com',
                        [to_email],
                    )

                    # Read image and attach
                    with open(image_path, 'rb') as f:
                        image_data = f.read()

                    email.attach(uploaded_file.name, image_data, uploaded_file.content_type)
                    email.send(fail_silently=False)
                    print("Email sent successfully with image attachment.")

                prediction = f"Prediction: {predicted_class}, Probability: {max(probs) * 100:.2f}%"
                return render(request, 'predict_result.html', {
                    'prediction': predicted_class,
                    'location': location_text,
                    'predicted_class': predicted_class
                })

            except Exception as e:
                print("Error during prediction:", e)

        return render(request, 'predict_result.html', {
            'prediction': prediction,
            'result_predicted': predicted_class if prediction else None
        })

    else:
        return render(request, 'predict_result.html', {'error': "No object detected."})