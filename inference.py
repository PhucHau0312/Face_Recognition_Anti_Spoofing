import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')



def infer(model_dir, device_id):
    model = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Fail to open camera"
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        image_bbox = model.get_bbox(image)
        prediction = np.zeros((1, 3))
        test_speed = 0
        for model_name in os.listdir(model_dir):
            h_input, w_input, _, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            prediction += model.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time()-start

        label = np.argmax(prediction)
        value = prediction[0][label]/2
        if label == 1:
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        print(result_text)
        print("Prediction cost {:.2f} s".format(test_speed))
        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

        cv2.imshow('Image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    args = parser.parse_args()
    infer(args.model_dir, args.device_id)
