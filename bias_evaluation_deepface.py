from deepface import DeepFace
import cv2
import json
import os
import pandas as pd

image_analysis = {}
def analyze_image(image_path, image_url):
    """
    Analyze gender and race of people in an image.

    :param image_path: Path to the image file
    """
    try:
        # Analyze the image
        analysis = DeepFace.analyze(img_path=image_path, actions=['gender', 'race'])
        faces = []
        for idx, face in enumerate(analysis):
            face_info = {}
            face_info['Gender'] = face['dominant_gender']
            face_info['Race'] = face['dominant_race']
            # face_info['GenderDetail'] = face['gender']
            # face_info['RaceDetail'] = face['race']
            print(f"Person {idx + 1}:")
            print(f"  Gender: {face['dominant_gender']}")
            print(f"  Race: {face['dominant_race']}")
            print(f"  Gender Details: {face['gender']}")
            print(f"  Race Details: {face['race']}")
            print()
            faces.append(face_info)
        image_analysis[image_url] = faces
    except Exception as e:
        print(f"Error analyzing image: {e}")
    print(image_analysis)

# Example usage




if __name__ == "__main__":
    analyze_image(args)
    image_path_list = pd.read_csv(os.path.join('../experiments/results', 'AR_Flux_20250508_202121.csv')).generated_image_url.values
    ref_path_list = pd.read_csv(os.path.join('../experiments/results', 'AR_Flux_20250508_202121.csv')).generated_image_url.values
    for row in image_path_list:
        image_url = row[0]
        if image_url not in ref_path_list:
            continue
        image_path = row[1]
        analyze_image(image_path, image_url)
        with open(os.path.join('../experiments/results', 'deepface_analysis_FLUX.json'), "w") as f:
            json.dump(image_analysis, f)