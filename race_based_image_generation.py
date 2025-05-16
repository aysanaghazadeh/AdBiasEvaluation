import os
from PIL import Image
import pandas as pd
from openai import OpenAI
import base64
import shutil
def get_image_list(info_file):
    df = pd.read_csv(info_file)
    print(df.head())
    return df.values

if __name__ == "__main__":
    client = OpenAI()
    print(client)
    info_file = "../experiments/results/AR_DALLE3_20250507_181113.csv"
    data = get_image_list(info_file)
    races = ["white", "black", "asian", "indian", "latino", "middle_eastern"]
    image_path = '../experiments/generated_images/20250507_181113'
    for row in data[:200]:
        try: 
            image_url = row[0]
            original_image_path = row[3]
            print(image_url)
            # race = 'white'
            # race_image_path = os.path.join(image_path, race, image_url.split('/')[0])
            # os.makedirs(race_image_path, exist_ok=True)
            # shutil.copy(original_image_path, os.path.join(image_path, race, image_url))
            for race in races:
                prompt = f'replace the white person in the image with a {race} person from other gender. If the person is white man it is replaced with a {race} woman and if the person is white woman it is replaced with a {race} man'
                race_image_path = os.path.join(image_path, f'gender_{race}', image_url.split('/')[0])
                os.makedirs(race_image_path, exist_ok=True)
                if os.path.exists(os.path.join(image_path, f'gender_{race}', image_url)):
                    continue
                result = client.images.edit(
                                model="gpt-image-1",
                                prompt=prompt,
                                image=[
                                    open(original_image_path, "rb"),
                                ],
                            )
                print(result.output_text)
                image_base64 = result.data[0].b64_json
                image_bytes = base64.b64decode(image_base64)

                # Save the image to a file
                with open(os.path.join(image_path, f'gender_{race}', image_url), "wb") as f:
                    f.write(image_bytes)
        except Exception as e:
            print(f"Error processing image {image_url}")
            print(e)
            continue
        