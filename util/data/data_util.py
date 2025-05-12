import random
from accelerate import PartialState
from util.data.trian_test_split import get_train_data
from transformers import AutoTokenizer
from datasets import Dataset, Features, Sequence, Value, Array3D
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop
import numpy as np
import cv2
# import mediapipe as mp
from PIL import Image, ImageDraw
from pathlib import Path

from PIL import Image
from PIL.ImageOps import exif_transpose
import itertools

from util.data.mapping import *


def get_SD_training_data(args, image_urls, pipe):
    # Image preprocessing function
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def preprocess(examples):
        # Process text inputs
        positive_inputs = pipe.tokenizer(
            examples["positive_text"],
            padding="max_length", 
            truncation=True, 
            max_length=77,
            return_tensors=None
        )
        negative_inputs = pipe.tokenizer(
            examples["negative_text"],
            padding="max_length", 
            truncation=True, 
            max_length=77,
            return_tensors=None
        )

        # Process images
        processed_images = [transform(img.convert("RGB")).numpy() for img in examples["image"]]

        return {
            "positive_input_ids": positive_inputs["input_ids"],
            "negative_input_ids": negative_inputs["input_ids"],
            "image": processed_images,
            "positive_text": examples["positive_text"],
            "negative_text": examples["negative_text"]
        }

    # Create initial dataset with correct column names
    dataset = {'image': [], 'positive_text': [], 'negative_text': []}
    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    negative_QAs = {}
    negative_QAs['reason'] = json.load(
        open(os.path.join(args.data_path, 'train/reason_hard_QA_Combined_Action_Reason_train.json')))
    negative_QAs['action'] = json.load(
        open(os.path.join(args.data_path, 'train/action_hard_QA_Combined_Action_Reason_train.json')))
    negative_QAs['adjective'] = json.load(
        open(os.path.join(args.data_path, 'train/adjective_hard_QA_Combined_Action_Reason_train.json')))
    negative_QAs['semantic'] = json.load(
        open(os.path.join(args.data_path, 'train/semantic_hard_QA_Combined_Action_Reason_train.json')))

    for image_url in image_urls[:10]:
        if image_url in negative_QAs['reason']:
            QA = QAs[image_url]
            for AR in QA[0]:
                for negative_type in negative_QAs:
                    for negative_option in negative_QAs[negative_type][image_url][1]:
                        if (negative_option in negative_QAs[negative_type][image_url][0]) or (negative_option in dataset['negative_text']):
                            continue
                        dataset['image'].append(Image.open(os.path.join(args.data_path, args.train_set_images, image_url)))
                        prompt = f"""Generate an advertisement image that conveys the following message:"""
                        positive = f'{prompt}\n{AR}'
                        negative = f'{prompt}\n{negative_option}'
                        dataset['positive_text'].append(positive)
                        dataset['negative_text'].append(negative)
    dataset = Dataset.from_dict(dataset)
    processed_dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=1,
        remove_columns=dataset.column_names,
        features=Features({
            "positive_input_ids": Sequence(Value("int64")),
            "negative_input_ids": Sequence(Value("int64")),
            "image": Array3D(shape=(3, 1024, 1024), dtype="float32"),
            "positive_text": Value("string"),
            "negative_text": Value("string")
        })
    )
    return processed_dataset


def get_train_SD_Dataloader(args, pipe):
    def collate_fn(examples):
        # Convert lists to tensors and stack them
        return {
            "positive_input_ids": torch.tensor([e["positive_input_ids"] for e in examples]),
            "negative_input_ids": torch.tensor([e["negative_input_ids"] for e in examples]),
            "image": torch.tensor([e["image"] for e in examples]),
            "positive_text": [e["positive_text"] for e in examples],
            "negative_text": [e["negative_text"] for e in examples]
        }

    image_urls = list(json.load(open(os.path.join(args.data_path,
                                               'train/combined_hard_QA_Combined_Action_Reason_train.json'))).keys())
    dataset = get_SD_training_data(args, image_urls, pipe)
    
    # Create DataLoader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # Use batch size from config instead of hardcoded value
        shuffle=True,
        collate_fn=collate_fn
    )
    return dataloader


def get_SDXL_training_data(args, image_urls, accelerator, tokenizer_one, tokenizer_two):
    # Image preprocessing function
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    def tokenize_prompt(tokenizer, prompt):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids
    
    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
        prompt_embeds_list = []

        for i, text_encoder in enumerate(text_encoders):
            if tokenizers is not None:
                tokenizer = tokenizers[i]
                text_input_ids = tokenize_prompt(tokenizer, prompt)
            else:
                assert text_input_ids_list is not None
                text_input_ids = text_input_ids_list[i]

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds
    

    # Create initial dataset with correct column names
    dataset = {'image': [], 'positive_text': [], 'negative_text': []}
    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    negative_QAs = {}
    negative_QAs['reason'] = json.load(
        open(os.path.join(args.data_path, 'train/reason_hard_QA_Combined_Action_Reason_train.json')))
    negative_QAs['action'] = json.load(
        open(os.path.join(args.data_path, 'train/action_hard_QA_Combined_Action_Reason_train.json')))
    negative_QAs['adjective'] = json.load(
        open(os.path.join(args.data_path, 'train/adjective_hard_QA_Combined_Action_Reason_train.json')))
    negative_QAs['semantic'] = json.load(
        open(os.path.join(args.data_path, 'train/semantic_hard_QA_Combined_Action_Reason_train.json')))

    for image_url in image_urls[:100]:
        if image_url in negative_QAs['reason']:
            QA = QAs[image_url]
            for AR in QA[0][:1]:
                for negative_type in negative_QAs:
                    if negative_type != 'reason':
                        continue
                    for negative_option in negative_QAs[negative_type][image_url][1][:1]:
                        if (negative_option in negative_QAs[negative_type][image_url][0]) or (negative_option in dataset['negative_text']):
                            continue
                        dataset['image'].append(Image.open(os.path.join(args.data_path, args.train_set_images, image_url)))
                        prompt = f"""Generate an advertisement image that conveys the following message:"""
                        positive = f'{prompt}\n{AR}'
                        negative = f'{prompt}\n{negative_option}'
                        dataset['positive_text'].append(positive)
                        dataset['negative_text'].append(negative)
    # dataset = {'train': dataset}
    dataset = Dataset.from_dict(dataset)
    dataset_columns = ['image', 'positive_text', 'negative_text']
    image_column = dataset_columns[0]
    pos_caption_column = dataset_columns[1]
    neg_caption_column = dataset_columns[2]
    
    def tokenize_captions(examples, is_train=True, caption_column=pos_caption_column):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        tokens_one = tokenize_prompt(tokenizer_one, captions)
        tokens_two = tokenize_prompt(tokenizer_two, captions)
        return tokens_one, tokens_two

    # Preprocessing the datasets.
    train_resize = transforms.Resize(
        args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
    )
    train_crop = (
        transforms.CenterCrop(args.resolution)
        if args.center_crop
        else transforms.RandomCrop(args.resolution)
    )
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose(
        [   
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        image_column = "image"
        images = [image.convert("RGB") for image in examples[image_column]]
        images[0].save('../experiments/generated_images/test.png')
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(
                    image, (args.resolution, args.resolution)
                )
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        pos_tokens_one, pos_tokens_two = tokenize_captions(examples, caption_column=pos_caption_column)
        neg_tokens_one, neg_tokens_two = tokenize_captions(examples, caption_column=neg_caption_column)
        examples["pos_input_ids_one"] = pos_tokens_one
        examples["pos_input_ids_two"] = pos_tokens_two
        examples["neg_input_ids_one"] = neg_tokens_one
        examples["neg_input_ids_two"] = neg_tokens_two
        if args.debug_loss:
            fnames = [
                os.path.basename(image.filename)
                for image in examples[image_column]
                if image.filename
            ]
            if fnames:
                examples["filenames"] = fnames
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(
                range(args.max_train_samples)
            )

        # Set the training transforms
        train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        pos_input_ids_one = torch.stack([example["pos_input_ids_one"] for example in examples])
        pos_input_ids_two = torch.stack([example["pos_input_ids_two"] for example in examples])
        neg_input_ids_one = torch.stack([example["neg_input_ids_one"] for example in examples])
        neg_input_ids_two = torch.stack([example["neg_input_ids_two"] for example in examples])

        # Get the original sizes and crop coordinates
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]

        return {
            "pixel_values": pixel_values,
            "pos_input_ids_one": pos_input_ids_one,
            "pos_input_ids_two": pos_input_ids_two,
            "neg_input_ids_one": neg_input_ids_one,
            "neg_input_ids_two": neg_input_ids_two,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    return train_dataloader

def get_train_SDXL_Dataloader(args, accelerator, tokenizer_one, tokenizer_two):
    image_urls = list(json.load(open(os.path.join(args.data_path,
                                               'train/combined_hard_QA_Combined_Action_Reason_train.json'))).keys())
    dataloader = get_SDXL_training_data(args, image_urls, accelerator, tokenizer_one, tokenizer_two)
    return dataloader

def detect_image_orientation(image_path):
    """
    Detect the orientation of an image and return whether it's landscape, portrait, or square.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: 'landscape', 'portrait', or 'square'
        tuple: (width, height) of the image
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Check if image is square
            if width == height:
                return 'square', (width, height)
            # Check if image is landscape
            elif width > height:
                return 'landscape', (width, height)
            # Image is portrait
            else:
                return 'portrait', (width, height)
                
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None

# Example usage
def get_image_stats(image_path):
    """
    Get detailed statistics about an image's orientation and aspect ratio.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary containing image statistics
    """
    orientation, dimensions = detect_image_orientation(image_path)
    
    if dimensions:
        width, height = dimensions
        aspect_ratio = width / height
        
        return {
            'orientation': orientation,
            'width': width,
            'height': height,
            'aspect_ratio': round(aspect_ratio, 2)
        }
    return None

def detect_object_orientations(image_path):
    """
    Detect objects in an image and determine their orientations based on bounding boxes.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        list: List of dictionaries containing object information including:
              - label: object class
              - orientation: 'vertical', 'horizontal', or 'square'
              - confidence: detection confidence
              - bbox: bounding box coordinates
              - aspect_ratio: width/height ratio of the object
    """
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    try:
        # Load and process image
        image = Image.open(image_path)
        results = model(image)
        
        # Process detections
        objects = []
        for *box, conf, cls in results.xyxy[0]:  # xyxy format
            x1, y1, x2, y2 = [int(coord) for coord in box]
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height
            
            # Determine orientation based on bounding box
            if abs(aspect_ratio - 1.0) < 0.1:  # Allow small deviation from perfect square
                orientation = 'square'
            elif width > height:
                orientation = 'horizontal'
            else:
                orientation = 'vertical'
                
            objects.append({
                'label': results.names[int(cls)],
                'orientation': orientation,
                'confidence': float(conf),
                'bbox': (x1, y1, x2, y2),
                'aspect_ratio': round(aspect_ratio, 2)
            })
            
        return objects
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def visualize_object_orientations(image_path, objects):
    """
    Visualize detected objects and their orientations on the image.
    
    Args:
        image_path (str): Path to the image file
        objects (list): List of detected objects from detect_object_orientations()
    
    Returns:
        PIL.Image: Image with visualized object orientations
    """
    try:
        import cv2
        
        # Load image
        image = cv2.imread(image_path)
        
        # Draw bounding boxes and orientations
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            label = f"{obj['label']} ({obj['orientation']})"
            
            # Different colors for different orientations
            color = {
                'horizontal': (0, 255, 0),  # Green
                'vertical': (0, 0, 255),    # Red
                'square': (255, 0, 0)       # Blue
            }[obj['orientation']]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
    except Exception as e:
        print(f"Error visualizing image {image_path}: {str(e)}")
        return None

# Example usage
def analyze_image_objects(image_path, visualize=True):
    """
    Analyze and optionally visualize object orientations in an image.
    
    Args:
        image_path (str): Path to the image file
        visualize (bool): Whether to create a visualization
        
    Returns:
        tuple: (objects_info, visualization_image)
    """
    objects = detect_object_orientations(image_path)
    
    if objects is None:
        return None, None
        
    if visualize:
        viz_image = visualize_object_orientations(image_path, objects)
        return objects, viz_image
    
    return objects, None

def detect_object_facing_direction(image_path):
    """
    Detect the direction that objects (particularly faces and bodies) are facing using MediaPipe.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        list: List of dictionaries containing object information including:
              - type: 'face' or 'body'
              - direction: angle in degrees (0-360)
              - confidence: detection confidence
              - bbox: bounding box coordinates
    """
    # Initialize MediaPipe Face Detection and Pose Detection
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not read image")
            
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        objects = []
        
        # Detect faces
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            face_results = face_detection.process(image_rgb)
            
            if face_results.detections:
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    
                    # Convert relative coordinates to absolute
                    xmin = int(bbox.xmin * w)
                    ymin = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Get face direction from keypoints
                    keypoints = detection.location_data.relative_keypoints
                    right_eye = (int(keypoints[0].x * w), int(keypoints[0].y * h))
                    left_eye = (int(keypoints[1].x * w), int(keypoints[1].y * h))
                    
                    # Calculate face direction angle
                    dx = right_eye[0] - left_eye[0]
                    dy = right_eye[1] - left_eye[1]
                    angle = np.degrees(np.arctan2(dy, dx))
                    
                    # Normalize angle to 0-360 range
                    angle = (angle + 360) % 360
                    
                    objects.append({
                        'type': 'face',
                        'direction': angle,
                        'confidence': detection.score[0],
                        'bbox': (xmin, ymin, xmin + width, ymin + height)
                    })
        
        # Detect body pose
        with mp_pose.Pose(min_detection_confidence=0.5) as pose_detection:
            pose_results = pose_detection.process(image_rgb)
            
            if pose_results.pose_landmarks:
                # Get body direction from shoulders
                landmarks = pose_results.pose_landmarks.landmark
                left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                               int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
                right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                                int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
                
                # Calculate body direction angle
                dx = right_shoulder[0] - left_shoulder[0]
                dy = right_shoulder[1] - left_shoulder[1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Normalize angle to 0-360 range
                angle = (angle + 360) % 360
                
                # Estimate bounding box from pose landmarks
                x_coords = [landmark.x * w for landmark in landmarks if landmark.visibility > 0.5]
                y_coords = [landmark.y * h for landmark in landmarks if landmark.visibility > 0.5]
                
                if x_coords and y_coords:
                    xmin, xmax = int(min(x_coords)), int(max(x_coords))
                    ymin, ymax = int(min(y_coords)), int(max(y_coords))
                    
                    objects.append({
                        'type': 'body',
                        'direction': angle,
                        'confidence': np.mean([landmark.visibility for landmark in landmarks]),
                        'bbox': (xmin, ymin, xmax, ymax)
                    })
        
        return objects
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def visualize_object_directions(image_path, objects):
    """
    Visualize detected objects and their facing directions on the image.
    """
    try:
        # Load image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        for obj in objects:
            xmin, ymin, xmax, ymax = obj['bbox']
            
            # Different colors for faces and bodies
            color = 'red' if obj['type'] == 'face' else 'blue'
            
            # Draw bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
            
            # Calculate arrow endpoint for direction visualization
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            angle = obj['direction'] * np.pi / 180
            arrow_length = min(xmax - xmin, ymax - ymin) / 2
            
            end_x = center_x + arrow_length * np.cos(angle)
            end_y = center_y + arrow_length * np.sin(angle)
            
            # Draw direction arrow
            draw.line([(center_x, center_y), (end_x, end_y)], 
                     fill='green', width=2)
            
            # Add label
            label = f"{obj['type']}: {int(obj['direction'])}°"
            draw.text((xmin, ymin-15), label, fill=color)
        
        return image
        
    except Exception as e:
        print(f"Error visualizing image {image_path}: {str(e)}")
        return None

def analyze_object_directions(image_path, visualize=True):
    """
    Analyze and optionally visualize object facing directions in an image.
    """
    objects = detect_object_facing_direction(image_path)
    
    if objects is None:
        return None, None
        
    if visualize:
        viz_image = visualize_object_directions(image_path, objects)
        return objects, viz_image
    
    return objects, None



def get_PPO_training_data(args, image_urls, tokenizer):
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
    #                                           #"meta-llama/Meta-Llama-3-8B-instruct",
    #                                           token='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',
    #                                           trust_remote_code=True,
    #                                           padding='right')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Set chat template based on model type
    if "qwen" in args.model_name.lower():
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    elif "llama" in args.model_name.lower():
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '\n' }}{% endif %}{% endfor %}"
    
    def process(row):
        # Apply chat template only if the prompt is in the correct format
        if isinstance(row["prompt"], list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in row["prompt"]):
            row["prompt"] = tokenizer.apply_chat_template(row["prompt"], tokenize=False)
        return row
    
    # Load QA data
    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    
    # Create dataset with proper prompt formatting
    dataset = {'prompt': []}
    for image_url in image_urls:
        if image_url in QAs:
            QA = QAs[image_url]
            # Format prompt based on model type
            if "qwen" in args.model_name.lower() or "llama" in args.model_name.lower():
                prompt = f"""Describe an advertisement image that conveys the following message in detail:

                            - {'\n-'.join(QA[0])}

                            Only return one paragraph of description without further explanation. Description of the image:"""
                prompt = [{'content': prompt, 'role': 'user'}]
            else:
                prompt = f"""Describe an advertisement image that conveys the following message in detail:

                            - {'\n-'.join(QA[0])}

                            Only return one paragraph of description without further explanation. Description of the image:"""
            dataset['prompt'].append(prompt)
    
    # Convert to dataset and process
    dataset = Dataset.from_dict(dataset)
    with PartialState().local_main_process_first():
        ds = dataset.map(process)
    
    return ds


def get_train_LLAMA3_PPO_Dataset(args):
    image_urls = get_train_data(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = get_PPO_training_data(args, image_urls, tokenizer)
    return dataset


def get_train_DDPO_persuasion_Dataset(args):
    image_urls = get_train_data(args)
    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    prompts = []
    for image_url in image_urls:
        prompt = f"""Generate an advertisement image that conveys the following message in detail: \n {'\n'.join(QAs[image_url][0])}"""
        prompts.append(prompt)
    
    return prompts



class DreamBoothDataset_modified(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        args,
        size=1024,
        repeats=1,
        center_crop=False,
    ):
        def load_train_data(args):
            train_set_images = get_train_data(args)
            
            print(f"Total training images available: {len(train_set_images)}")
            QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
            image_country_map = json.load(open(os.path.join(args.data_path, 'train/image_country_map.json')))
            image_cultural_components_map = json.load(open(os.path.join(args.data_path, 'train/components.json')))
            country_image_map = json.load(open(os.path.join(args.data_path, 'train/countries_image_map.json')))
            
            dataset = {
                'image': [], 
                'positive_text': [], 
                'reasons': [],
                'cultural_components': [],
                'style_images': [],
                'country': []
            }
            
            for image_url in train_set_images:
                # Get atypicality information for this image
                country = image_country_map[image_url]
                if len(country_image_map[country]) > 5:
                    style_images = random.sample(country_image_map[country], 5)
                else:
                    style_images = country_image_map[country]
                cultural_components = ''
                for i in style_images:
                    cultural_components += ' ' + ' '.join(image_cultural_components_map[i])
                style_image = style_images[0]
                dataset['style_images'].append(style_image)
                dataset['country'].append(country)
                dataset['cultural_components'].append(cultural_components)
                dataset['reasons'].append(QAs[image_url][0][-1].lower().split('because')[-1])
                dataset['image'].append(Image.open(os.path.join(args.data_path, args.train_set_images, image_url)))
                prompt = f"""Generate an advertisement image that targets the people from {country} and conveys the following message in detail: \n -{'\n-'.join(QAs[image_url][0])}"""
                dataset['positive_text'].append(prompt)
            
            print(f"Final dataset size: {len(dataset['image'])} samples")
            return dataset
            
        self.size = size
        self.center_crop = center_crop

        self.args = args
        self.custom_instance_prompts = None

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        
        self.instance_data_root = Path(args.data_path)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        dataset = load_train_data(args)
        self.custom_instance_prompts = dataset['positive_text']
        self.instance_positive_text = dataset['positive_text']
        self.country = dataset['country']
        self.cultural_components = dataset['cultural_components']
        self.reasons = dataset['reasons']
        self.style_images = dataset['style_images']
        self.instance_images = []
        for img in dataset['image']:
            self.instance_images.extend(itertools.repeat(img, repeats))

        self.pixel_values = []
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            image = train_transforms(image)
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images
        
        self.image_transforms = transforms.Compose(
            [
                # transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # Handle both single index and list of indices
        if isinstance(index, (list, tuple)):
            instance_images = [self.pixel_values[i] for i in index]
            instance_prompts = [self.instance_positive_text[i] for i in index]
            style_images = [self.style_images[i] for i in index]
            country = [self.country[i] for i in index]
            cultural_components = [self.cultural_components[i] for i in index]
            reasons = [self.reasons[i] for i in index]
            # print(atypicality_image)
            # print(len(index))
            example["instance_images"] = torch.stack(instance_images)
            example["instance_prompt"] = instance_prompts
            example["style_images"] = style_images
            example["country"] = country
            example["cultural_components"] = cultural_components
            example["reasons"] = reasons
        else:
            instance_image = self.pixel_values[index]
            example["instance_images"] = instance_image
            example["instance_prompt"] = self.instance_positive_text[index]
            example["style_image"] = self.style_images[index]
            example["country"] = self.country[index]
            example["cultural_components"] = self.cultural_components[index]
            example["reasons"] = self.reasons[index]
        return example


def collate_fn_modified(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    style_images = [example["style_images"] for example in examples]
    country = [example["country"] for example in examples]
    cultural_components = [example["cultural_components"] for example in examples]
    reasons = [example["reasons"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]
        country += [example["country"] for example in examples]
        cultural_components += [example["cultural_components"] for example in examples]
        reasons += [example["reasons"] for example in examples]
        style_images += [example["style_images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    style_images = torch.stack(style_images)
    style_images = style_images.to(memory_format=torch.contiguous_format).float()
    batch = {"pixel_values": pixel_values, "prompts": prompts, "country": country, "cultural_components": cultural_components, "reasons": reasons, "style_images": style_images}
    return batch


class PromptDataset_modified(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["country"] = self.country
        example["cultural_components"] = self.cultural_components
        example["reasons"] = self.reasons
        example["index"] = index
        return example

def tokenize_prompt_modified(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5_modified(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip_modified(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt_modified(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip_modified(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5_modified(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds