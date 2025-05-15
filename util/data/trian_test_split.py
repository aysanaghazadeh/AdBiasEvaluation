import os
import pandas as pd
import random
import csv
import json
from collections import Counter, defaultdict
from util.data.mapping import TOPIC_MAP as topic_map


def get_train_data(args):
    train_file = os.path.join(args.data_path, 'train/country_train_image_large.csv')
    
    if os.path.exists(train_file):
        return pd.read_csv(train_file).ID.values
    if 'country' in train_file:
        train_image_urls = []
        country_image_map = json.load(open(os.path.join(args.data_path, 'train/countries_image_map.json')))
        for country in country_image_map:
            if len(country_image_map[country]) > 10:
                train_image_urls += random.sample(country_image_map[country], 5)
            # else:
            #     train_image_urls += list(country_image_map[country])
        with open(train_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID'])
            for i in train_image_urls:
                writer.writerow([i])
        return train_image_urls
    else:
        train_image_urls = pd.read_csv(train_file).ID.values
    if os.path.exists(os.path.join(args.data_path, 'Action_Reason_statements.json')):
        QA_base = json.load(open(os.path.join(args.data_path, 'Action_Reason_statements.json')))
    else:
        QA_base = {}
    if os.path.exists(os.path.join(args.data_path, 'train/test_image.csv')):
        test_files = set(list(pd.read_csv(os.path.join(args.data_path, 'train/test_image.csv')).ID.values))
    else:
        test_files = set()
    QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    train_QA = {}
    for image_url in QA:
        if image_url not in test_files:
            train_QA[image_url] = QA[image_url]
    # image_urls = list(QA.keys())
    # print(len(image_urls))
    # train_size = int(args.train_ratio * len(image_urls))
    # train_image_urls = random.sample(image_urls, train_size)
    train_image_urls = train_QA.keys()
    train_size = int(args.train_ratio * len(train_image_urls))
    train_image_urls = random.sample(train_image_urls, train_size)
    print(f'train size is: {len(train_image_urls)}')
    print('saving train data')
    with open(train_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['ID'])

        # Write the data
        for i in train_image_urls:
            writer.writerow([i])
    return pd.read_csv(train_file)


def get_test_data(args):
    topics_data_file = os.path.join(args.data_path, 'train/Topics_train.json')
    test_file = os.path.join(args.data_path, f'train/test_set_images_{args.AD_type}.csv')
    test_file = os.path.join(args.data_path, f'train/test_set_images_country.csv')
    if os.path.exists(test_file):
        return pd.read_csv(test_file)
    if 'country' in test_file:
        train_image_urls = pd.read_csv(os.path.join(args.data_path, 'train/country_train_image_large.csv')).ID.values
        test_image_urls = []
        country_image_map = json.load(open(os.path.join(args.data_path, 'train/countries_image_map.json')))
        target_countries = ['india', 'china', 'united states', 'united arab emirates', 'france', 'saudi arabia', 'japan', 'south korea', 'south africa', 'mexico', 'Turkey']
        for country in country_image_map:
            if country not in target_countries:
                continue
            country_image_urls = set([])
            if len(country_image_map[country]) > 14:
                while len(country_image_urls) < 5:
                    random_idx = random.randint(0, len(country_image_map[country]) - 1)
                    image_url = country_image_map[country][random_idx]
                    if image_url not in train_image_urls:
                        country_image_urls.add(image_url)
                test_image_urls += list(country_image_urls)
            elif len(country_image_map[country]) > 5:
                while len(country_image_urls) < 5:
                    random_idx = random.randint(0, len(country_image_map[country]) - 1)
                    image_url = country_image_map[country][random_idx]
                    country_image_urls.add(image_url)
            else:
                test_image_urls += list(country_image_map[country])
        with open(test_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID'])
            for i in test_image_urls:
                writer.writerow([i])
        return test_image_urls
        
    if args.AD_type == 'all':
        # Take exactly 290 samples from each dataset (PSA and Commercial)
        test_set = pd.concat([
            pd.read_csv(os.path.join(args.data_path, f'train/test_set_images_PSA.csv')).ID.head(290),
            pd.read_csv(os.path.join(args.data_path, f'train/test_set_images_COM.csv')).ID.head(290)
        ], ignore_index=True)
        with open(test_file, 'w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(['ID'])

            for filename in test_set:
                writer.writerow([filename])
        return test_set
    
    topics_data = json.load(open(topics_data_file))
    all_topics = [topic for topics in topics_data.values() for topic in set(topics)]
    topic_counter = Counter(all_topics)
    most_common_topics = [topic for topic, count in topic_counter.most_common(10)]
    selected_files = defaultdict(list)
    train_files = get_train_data(args)
    QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    for file, topics in topics_data.items():
        if file in train_files or file not in QA:
            continue
        for topic in set(topics):
            if topic in most_common_topics:
                if int(topic) in topic_map:
                    if len(selected_files[topic]) < 300:
                        selected_files[topic].append(file)
    print('saving test files...')
    with open(test_file, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['ID', 'topic'])

        for topic, files in selected_files.items():
            for filename in files:
                writer.writerow([filename, '-'.join(topic_map[int(topic)])])
    return pd.read_csv(test_file)

