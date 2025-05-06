import pandas as pd
import json
import ImageReward as RM
model = RM.load("ImageReward-v1.0")



black_set = pd.read_csv('../experiments/results/AR_Flux_20250505_222356.csv').values
white_set = pd.read_csv('../experiments/results/AR_Flux_20250506_004816.csv').values

images_scores = {}
# clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xl') # our recommended scoring model


for i, row in enumerate(black_set):
    if i >  len(white_set):
        break
    image_url = row[0]
    action_reason = row[1]
    image_url_black = row[3]
    image_url_white = white_set[i][3]
    score_black = model.score(action_reason, [image_url_black])
    score_white = model.score(action_reason, [image_url_white])
    images_scores[image_url] = [score_black, score_white, 1 if score_white > score_black else 0]

white_chosen = 0
white_scores, black_scores = 0, 0
for image_url in images_scores:
    if images_scores[image_url][2] == 1:
        white_chosen += 1
    white_scores += images_scores[image_url][1]
    black_scores += images_scores[image_url][0]

print(white_chosen / len(images_scores))
print(white_scores / len(images_scores))
print(black_scores / len(images_scores))