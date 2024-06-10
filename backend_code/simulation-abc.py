from al import recommend_superpixels, train, ann_to_labels
import cv2
import numpy as np
from PIL import Image
import config
import os
import time
from tqdm import tqdm

def elevation_guided_BFS(new_labels, dem, selected_superpixels, superpixels_group):
    height, width = new_labels.shape
    bfs_visited = np.zeros((height, width)).astype('int')

    # group superpixel into flood and dry pixels
    # flood: start BFS from highest 5 pixels
    # dry: start BFS from lowest 5 pixels
    # if not ~ 90% annotated, select next 5 pixels each
    # continue till ~90% is reached

    for sid in tqdm(selected_superpixels):

        pixels = selected_superpixels[sid]
        total_pixels = len(pixels) # total pixels in this superpixel

        bfs_queue = []
        annotated_percent = 0

        flood_pixels = []
        dry_pixels = []
        for (j, i) in pixels: # height, width
            elev = dem[j][i]
            if new_labels[j][i] == 1:
                flood_pixels.append(((j,i), elev))
            elif new_labels[j][i] == -1:
                dry_pixels.append(((j,i), elev))

        flood_pixels = sorted(flood_pixels, key=lambda x: x[1], reverse=True)
        dry_pixels = sorted(dry_pixels, key=lambda x: x[1])

        budget = 5
        flood_expense = 0
        dry_expense = 0

        annotation_count = 0
        while annotated_percent <= 0.9:
            if flood_expense < (len(flood_pixels) - 1):
                end = flood_expense+budget
                if end > len(flood_pixels):
                    end = len(flood_pixels)
                for f_p in flood_pixels[flood_expense:end]:
                    (j,i), elev = f_p

                    annotation_count += 1

                    # this pixel might be selected twice
                    if bfs_visited[j][i]:
                        continue

                    bfs_queue.append((j,i))
                    bfs_visited[j][i] = 1

                    while bfs_queue:
                        (j, i) = bfs_queue.pop(0)
                        bfs_visited[j][i] = 1

                        # go through the 8 neighbors
                        for l in [-1, 0, 1]:
                            for r in [-1, 0, 1]:
                                if (l == r == 0):
                                    continue

                                j_nei, i_nei = (j+l, i+r) # get the neighboring i and j

                                # check for boundary cases
                                if i_nei < 0 or j_nei < 0 or i_nei >= width or j_nei >= height:
                                    continue

                                # check if already visited or not
                                if bfs_visited[j_nei][i_nei]:
                                    continue

                                # check current pixel's elevation with neighbor's elevation
                                if (new_labels[j][i] == 1 and (dem[j_nei][i_nei] <= dem[j][i]) and (new_labels[j_nei][i_nei] != 1)):
                                    new_labels[j_nei][i_nei] = 1
                                    bfs_queue.append((j_nei, i_nei))
                                elif (new_labels[j][i] == -1 and (dem[j_nei][i_nei] >= dem[j][i]) and (new_labels[j_nei][i_nei] != -1)):
                                    new_labels[j_nei][i_nei] = -1
                                    bfs_queue.append((j_nei, i_nei))

            if dry_expense < (len(dry_pixels) - 1):
                end = dry_expense+budget
                if end > len(dry_pixels):
                    end = len(dry_pixels)
                for d_p in dry_pixels[dry_expense:end]:
                    (j,i), elev = d_p

                    annotation_count += 1

                    # this pixel might be selected twice
                    if bfs_visited[j][i]:
                        continue

                    bfs_queue.append((j,i))
                    bfs_visited[j][i] = 1

                    while bfs_queue:
                        (j, i) = bfs_queue.pop(0)
                        bfs_visited[j][i] = 1

                        # go through the 8 neighbors
                        for l in [-1, 0, 1]:
                            for r in [-1, 0, 1]:
                                if (l == r == 0):
                                    continue

                                j_nei, i_nei = (j+l, i+r) # get the neighboring i and j

                                # check for boundary cases
                                if i_nei < 0 or j_nei < 0 or i_nei >= width or j_nei >= height:
                                    continue

                                # check if already visited or not
                                if bfs_visited[j_nei][i_nei]:
                                    continue

                                # check current pixel's elevation with neighbor's elevation
                                if (new_labels[j][i] == 1 and (dem[j_nei][i_nei] <= dem[j][i]) and (new_labels[j_nei][i_nei] != 1)):
                                    new_labels[j_nei][i_nei] = 1
                                    bfs_queue.append((j_nei, i_nei))
                                elif (new_labels[j][i] == -1 and (dem[j_nei][i_nei] >= dem[j][i]) and (new_labels[j_nei][i_nei] != -1)):
                                    new_labels[j_nei][i_nei] = -1
                                    bfs_queue.append((j_nei, i_nei))

            annotated_percent = annotation_count / total_pixels
            flood_expense += budget
            dry_expense += budget

    return new_labels


def acquire_labels(i, lambda_1, probability, entropy, cod, transformation_agg, superpixel_agg, ent_var, TEST_REGION, selected_superpixels, superpixels_group):
    recommended_superpixels = cv2.imread(f'./users/{student_id}/output/R{TEST_REGION}_superpixels_test.png')
    recommended_superpixels = np.sum(recommended_superpixels, axis=-1)
    recommended_superpixels = np.where(recommended_superpixels > 0)
    gt = np.load(f"./data_al/repo/groundTruths/Region_{TEST_REGION}_GT_Labels.npy")

    try:
        new_labels = ann_to_labels(f'./users/{student_id}/output/R{TEST_REGION}_labels.png', TEST_REGION)
    except:
        new_labels = np.zeros((gt.shape[0], gt.shape[1]))

    new_labels[recommended_superpixels] = gt[recommended_superpixels]

    # TODO: Run elevation-guided BFS
    dem = np.load(f"./data_al/repo/Features_7_Channels/Region_{TEST_REGION}_Features7Channel.npy")[:,:,3]
    new_labels = elevation_guided_BFS(new_labels, dem, selected_superpixels, superpixels_group)

    flood_labels = np.where(new_labels == 1, 1, 0)
    dry_labels = np.where(new_labels == -1, 1, 0)
    flood_labels = np.expand_dims(flood_labels, axis=-1)
    dry_labels = np.expand_dims(dry_labels, axis=-1)
    flood_labels = flood_labels*np.array([ [ [255, 0, 0] ] ])
    dry_labels = dry_labels*np.array([ [ [0, 0, 255] ] ])
    final_labels = (flood_labels + dry_labels).astype('uint8')
    pim = Image.fromarray(final_labels)

#     pim.convert('RGB').save(f'./users/{student_id}/output/Region_{TEST_REGION}_TEST/L1.{lambda_1}_L2.{lambda_2}_EV.{ent_var}_B1.{beta_1}_B2.{beta_2}_P.{probability}_E.{entropy}_C.{cod}_TA.{transformation_agg.upper()}_SA.{superpixel_agg.upper()}/R{TEST_REGION}_labels_{i}.png')
    pim.convert('RGB').save(f'./users/{student_id}/output/R{TEST_REGION}_labels.png')




beta_1_2 = [
    (0.001, 0.005),
    (0.05, 0.05),
    (0.05, 0.1),
    (0.1, 0.05),
    (0.1, 0.1),
    (0.15, 0.1),
    (0.1, 0.2),
    (0.2, 0.2),
    (0.2, 0.3),
#     (0.5, 0.5, 0.5),
#     (1, 1, 1),
#     (5, 5, 5),
#     (10,10,10),
#     (100, 100, 100)
]

lambda_1 = 0.1
lambda_2 = 0.05
lambda_3 = 0.1

# Fixed beta, variable lambda
print("start time: ", time.ctime())
for TEST_REGION in [1]:
    print("TEST_REGION: ", TEST_REGION)
    for betas in beta_1_2:
        beta_1, beta_2 = betas
        for ent_var in [1]:
            print("ent_var: ", ent_var)
            for entropy in [0]:
                probability = 1 - entropy
                recommend = 1
                initial = 1
                
                cod = 1
                transformation_agg = "avg"
                superpixel_agg = "avg"
                student_id = "saugat"


                if os.path.exists(f"./users/{student_id}/output/R{TEST_REGION}_labels.png"):
                    os.remove(f"./users/{student_id}/output/R{TEST_REGION}_labels.png")
                    
                if entropy: # TODO: change this
                    lambda_1 = 0.15
                    lambda_2 = 0.1
                    lambda_3 = 0.1
                else:
                    lambda_1 = 0.1
                    lambda_2 = 0.05
                    lambda_3 = 0.1
                    

                config.ENT_VAR = ent_var

                config.LAMBDA_1 = lambda_1
                config.LAMBDA_1_A2 = lambda_1
                config.LAMBDA_2 = lambda_2
                config.LAMBDA_2_A2 = lambda_2
                config.LAMBDA_3 = lambda_3

                config.BETA_1 = beta_1
                config.BETA_2 = beta_2


                try:
                    with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'r') as file:
                        content = file.read()
                        al_cycle = int(content) 
                except FileNotFoundError:
                    al_cycle = 0

                # to handle the situation where a user does AL for a while, terminated and starts again (all the process has to be completed in 1 go, there can be no break in between)
                if initial:
                    try:
                        with open(f"./users/{student_id}/resume_epoch/R{TEST_REGION}.txt", 'w') as file:
                            file.write(str(0))
                    except FileNotFoundError:
                        pass

                    try:
                        with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'w') as file:
                            file.write(str(0))
                            al_cycle = 0
                    except FileNotFoundError:
                        pass

                    try:
                        with open(f"./users/{student_id}/al_iters/R{TEST_REGION}.txt", 'w') as file:
                            file.write(str(0))
                    except FileNotFoundError:
                        pass

                metrices = {}
                i=0
                if int(recommend):
                    metrices, metrices_unlabeled, selected_superpixels, superpixels_group = recommend_superpixels(TEST_REGION, entropy, probability, cod, transformation_agg, superpixel_agg, student_id, al_cycle)
                    acquire_labels(i, lambda_1, probability, entropy, cod, transformation_agg, superpixel_agg, ent_var, TEST_REGION, selected_superpixels, superpixels_group)


                for i in range(5):
                    print(i+1)
                    # read cycle from txt file
                    try:
                        with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'r') as file:
                            content = file.read()
                            al_cycle = int(content) 
                    except FileNotFoundError:
                        al_cycle = 0

                    print("AL_cycle: ", al_cycle)

                    # read iters from txt file
                    try:
                        with open(f"./users/{student_id}/al_iters/R{TEST_REGION}.txt", 'r') as file:
                            content = file.read()
                            al_iters = int(content) 
                    except FileNotFoundError:
                        al_iters = 0

                    metrices, metrices_unlabeled, selected_superpixels, superpixels_group = train(TEST_REGION, entropy, probability, cod, transformation_agg, superpixel_agg, student_id, al_cycle, al_iters)
                    acquire_labels(i+1, lambda_1, probability, entropy, cod, transformation_agg, superpixel_agg, ent_var, TEST_REGION, selected_superpixels, superpixels_group)


print("end time: ", time.ctime())

# Fixed beta, variable lambda
print("start time: ", time.ctime())
for TEST_REGION in [1]:
    print("TEST_REGION: ", TEST_REGION)
    for betas in beta_1_2:
        beta_1, beta_2 = betas
        for ent_var in [1]:
            print("ent_var: ", ent_var)
            for entropy in [1]:
                probability = 1 - entropy
                recommend = 1
                initial = 1
                
                cod = 1
                transformation_agg = "avg"
                superpixel_agg = "avg"
                student_id = "saugat"


                if os.path.exists(f"./users/{student_id}/output/R{TEST_REGION}_labels.png"):
                    os.remove(f"./users/{student_id}/output/R{TEST_REGION}_labels.png")
                    
                if entropy: # TODO: change this
                    lambda_1 = 0.15
                    lambda_2 = 0.1
                    lambda_3 = 0.1
                else:
                    lambda_1 = 0.1
                    lambda_2 = 0.05
                    lambda_3 = 0.1
                    

                config.ENT_VAR = ent_var

                config.LAMBDA_1 = lambda_1
                config.LAMBDA_1_A2 = lambda_1
                config.LAMBDA_2 = lambda_2
                config.LAMBDA_2_A2 = lambda_2
                config.LAMBDA_3 = lambda_3

                config.BETA_1 = beta_1
                config.BETA_2 = beta_2


                try:
                    with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'r') as file:
                        content = file.read()
                        al_cycle = int(content) 
                except FileNotFoundError:
                    al_cycle = 0

                # to handle the situation where a user does AL for a while, terminated and starts again (all the process has to be completed in 1 go, there can be no break in between)
                if initial:
                    try:
                        with open(f"./users/{student_id}/resume_epoch/R{TEST_REGION}.txt", 'w') as file:
                            file.write(str(0))
                    except FileNotFoundError:
                        pass

                    try:
                        with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'w') as file:
                            file.write(str(0))
                            al_cycle = 0
                    except FileNotFoundError:
                        pass

                    try:
                        with open(f"./users/{student_id}/al_iters/R{TEST_REGION}.txt", 'w') as file:
                            file.write(str(0))
                    except FileNotFoundError:
                        pass

                metrices = {}
                i=0
                if int(recommend):
                    metrices = recommend_superpixels(TEST_REGION, entropy, probability, cod, transformation_agg, superpixel_agg, student_id, al_cycle)
                    acquire_labels(i, lambda_1, probability, entropy, cod, transformation_agg, superpixel_agg, ent_var, TEST_REGION)


                for i in range(5):
                    print(i+1)
                    # read cycle from txt file
                    try:
                        with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'r') as file:
                            content = file.read()
                            al_cycle = int(content) 
                    except FileNotFoundError:
                        al_cycle = 0

                    print("AL_cycle: ", al_cycle)

                    # read iters from txt file
                    try:
                        with open(f"./users/{student_id}/al_iters/R{TEST_REGION}.txt", 'r') as file:
                            content = file.read()
                            al_iters = int(content) 
                    except FileNotFoundError:
                        al_iters = 0

                    metrices = train(TEST_REGION, entropy, probability, cod, transformation_agg, superpixel_agg, student_id, al_cycle, al_iters)
                    acquire_labels(i+1, lambda_1, probability, entropy, cod, transformation_agg, superpixel_agg, ent_var, TEST_REGION)

