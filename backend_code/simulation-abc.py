from al import recommend_superpixels, train, ann_to_labels
import cv2
import numpy as np
from PIL import Image
import config
import os

def elevation_guided_BFS(new_labels, dem):
    selected_pixels = np.where(new_labels != 0)
    
    bfs_queue = []
    height, width = new_labels.shape
    bfs_visited = np.zeros((height, width)).astype('int')

    for j in selected_pixels[0]:
        for i in selected_pixels[1]:

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

    return new_labels


def acquire_labels(i, lambda_1, probability, entropy, cod, transformation_agg, superpixel_agg, TEST_REGION):
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
#     new_labels = elevation_guided_BFS(new_labels, dem)

    flood_labels = np.where(new_labels == 1, 1, 0)
    dry_labels = np.where(new_labels == -1, 1, 0)
    flood_labels = np.expand_dims(flood_labels, axis=-1)
    dry_labels = np.expand_dims(dry_labels, axis=-1)
    flood_labels = flood_labels*np.array([ [ [255, 0, 0] ] ])
    dry_labels = dry_labels*np.array([ [ [0, 0, 255] ] ])
    final_labels = (flood_labels + dry_labels).astype('uint8')
    pim = Image.fromarray(final_labels)

    pim.convert('RGB').save(f'./users/{student_id}/output/lambda_search/L1.{lambda_1}_L2.{lambda_2}_B1.{beta_1}_B2.{beta_2}_P.{probability}_E.{entropy}_C.{cod}_TA.{transformation_agg}_SA.{superpixel_agg}/R{TEST_REGION}_labels_{i}.png')
    pim.convert('RGB').save(f'./users/{student_id}/output/R{TEST_REGION}_labels.png')

LAMBDA_1_UNCERTAINTY_GRID = [0.1, 0.2]
LAMBDA_2_UNCERTAINTY_GRID = [0.1, 0.2, 0.3]

for TEST_REGION in [1, 2, 3]:
    print("TEST_REGION: ", TEST_REGION)
    for lambda_1 in LAMBDA_1_UNCERTAINTY_GRID:
        for lambda_2 in LAMBDA_2_UNCERTAINTY_GRID:
            recommend = 1
            initial = 1
            probability = 1
            entropy = 0
            cod = 1
            transformation_agg = "avg"
            superpixel_agg = "avg"
            student_id = "saugat"


            if os.path.exists(f"./users/{student_id}/output/R{TEST_REGION}_labels.png"):
                os.remove(f"./users/{student_id}/output/R{TEST_REGION}_labels.png")

            config.LAMBDA_1 = lambda_1
#             lambda_2 = 0
            config.LAMBDA_2 = lambda_2
            beta_1 = 0.1
            beta_2 = 0.1

            config.BETA_1 = beta_1
            config.BETA_2 = beta_2



            if not os.path.exists(f"./users/{student_id}/output/lambda_search/L1.{lambda_1}_L2.{lambda_2}_B1.{beta_1}_B2.{beta_2}_P.{probability}_E.{entropy}_C.{cod}_TA.{transformation_agg}_SA.{superpixel_agg}/"):
                os.mkdir(f"./users/{student_id}/output/lambda_search/L1.{lambda_1}_L2.{lambda_2}_B1.{beta_1}_B2.{beta_2}_P.{probability}_E.{entropy}_C.{cod}_TA.{transformation_agg}_SA.{superpixel_agg}/")

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
                acquire_labels(i, lambda_1, probability, entropy, cod, transformation_agg, superpixel_agg, TEST_REGION)

        #         file_path = f"./users/{student_id}/output/lambda_search/L1.{lambda_1}_P.{probability}_E.{entropy}_C.{cod}_TA.{transformation_agg}_SA.{superpixel_agg}/Region_{TEST_REGION}_Metrics_{i}.txt"
        #         with open(file_path, "w") as fp:
        #             fp.write(metrices)


            for i in range(3):
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
                acquire_labels(i+1, lambda_1, probability, entropy, cod, transformation_agg, superpixel_agg, TEST_REGION)

        #         file_path = f"./users/{student_id}/output/lambda_search/L1.{lambda_1}_P.{probability}_E.{entropy}_C.{cod}_TA.{transformation_agg}_SA.{superpixel_agg}/Region_{TEST_REGION}_Metrics_{i+1}.txt"
        #         with open(file_path, "w") as fp:
        #             fp.write(metrices)
