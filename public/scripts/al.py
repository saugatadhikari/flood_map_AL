import torch
from torch.optim import Adam, SGD
import time
import asyncio

import numpy as np

from data.data import *
from data_al.data_al_evanet import get_dataset_al
from data_al.data_remaker_al import remake_data
from eva_net_model import *
from loss import *
from metrics import *

from tqdm import tqdm
# from osgeo import gdal

import matplotlib.pyplot as plt
import cv2

import numpy as np
import torch
import itertools
import time
from collections import defaultdict
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

def get_meta_data(DATASET_PATH):
    
    DATASET = os.listdir(DATASET_PATH)
    DATASET = [file for file in DATASET if  file.endswith(".npy") and re.search("Features", file)]

    META_DATA = dict()

    for file_name in DATASET:
        file = np.load(os.path.join(DATASET_PATH, file_name))
        #print(file.shape)
        file_height, file_width, _ = file.shape
        #print(file_height)
        #print(file_width)

        elev_data = file[:, :, 3]
        file_elev_max = np.max(elev_data)
        file_elev_min = np.min(elev_data)
        # print(file_elev_max)
        # print(file_elev_min)

        if file_elev_max>config.GLOBAL_MAX:
            config.GLOBAL_MAX = file_elev_max
        if file_elev_min<config.GLOBAL_MIN:
            config.GLOBAL_MIN = file_elev_min


        META_DATA[file_name] = {"height": file_height,
                                "width": file_width}
        
    return META_DATA

def run_pred_al(model, data_loader):
    
    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()
    
    for data_dict in tqdm(data_loader):
        
        ## RGB data
        rgb_data = data_dict['rgb_data'].float().to(DEVICE)

        ## Elevation data
        norm_elev_data = data_dict['norm_elev_data'].float().to(DEVICE)

        ## Get filename
        filename = data_dict['filename']

        ## Get model prediction
        pred = model(rgb_data, norm_elev_data)

        # TODO: flip and rotate
        rgb_data_flipx = torch.flip(rgb_data, dims=(-1,))
        rgb_data_flipy = torchvision.transforms.functional.vflip(rgb_data)
        rgb_data_rot90 = torchvision.transforms.functional.rotate(rgb_data, angle=90)
        rgb_data_rot180 = torchvision.transforms.functional.rotate(rgb_data, angle=180)
        rgb_data_rot270 = torchvision.transforms.functional.rotate(rgb_data, angle=270)
        
        norm_elev_data_flipx = torch.flip(norm_elev_data, dims=(-1,))
        norm_elev_data_flipy = torchvision.transforms.functional.vflip(norm_elev_data)
        norm_elev_data_rot90 = torchvision.transforms.functional.rotate(norm_elev_data, angle=90)
        norm_elev_data_rot180 = torchvision.transforms.functional.rotate(norm_elev_data, angle=180)
        norm_elev_data_rot270 = torchvision.transforms.functional.rotate(norm_elev_data, angle=270)

        pred_flipx = model(rgb_data_flipx, norm_elev_data_flipx) # get pred for horizontal flip
        pred_flipy = model(rgb_data_flipy, norm_elev_data_flipy) # get pred for horizontal flip
        pred_rot90 = model(rgb_data_rot90, norm_elev_data_rot90) # get pred for horizontal flip
        pred_rot180 = model(rgb_data_rot180, norm_elev_data_rot180) # get pred for horizontal flip
        pred_rot270 = model(rgb_data_rot270, norm_elev_data_rot270) # get pred for horizontal flip
        
        pred_flipx_inv = torch.flip(pred_flipx, dims=(-1,)) # flip back to original orientation
        pred_flipy_inv = torchvision.transforms.functional.vflip(pred_flipy) # flip back to original orientation
        pred_rot90_inv = torchvision.transforms.functional.rotate(pred_rot90, angle=270)
        pred_rot180_inv = torchvision.transforms.functional.rotate(pred_rot180, angle=180)
        pred_rot270_inv = torchvision.transforms.functional.rotate(pred_rot270, angle=90)

        s1,s2,s3,s4 = pred.shape
        half_array = torch.tensor(np.full((s1, s2, s3, s4), 0.5)).to(DEVICE)
        
        pred_abs = torch.abs(pred - half_array)
        pred_flipx_abs = torch.abs(pred_flipx_inv - half_array)
        pred_flipy_abs = torch.abs(pred_flipy_inv - half_array)
        pred_rot90_abs = torch.abs(pred_rot90_inv - half_array)
        pred_rot180_abs = torch.abs(pred_rot180_inv - half_array)
        pred_rot270_abs = torch.abs(pred_rot270_inv - half_array)

        avg_prob = torch.sum(torch.stack([pred, pred_flipx_inv, pred_flipy_inv, pred_rot90_inv, pred_rot180_inv, pred_rot270_inv]), dim=0) / 6
#         avg_prob = torch.sum(torch.stack([pred_abs, pred_flipx_abs, pred_flipy_abs, pred_rot90_abs, pred_rot180_abs, pred_rot270_abs]), dim=0) / 6
        avg_prob_np = avg_prob.detach().cpu().numpy()

#         avg_prob_np = pred.detach().cpu().numpy()

        # ## Remove pred and GT from GPU and convert to np array
        # pred_labels_np = pred.detach().cpu().numpy() 
        # gt_labels_np = labels.detach().cpu().numpy()
        
        ## Save Image and RGB patch
        for idx in range(rgb_data.shape[0]):
            pred_patches_dict[filename[idx]] = avg_prob_np[idx, :, :, :]
        
    return pred_patches_dict 

def run_pred_final(model, data_loader):
    
    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()
    
    for data_dict in tqdm(data_loader):
        
        ## RGB data
        rgb_data = data_dict['rgb_data'].float().to(DEVICE)

        ## Elevation data
        norm_elev_data = data_dict['norm_elev_data'].float().to(DEVICE)

        ## Get filename
        filename = data_dict['filename']

        ## Get model prediction
        pred = model(rgb_data, norm_elev_data)

        # ## Remove pred and GT from GPU and convert to np array
        pred_labels_np = pred.detach().cpu().numpy()
        
        ## Save Image and RGB patch
        for idx in range(rgb_data.shape[0]):
            pred_patches_dict[filename[idx]] = pred_labels_np[idx, :, :, :]
        
    return pred_patches_dict 


def run_pred_final_avg(model, data_loader):
    
    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()
    
    for data_dict in tqdm(data_loader):
        
        ## RGB data
        rgb_data = data_dict['rgb_data'].float().to(DEVICE)

        ## Elevation data
        norm_elev_data = data_dict['norm_elev_data'].float().to(DEVICE)

        ## Get filename
        filename = data_dict['filename']

        ## Get model prediction
        pred = model(rgb_data, norm_elev_data)

        # TODO: flip and rotate
        rgb_data_flipx = torch.flip(rgb_data, dims=(-1,))
        rgb_data_flipy = torchvision.transforms.functional.vflip(rgb_data)
        rgb_data_rot90 = torchvision.transforms.functional.rotate(rgb_data, angle=90)
        rgb_data_rot180 = torchvision.transforms.functional.rotate(rgb_data, angle=180)
        rgb_data_rot270 = torchvision.transforms.functional.rotate(rgb_data, angle=270)
        
        norm_elev_data_flipx = torch.flip(norm_elev_data, dims=(-1,))
        norm_elev_data_flipy = torchvision.transforms.functional.vflip(norm_elev_data)
        norm_elev_data_rot90 = torchvision.transforms.functional.rotate(norm_elev_data, angle=90)
        norm_elev_data_rot180 = torchvision.transforms.functional.rotate(norm_elev_data, angle=180)
        norm_elev_data_rot270 = torchvision.transforms.functional.rotate(norm_elev_data, angle=270)

        pred_flipx = model(rgb_data_flipx, norm_elev_data_flipx) # get pred for horizontal flip
        pred_flipy = model(rgb_data_flipy, norm_elev_data_flipy) # get pred for horizontal flip
        pred_rot90 = model(rgb_data_rot90, norm_elev_data_rot90) # get pred for horizontal flip
        pred_rot180 = model(rgb_data_rot180, norm_elev_data_rot180) # get pred for horizontal flip
        pred_rot270 = model(rgb_data_rot270, norm_elev_data_rot270) # get pred for horizontal flip
        
        pred_flipx_inv = torch.flip(pred_flipx, dims=(-1,)) # flip back to original orientation
        pred_flipy_inv = torchvision.transforms.functional.vflip(pred_flipy) # flip back to original orientation
        pred_rot90_inv = torchvision.transforms.functional.rotate(pred_rot90, angle=270)
        pred_rot180_inv = torchvision.transforms.functional.rotate(pred_rot180, angle=180)
        pred_rot270_inv = torchvision.transforms.functional.rotate(pred_rot270, angle=90)

#         s1,s2,s3,s4 = pred.shape
#         half_array = torch.tensor(np.full((s1, s2, s3, s4), 0.5)).to(DEVICE)
        
#         pred_abs = torch.abs(pred - half_array)
#         pred_flipx_abs = torch.abs(pred_flipx_inv - half_array)
#         pred_flipy_abs = torch.abs(pred_flipy_inv - half_array)
#         pred_rot90_abs = torch.abs(pred_rot90_inv - half_array)
#         pred_rot180_abs = torch.abs(pred_rot180_inv - half_array)
#         pred_rot270_abs = torch.abs(pred_rot270_inv - half_array)

#         avg_prob = torch.sum(torch.stack([pred, pred_flipx_inv, pred_flipy_inv, pred_rot90_inv, pred_rot180_inv, pred_rot270_inv]), dim=0) / 6
        avg_prob = torch.sum(torch.stack([pred, pred_flipx_inv, pred_flipy_inv, pred_rot90_inv, pred_rot180_inv, pred_rot270_inv]), dim=0) / 6
        avg_prob_np = avg_prob.detach().cpu().numpy()

#         avg_prob_np = pred.detach().cpu().numpy()

        # ## Remove pred and GT from GPU and convert to np array
        # pred_labels_np = pred.detach().cpu().numpy() 
        # gt_labels_np = labels.detach().cpu().numpy()
        
        ## Save Image and RGB patch
        for idx in range(rgb_data.shape[0]):
            pred_patches_dict[filename[idx]] = avg_prob_np[idx, :, :, :]
        
    return pred_patches_dict 


def find_patch_meta(pred_patches_dict):
    y_max = 0
    x_max = 0

    for item in pred_patches_dict:

        temp = int(item.split("_")[3])
        if temp>y_max:
            y_max = temp

        temp = int(item.split("_")[5])
        if temp>x_max:
            x_max = temp


    y_max+=1
    x_max+=1
    
    return y_max, x_max


def stitch_patches_GT_labels(pred_patches_dict, TEST_REGION):
    cropped_data_path = f"./data_al/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    y_max, x_max = find_patch_meta(pred_patches_dict)
    
    for i in range(y_max):
        for j in range(x_max):
            dict_key = f"Region_{TEST_REGION}_y_{i}_x_{j}_features.npy"
            dict_key_label = f"Region_{TEST_REGION}_y_{i}_x_{j}_label.npy"
            #print(dict_key)
        
            pred_patch = pred_patches_dict[dict_key]
            pred_patch = np.transpose(pred_patch, (1, 2, 0))

            label_patch = np.load(os.path.join(cropped_data_path, dict_key_label))

            if j == 0:
                label_x_patches = label_patch
                pred_x_patches = pred_patch
            else:
                label_x_patches = np.concatenate((label_x_patches, label_patch), axis = 1)
                pred_x_patches = np.concatenate((pred_x_patches, pred_patch), axis = 1)

            ## rgb_patches.append(rgb_patch)
            ## pred_patches.append(pred_patch)
    
        if i == 0:
            label_y_patches = label_x_patches
            pred_y_patches = pred_x_patches
        else:
            label_y_patches = np.vstack((label_y_patches, label_x_patches))
            pred_y_patches = np.vstack((pred_y_patches, pred_x_patches))
        

    label_stitched = label_y_patches
#     pred_stitched = np.argmax(pred_y_patches, axis = -1)
    pred_stitched = pred_y_patches.copy()
    
    return label_stitched, pred_stitched

def stitch_patches(pred_patches_dict, TEST_REGION):
    cropped_data_path = f"./data_al/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    y_max, x_max = find_patch_meta(pred_patches_dict)
    
    for i in range(y_max):
        for j in range(x_max):
            dict_key = f"Region_{TEST_REGION}_y_{i}_x_{j}_features.npy"
            #print(dict_key)
        
            pred_patch = pred_patches_dict[dict_key]
            pred_patch = np.transpose(pred_patch, (1, 2, 0))

            rgb_patch = np.load(os.path.join(cropped_data_path, dict_key))[:, :, :3]


            if j == 0:
                rgb_x_patches = rgb_patch
                pred_x_patches = pred_patch
            else:
                rgb_x_patches = np.concatenate((rgb_x_patches, rgb_patch), axis = 1)
                pred_x_patches = np.concatenate((pred_x_patches, pred_patch), axis = 1)

            ## rgb_patches.append(rgb_patch)
            ## pred_patches.append(pred_patch)
    
        if i == 0:
            rgb_y_patches = rgb_x_patches
            pred_y_patches = pred_x_patches
        else:
            rgb_y_patches = np.vstack((rgb_y_patches, rgb_x_patches))
            pred_y_patches = np.vstack((pred_y_patches, pred_x_patches))
        

    rgb_stitched = rgb_y_patches.astype('uint8')
#     pred_stitched = np.argmax(pred_y_patches, axis = -1)
    pred_stitched = pred_y_patches.copy()
    
    return rgb_stitched, pred_stitched

def center_crop(stictched_data, original_height, original_width, image = False):
    # dict_key = f"Region_{TEST_REGION}_Features7Channel.npy"
    
#     if image:
    current_height, current_width = stictched_data.shape[0], stictched_data.shape[1]
#     else:
#         current_height, current_width = stictched_data.shape    
#     print("current_height: ", current_height)
#     print("current_width: ", current_width)
    
    # original_height = META_DATA[dict_key]['height']
    # original_width = META_DATA[dict_key]['width']
#     print("original_height: ", original_height)
#     print("original_width: ", original_width)
    
    height_diff = current_height-original_height
    width_diff = current_width-original_width
    
#     print("height_diff: ", height_diff)
#     print("width_diff: ", width_diff)
    
    
    cropped = stictched_data[height_diff//2:current_height-height_diff//2, width_diff//2: current_width-width_diff//2, :]
    
    return cropped

# TODO: try different schemes like min
def get_superpixel_scores(superpixels_group, logits):
    superpixel_scores = {}
    for sid, pixels in superpixels_group.items():
        total_score = 0
        total_pixels = len(pixels)
        for (row, col) in pixels:
            prob_score = logits[row][col]
            total_score += prob_score
        avg_score = total_score / total_pixels # average of all pixel's score
        superpixel_scores[sid] = avg_score
    
    return superpixel_scores

def get_superpixel_scores_min(superpixels_group, logits):
    superpixel_scores = {}
    for sid, pixels in superpixels_group.items():
        total_score = 0
        total_pixels = len(pixels)
        min_score = 100
        for (row, col) in pixels:
            prob_score = logits[row][col]
            if prob_score < min_score:
                min_score = prob_score
                
        superpixel_scores[sid] = min_score
    
    return superpixel_scores


def loss_self_consistency(logits: list, labels):
    if not logits:
        return 0

    total_sum = 0
    counter = 0
#     for comb in itertools.combinations(logits, 2):
#         l1, l2 = comb
#         diff = torch.subtract(l1, l2)
# #         norm = diff.norm(dim=1, p=2) # calculate L2-norm
#         norm = torch.norm(diff)
#         total_sum += norm
#         counter += 1

     ## Generate Pred Masks
#     ones = torch.ones_like(labels)

#     known_mask = torch.where(labels != 0, ones, 0)
    
    original_logit = logits[0]
    for transformed_logit in logits[1:]:
        l1, l2 = original_logit, transformed_logit
        diff = torch.subtract(l1, l2)
#         diff = known_mask * diff
        norm = torch.norm(diff)
        total_sum += norm
        counter += 1

    _, W, H, C = logits[0].shape
    loss = (total_sum)/(W * H * C * counter)
#     loss = (total_sum)/(W * H * C)
    return loss
#     loss = np.asarray([loss])
#     return torch.tensor(loss)

def label_acquisition(selected_superpixels, elev_data, gt_labels, current_labels, superpixels_group):
    """
        Acquire labels of current_pixel as well as its BFS neighbors 

        Parameters:
            selected_pixels: 2D array where selected pixel = 1, others are 0
            elev_data: DEM of all pixels
            gt_labels: GT labels from our annotation
            current_labels: labels acquired till now; this needs to be updated and returned back                                    
    """

    updated_labels = current_labels.copy() # TODO: initially current labels should all be 0 (unknown)
    
    for sid in selected_superpixels:
        for (row, col) in superpixels_group[sid]:
            updated_labels[row][col] = gt_labels[row][col]
    
    return updated_labels

def select_superpixels(total_superpixels, superpixel_scores, labeled_superpixels, rejection_superpixels):
    max_items = min(config.NUM_RECOMMEND, total_superpixels)
    select_count = 0
    selected_superpixels = []
    for i, (sid, prob_score) in enumerate(superpixel_scores.items()):
        is_labeled = labeled_superpixels.get(sid, False)
        is_reject = rejection_superpixels.get(sid, False)
        if not is_labeled and not is_reject:
            if select_count < max_items:
                selected_superpixels.append(sid)
                labeled_superpixels[sid] = True
                select_count += 1
    
    return labeled_superpixels, selected_superpixels, max_items

class EarlyStopping:
	"""
	Early stopping to stop the training when the loss does not improve after
	certain epochs.
	"""

	def __init__(self, patience=5, min_delta=0):
		"""
		:param patience: how many epochs to wait before stopping when loss is
			   not improving
		:param min_delta: minimum difference between new loss and old loss for
			   new loss to be considered as an improvement
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False

	def __call__(self, val_loss):
		if self.best_loss == None:
			self.best_loss = val_loss
		elif self.best_loss - val_loss > self.min_delta:
			self.best_loss = val_loss
			# reset counter if validation loss improves
			self.counter = 0
		elif self.best_loss - val_loss < self.min_delta:
			self.counter += 1
			# logging.info(f"Early stopping counter {self.counter} of {self.patience}")
			if self.counter >= self.patience:
				# logging.info('Early stop!')
				self.early_stop = True

def convert_to_rgb(input_array):
    # Ensure the input array is in the range [0, 1]
    input_array = np.clip(input_array, 0, 1)

    # Create a colormap from blue to red
    cmap = plt.get_cmap('RdYlGn')

    # Apply the colormap to the input array
    rgb_image = cmap(input_array)

    # Remove the alpha channel if it exists
    if rgb_image.shape[-1] == 4:
        rgb_image = rgb_image[:, :, :3]

    return rgb_image

def recommend_superpixels(TEST_REGION):
    return # TODO: remove after test

    start = time.time()

    superpixels = np.load(f"./data_al/superpixels/Region_{TEST_REGION}/Region_{TEST_REGION}_superpixels.npy")
    rejection = np.load(f"./data_al/rejection/Region_{TEST_REGION}_rejection.npy")

    superpixels_group = defaultdict(list)
    rejection_superpixels = {}

    # Iterate through the NumPy array to group pixels
    height = superpixels.shape[0]
    width = superpixels.shape[1]
    for i in range(height):
        for j in range(width):
            pixel_value = superpixels[i][j]
            superpixels_group[pixel_value].append((i, j))
            
    for sid, pixels in superpixels_group.items():
        reject_count = 0
        total_pixels = len(pixels)
        for (row, col) in pixels:
            is_reject = rejection[row][col]
            if is_reject:
                reject_count += 1

        reject_fraction = reject_count / total_pixels
        if reject_fraction >= 1.0:
            rejection_superpixels[sid] = True
        else:
            rejection_superpixels[sid] = False

    labeled_superpixels = {}

    # read resume epoch from text file if exists
    try:
        with open("./resume_epoch.txt", 'r') as file:
            content = file.read()
            resume_epoch = int(content) 
    except FileNotFoundError:
        resume_epoch = 0

    print("Starting from epoch: ", resume_epoch)

    test_filename = f"Region_{TEST_REGION}_Features7Channel.npy"

    VAL_FREQUENCY = 1
    SAVE_FREQUENCY = 1

    elev_data = np.load(f"./data_al/repo/Features_7_Channels/Region_{TEST_REGION}_Features7Channel.npy")[:,:,3]
    gt_labels = np.load(f"./data_al/repo/groundTruths/Region_{TEST_REGION}_GT_Labels.npy")

    total_superpixels = len(superpixels_group.items())

    ######### Pixel Selection using Active Learning #######################
    model = EvaNet(config.BATCH_SIZE, config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)
    optimizer = SGD(model.parameters(), lr = 1e-7)
    criterion = ElevationLoss()
    elev_eval = Evaluator()

    model_path = f"./saved_models_evanet/Region_{TEST_REGION}_TEST/saved_model_AL_{resume_epoch}.ckpt"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        print(f"Resuming from epoch {resume_epoch}")
    # else:
    #     print("No model found!!!")
    #     exit(0)
    
    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()
    
    ## Run prediciton
    cropped_data_path = f"./data_al/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    test_dataset = get_dataset(cropped_data_path)
    test_loader = DataLoader(test_dataset, batch_size = config.BATCH_SIZE)
    pred_patches_dict = run_pred_al(model, test_loader)

    ## Stitch pred patches back together
    rgb_stitched, pred_stitched = stitch_patches(pred_patches_dict, TEST_REGION)

    ## Remove border padding
    pred_unpadded = center_crop(pred_stitched, height, width, image = False)
    
    s1,s2,s3 = pred_unpadded.shape
    half_array = np.full((s1, s2, s3), 0.5) 
    pred_abs = np.absolute(pred_unpadded - half_array)
    
    pred_unpadded = pred_abs[:,:,0]
    superpixel_scores = get_superpixel_scores(superpixels_group, pred_unpadded)
    # superpixel_scores = get_superpixel_scores_min(superpixels_group, pred_unpadded)
    
    # sort by prob score in ascending order; most uncertain superpixel first (whichever is close to 0.5)
#     superpixel_scores = dict(sorted(superpixel_scores.items(), key=lambda item: abs(item[1] - 0.5)))
    superpixel_scores = dict(sorted(superpixel_scores.items(), key=lambda item: item[1]))

    labeled_superpixels, selected_superpixels, max_items = select_superpixels(total_superpixels, superpixel_scores, labeled_superpixels, rejection_superpixels)

    # get the superpixels to be recommended in this iteration and save as png
    interval = (139 - 25) / (max_items - 1)
    slot_values = np.linspace(0.1, 0.99, max_items)

    recommended_superpixels = np.zeros((height, width))
    for i, sid in enumerate(selected_superpixels):
        pixels = superpixels_group[sid]
        # slot_val = int(139 - i * interval)
        recommended_superpixels[tuple(zip(*pixels))] = slot_values[i]
    
    mask = np.where(recommended_superpixels > 0, 1, 0)
    mask = np.expand_dims(mask, axis=-1)
    recommended_superpixels = recommended_superpixels.astype('float32')
    result_array = convert_to_rgb(recommended_superpixels)
    result_array = result_array * mask
    plt.imsave('./R1_superpixels_test.png', result_array)

    # save current prediction as png
    flood_labels = np.where(pred_unpadded > 0.5, 1, 0)
    dry_labels = np.where(pred_unpadded <= 0.5, 1, 0)
    flood_labels = np.expand_dims(flood_labels, axis=-1)
    dry_labels = np.expand_dims(dry_labels, axis=-1)
    flood_labels = flood_labels*np.array([ [ [255, 0, 0] ] ])
    dry_labels = dry_labels*np.array([ [ [0, 0, 255] ] ])
    pred_labels = (flood_labels + dry_labels).astype('uint8')
    pim = Image.fromarray(pred_labels)
    pim.convert('RGB').save("./R1_pred_test.png")

    return


def ann_to_labels(png_image):
    ann = cv2.imread(png_image)
    ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

    flood = ann[:, :, 0] == 255
    dry = ann[:, :, 2] == 255

    flood_arr = np.where(flood, 1, 0)
    dry_arr = np.where(dry, -1, 0)

    final_arr = flood_arr + dry_arr
    
    return final_arr


def train(TEST_REGION):
    print("Retraining the Model with new labels")
    time.sleep(30)
    return # TODO: remove after test


    model = EvaNet(config.BATCH_SIZE, config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)
    optimizer = SGD(model.parameters(), lr = 1e-7)
    criterion = ElevationLoss()
    elev_eval = Evaluator()

    # read resume epoch from text file if exists
    try:
        with open("./resume_epoch.txt", 'r') as file:
            content = file.read()
            resume_epoch = int(content) 
    except FileNotFoundError:
        resume_epoch = 0

    model_path = f"./saved_models_evanet/Region_{TEST_REGION}_TEST/saved_model_AL_{resume_epoch}.ckpt"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        print(f"Resuming from epoch {resume_epoch}")
    # else:
    #     print("Model not found!!!")
    #     exit(0)

    updated_labels = ann_to_labels("./R1_labels.png")

    # need to remake labels after getting updated labels
    remake_data(updated_labels, TEST_REGION)
    
    cropped_data_path_al = f"./data_al/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    elev_val_test_dataset_al = get_dataset_al(cropped_data_path_al)
    elev_val_test_seq = np.arange(0, len(elev_val_test_dataset_al), dtype=int)
    elev_val_test_dataset_al = torch.utils.data.Subset(elev_val_test_dataset_al, elev_val_test_seq)
    al_loader = DataLoader(elev_val_test_dataset_al, batch_size = config.BATCH_SIZE)
    
    # Retrain
    ################################## Training Loop#####################################    
    al_loss_dict = dict()
    val_loss_dict = dict()
    min_val_loss = 1e10   
    
    early_stop = EarlyStopping(patience=7) # TODO: this should be a parameter
    VAL_FREQUENCY = 1

    for epoch in range(resume_epoch, resume_epoch + config.EPOCHS):
        print(f"EPOCH: {epoch}/{resume_epoch+config.EPOCHS} \r")
        ## Model gets set to training mode
        model.train()
        al_loss = 0 

        for data_dict in tqdm(al_loader):

            ## Retrieve data from data dict and send to deivce

            ## RGB data
            rgb_data = data_dict['rgb_data'].float().to(DEVICE)
            rgb_data.requires_grad = True

            ## Elevation data
            elev_data = data_dict['elev_data'].float().to(DEVICE)
            elev_data.requires_grad = False

            norm_elev_data = data_dict['norm_elev_data'].float().to(DEVICE)
            norm_elev_data.requires_grad = False

            """
            ## Data labels
            Elev Loss function label format: Flood = 1, Unknown = 0, Dry = -1 
            """
            labels = data_dict['labels'].float().to(DEVICE)
            labels.requires_grad = False  

            ## Get model prediction
            pred = model(rgb_data, norm_elev_data)
            
            rgb_data_flipx = torch.flip(rgb_data, dims=(-1,))
            rgb_data_flipy = torchvision.transforms.functional.vflip(rgb_data)
            rgb_data_rot90 = torchvision.transforms.functional.rotate(rgb_data, angle=90)
            rgb_data_rot180 = torchvision.transforms.functional.rotate(rgb_data, angle=180)
            rgb_data_rot270 = torchvision.transforms.functional.rotate(rgb_data, angle=270)

            norm_elev_data_flipx = torch.flip(norm_elev_data, dims=(-1,))
            norm_elev_data_flipy = torchvision.transforms.functional.vflip(norm_elev_data)
            norm_elev_data_rot90 = torchvision.transforms.functional.rotate(norm_elev_data, angle=90)
            norm_elev_data_rot180 = torchvision.transforms.functional.rotate(norm_elev_data, angle=180)
            norm_elev_data_rot270 = torchvision.transforms.functional.rotate(norm_elev_data, angle=270)

            pred_flipx = model(rgb_data_flipx, norm_elev_data_flipx) # get pred for horizontal flip
            pred_flipy = model(rgb_data_flipy, norm_elev_data_flipy) # get pred for horizontal flip
            pred_rot90 = model(rgb_data_rot90, norm_elev_data_rot90) # get pred for horizontal flip
            pred_rot180 = model(rgb_data_rot180, norm_elev_data_rot180) # get pred for horizontal flip
            pred_rot270 = model(rgb_data_rot270, norm_elev_data_rot270) # get pred for horizontal flip

            pred_flipx_inv = torch.flip(pred_flipx, dims=(-1,)) # flip back to original orientation
            pred_flipy_inv = torchvision.transforms.functional.vflip(pred_flipy) # flip back to original orientation
            pred_rot90_inv = torchvision.transforms.functional.rotate(pred_rot90, angle=270)
            pred_rot180_inv = torchvision.transforms.functional.rotate(pred_rot180, angle=180)
            pred_rot270_inv = torchvision.transforms.functional.rotate(pred_rot270, angle=90)
            
            # elev_data_flipx = torch.flip(elev_data, dims=(-1,))
            # elev_data_flipy = torchvision.transforms.functional.vflip(elev_data)
            # elev_data_rot90 = torchvision.transforms.functional.rotate(elev_data, angle=90)
            # elev_data_rot180 = torchvision.transforms.functional.rotate(elev_data, angle=180)
            # elev_data_rot270 = torchvision.transforms.functional.rotate(elev_data, angle=270)

            ## Backprop Loss
            loss1 = criterion.forward(pred, elev_data, labels)
            loss2 = criterion.forward(pred_flipx_inv, elev_data, labels)
            loss3 = criterion.forward(pred_flipy_inv, elev_data, labels)
            loss4 = criterion.forward(pred_rot90_inv, elev_data, labels)
            loss5 = criterion.forward(pred_rot180_inv, elev_data, labels)
            loss6 = criterion.forward(pred_rot270_inv, elev_data, labels)
#                 loss = criterion.forward(avg_pred, elev_data, labels)
            ##print("Loss: ", loss.item())

            # flip and rotate; add all 6
            all_logits = [pred, pred_flipx_inv, pred_flipy_inv, pred_rot90_inv, pred_rot180_inv, pred_rot270_inv]

            # TODO: calculate L(self-consistency) based on eqn 5(c) EquAL
            loss_sc = loss_self_consistency(all_logits, labels)
#             total_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss_sc
#             total_loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6 + loss_sc
            total_loss = loss1 + loss_sc

            # backpropagate the total loss
#             loss.backward()
            total_loss.backward()
            optimizer.step()

            ## Record loss for batch
#             al_loss += loss.item()
            al_loss += total_loss.item()

        al_loss /= len(al_loader)
        al_loss_dict[epoch+1] = al_loss
        print(f"Epoch: {epoch+1} AL Loss: {al_loss}" )
        
        #=====================================================================================
    
        ## Do model validation for epochs that match VAL_FREQUENCY
        if (epoch+1)%VAL_FREQUENCY == 0:    

            ## Model gets set to evaluation mode
            model.eval()
            val_loss = 0 

            print("Starting Evaluation")

            for data_dict in tqdm(al_loader):

                ## RGB data
                rgb_data = data_dict['rgb_data'].float().to(DEVICE)
                ## Elevation data
                elev_data = data_dict['elev_data'].float().to(DEVICE)
                norm_elev_data = data_dict['norm_elev_data'].float().to(DEVICE)
                ## Data labels
                labels = data_dict['labels'].float().to(DEVICE)

                ## Get model prediction
                pred = model(rgb_data, norm_elev_data)
                
                # TODO: flip and rotate
                rgb_data_flipx = torch.flip(rgb_data, dims=(-1,))
                rgb_data_flipy = torchvision.transforms.functional.vflip(rgb_data)
                rgb_data_rot90 = torchvision.transforms.functional.rotate(rgb_data, angle=90)
                rgb_data_rot180 = torchvision.transforms.functional.rotate(rgb_data, angle=180)
                rgb_data_rot270 = torchvision.transforms.functional.rotate(rgb_data, angle=270)

                norm_elev_data_flipx = torch.flip(norm_elev_data, dims=(-1,))
                norm_elev_data_flipy = torchvision.transforms.functional.vflip(norm_elev_data)
                norm_elev_data_rot90 = torchvision.transforms.functional.rotate(norm_elev_data, angle=90)
                norm_elev_data_rot180 = torchvision.transforms.functional.rotate(norm_elev_data, angle=180)
                norm_elev_data_rot270 = torchvision.transforms.functional.rotate(norm_elev_data, angle=270)

                pred_flipx = model(rgb_data_flipx, norm_elev_data_flipx) # get pred for horizontal flip
                pred_flipy = model(rgb_data_flipy, norm_elev_data_flipy) # get pred for horizontal flip
                pred_rot90 = model(rgb_data_rot90, norm_elev_data_rot90) # get pred for horizontal flip
                pred_rot180 = model(rgb_data_rot180, norm_elev_data_rot180) # get pred for horizontal flip
                pred_rot270 = model(rgb_data_rot270, norm_elev_data_rot270) # get pred for horizontal flip

                pred_flipx_inv = torch.flip(pred_flipx, dims=(-1,)) # flip back to original orientation
                pred_flipy_inv = torchvision.transforms.functional.vflip(pred_flipy) # flip back to original orientation
                pred_rot90_inv = torchvision.transforms.functional.rotate(pred_rot90, angle=270)
                pred_rot180_inv = torchvision.transforms.functional.rotate(pred_rot180, angle=180)
                pred_rot270_inv = torchvision.transforms.functional.rotate(pred_rot270, angle=90)

#                 elev_data_flipx = torch.flip(elev_data, dims=(-1,))
#                 elev_data_flipy = torchvision.transforms.functional.vflip(elev_data)
#                 elev_data_rot90 = torchvision.transforms.functional.rotate(elev_data, angle=90)
#                 elev_data_rot180 = torchvision.transforms.functional.rotate(elev_data, angle=180)
#                 elev_data_rot270 = torchvision.transforms.functional.rotate(elev_data, angle=270)

                ## Backprop Loss
                loss1 = criterion.forward(pred, elev_data, labels)
                loss2 = criterion.forward(pred_flipx_inv, elev_data, labels)
                loss3 = criterion.forward(pred_flipy_inv, elev_data, labels)
                loss4 = criterion.forward(pred_rot90_inv, elev_data, labels)
                loss5 = criterion.forward(pred_rot180_inv, elev_data, labels)
                loss6 = criterion.forward(pred_rot270_inv, elev_data, labels)
    #                 loss = criterion.forward(avg_pred, elev_data, labels)
                ##print("Loss: ", loss.item())

                # flip and rotate; add all 6
                all_logits = [pred, pred_flipx_inv, pred_flipy_inv, pred_rot90_inv, pred_rot180_inv, pred_rot270_inv]

                # TODO: calculate L(self-consistency) based on eqn 5(c) EquAL
                loss_sc = loss_self_consistency(all_logits, labels)
#                 total_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss_sc
#                 total_loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6 + loss_sc
                total_loss = loss1 + loss_sc

                ## Record loss for batch
#                 val_loss += loss.item()
                val_loss += total_loss.item()

                ## Remove pred and GT from GPU and convert to np array
#                 pred_labels_np = avg_pred.detach().cpu().numpy() 
#                 gt_labels_np = labels.detach().cpu().numpy()

            val_loss /= len(al_loader)
            val_loss_dict[epoch+1] = val_loss
            print(f"Epoch: {epoch+1} Validation Loss: {val_loss}" )
            
            
#             early_stop = EarlyStopping(patience=5)
            
            early_stop(val_loss)
            if early_stop.early_stop:
                print('Early stop!')
                break
                
            if val_loss < min_val_loss:
                resume_epoch = epoch + 1
                min_val_loss = val_loss
                print("Saving Model")
                torch.save({'epoch': epoch + 1,  # when resuming, we will start at the next epoch
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, 
                            f"./saved_models_evanet/Region_{TEST_REGION}_TEST/saved_model_AL_{epoch+1}.ckpt")
    
    with open("./resume_epoch.txt", 'w') as file:
        file.write(str(resume_epoch))
    
    # call AL pipeline once the model is retrained
    recommend_superpixels(TEST_REGION)
    
    return


if __name__ == "__main__":
    TEST_REGION = "1"

    







