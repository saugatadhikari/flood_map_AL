from flask import jsonify, Flask, send_file, request, make_response
from werkzeug.utils import secure_filename
import os
from PIL import Image
import gzip
import numpy as np
from flask_cors import CORS

# from vtkmodules.all import (
#     vtkTIFFReader,
#     vtkImageExtractComponents
# )

import cv2
import gc
# from vtkmodules.util.numpy_support import vtk_to_numpy
import json
import subprocess, sys

# from topologytoolkit import (
#     ttkFTMTree,
#     ttkTopologicalSimplificationByPersistence,
#     ttkScalarFieldSmoother
# )

from al import train, recommend_superpixels


app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "static/files"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(APP_ROOT, app.config["UPLOAD_FOLDER"])


# extractComponent = vtkImageExtractComponents()
# extractComponent.SetInputConnection(pread.GetOutputPort())
# extractComponent.SetComponents(0)
# extractComponent.Update()

# smoother = ttkScalarFieldSmoother()
# smoother.SetInputConnection(0, pread.GetOutputPort())
# smoother.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
# smoother.SetNumberOfIterations(5)
# smoother.Update()

# simplify = ttkTopologicalSimplificationByPersistence()
# simplify.SetInputConnection(0, smoother.GetOutputPort())
# simplify.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
# simplify.SetThresholdIsAbsolute(False)
# simplify.SetPersistenceThreshold(50)
# simplify.Update()

# tree = ttkFTMTree()
# tree.SetInputConnection(0, simplify.GetOutputPort())
# tree.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
# tree.SetTreeType(2)
# tree.SetWithSegmentation(1)
# tree.Update()

@app.route('/stl', methods=['POST'])
def stl():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
#         subprocess.check_output(['./hmm', f.filename, 'a.stl', '-z', '500', '-t', '10000000'])
        payload = make_response(send_file('a.stl'))
        payload.headers.add('Access-Control-Allow-Origin', '*')
        # os.remove('a.stl')
        # os.remove(f.filename)

        return payload
    
@app.route('/superpixel', methods=['GET'])
def superpixel():
    recommend = request.args.get('recommend')

    TEST_REGION = 1 # TODO: this should be received from frontend
    metrices = {}
    if int(recommend):
        metrices = recommend_superpixels(TEST_REGION)
    
    payload = make_response(send_file('R1_superpixels_test.png'))
    payload.headers.add('Access-Control-Allow-Origin', '*')

    # # Create a response with the file and JSON data
    # payload = make_response()

    # # Set the file part of the response
    # file_path = 'R1_superpixels_test.png'
    # payload.headers['Content-Disposition'] = f'attachment; filename={file_path}'
    # payload.headers['Content-Type'] = 'text/plain'
    # payload.headers.add('Access-Control-Allow-Origin', '*')
    # payload.set_data(open(file_path, 'rb').read())

    # # Attach JSON data to the response
    # payload.json = metrices

    return payload


@app.route('/pred')
def pred():
    payload = make_response(send_file('R1_pred_test.png'))
    payload.headers.add('Access-Control-Allow-Origin', '*')
    return payload


@app.route('/confidence')
def confidence():
    # forest_arr = cv2.imread('R1_forest.png')
    # list_data = forest_arr.tolist()
    # json_data = {}
    # for i in range(forest_arr.shape[0]):
    #     for j in range(forest_arr.shape[1]):
    #         index = i * forest_arr.shape[1] + j
    #         json_data[index] = list_data[i][j][1]

    payload = make_response(send_file('R1_forest.png'))
    # payload.json = json_data
    payload.headers.add('Access-Control-Allow-Origin', '*')

    return payload

@app.route('/forest-json', methods=['GET'])
def forest_json():
    TEST_REGION = 1
    file_path = f"./data_al/forest/Region_{TEST_REGION}_forest_json.json"

    forests = {}
    with open(file_path, 'r') as json_file:
        forests = json.load(json_file)

    payload = make_response(jsonify(forests), 200)
    # payload.json = json_data
    payload.headers.add('Access-Control-Allow-Origin', '*')


    # forest_arr = cv2.imread('R1_forest.png')
    # list_data = forest_arr.tolist()
    # json_data = {}
    # for i in range(forest_arr.shape[0]):
    #     for j in range(forest_arr.shape[1]):
    #         index = i * forest_arr.shape[1] + j
    #         json_data[index] = list_data[i][j][1]

    # payload = make_response(jsonify(json_data), 200)
    # # payload.json = json_data
    # payload.headers.add('Access-Control-Allow-Origin', '*')

    return payload

@app.route('/metrics-json', methods=['GET'])
def metrics_json():
    TEST_REGION = 1
    file_path = f"./Region_{TEST_REGION}_Metrics.json"

    metrices = {}
    with open(file_path, 'r') as json_file:
        metrices = json.load(json_file)

    payload = make_response(jsonify(metrices), 200)
    # payload.json = json_data
    payload.headers.add('Access-Control-Allow-Origin', '*')

    return payload


@app.route('/retrain', methods=['POST'])
def retrain():
    task_id = request.args.get('taskId')
    file = request.files.get('image')
    if file:
        print('image is here')
        # file = request.files['image']

        # Process the file as needed, for example, save it to the server
        file.save('./R1_labels.png')

        # TODO: call train function in al.py
        TEST_REGION = 1 # TODO: this should be received from frontend
        train(TEST_REGION)

        payload = make_response(jsonify({'status': 'success', 'taskId': task_id}), 200)
        payload.headers.add('Access-Control-Allow-Origin', '*')

        with open(f"./status_{task_id}.txt", 'w') as file:
            file.write("completed")

        return payload

    payload = make_response(jsonify({'status': 'error', 'taskId': task_id}), 400)
    payload.headers.add('Access-Control-Allow-Origin', '*')

    with open(f"./status_{task_id}.txt", 'w') as file:
        file.write("error")

    return payload


@app.route('/check-status', methods=['GET'])
def check_status():
    task_id = request.args.get('taskId')
    print("task_id: ", task_id)

    # logic to check the status of the task
    with open(f"./status_{task_id}.txt", 'r') as file:
        status = file.read()
    
    print("status: ", status)

    payload = make_response(jsonify({'status': status}), 200)
    payload.headers.add('Access-Control-Allow-Origin', '*')

    return payload



# @app.route('/topology', methods=['POST'])
# def topology():
#     if request.method == 'POST':
#         f = request.files['file']
#         f.save(f.filename)
#         pread = vtkTIFFReader()
#         pread.SetFileName(f.filename)
#         extractComponent = vtkImageExtractComponents()
#         extractComponent.SetInputConnection(pread.GetOutputPort())
#         extractComponent.SetComponents(0)
#         extractComponent.Update()
#         simplify = ttkTopologicalSimplificationByPersistence()
#         simplify.SetInputConnection(0, extractComponent.GetOutputPort())
#         simplify.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
#         simplify.SetThresholdIsAbsolute(False)
#         tree = ttkFTMTree()
#         tree.SetInputConnection(0, simplify.GetOutputPort())
#         tree.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
#         tree.SetTreeType(2)
#         tree.SetWithSegmentation(1)
#         response = {'data': {}, 'segmentation': {}}
#         dmax = 1
#         dmin = 0
#         for i in [0, 0.02, 0.04, 0.08, 0.16, 0.32]:
#             simplify.SetPersistenceThreshold(i)
#             simplify.Update()
#             tree.Update()
#             if i == 0:
#                 dmax = np.max(vtk_to_numpy(simplify.GetOutput().GetPointData().GetArray(0)))
#                 dmin = np.min(vtk_to_numpy(simplify.GetOutput().GetPointData().GetArray(0)))
#                 response['data'][i] = ((vtk_to_numpy(simplify.GetOutput().GetPointData().GetArray(0)) - dmin) / (dmax - dmin)).tolist()
#             else:
#                 response['data'][i] = (vtk_to_numpy(simplify.GetOutput().GetPointData().GetArray(0))).tolist()
#             response['segmentation'][i] = vtk_to_numpy(tree.GetOutput(2).GetPointData().GetArray(2)).tolist()
#         content = gzip.compress(json.dumps(response).encode('utf8'), 9)
#         payload = make_response(content)
#         payload.headers.add('Access-Control-Allow-Origin', '*')
#         payload.headers['Content-length'] = len(content)
#         payload.headers['Content-Encoding'] = 'gzip'
#         del(response)
#         os.remove(f.filename)
#         return payload
    
# @app.route('/test', methods=['POST'])
# def test():
#     response = {"success": "success"}
#     # ranges = [0.02, 0.04, 0.06, 0.08, 0.1]
#     # ranges = [0.02]
#     # for i in ranges:
#     #     simplify.SetPersistenceThreshold(i)
#     #     simplify.Update()
#     #     tree.Update()
#     #     test = vtk_to_numpy(tree.GetOutput(2).GetPointData().GetArray(2))
#     #     response[i] = {'array': test.tolist(), "max": int(np.max(test))}
#     #     del test
#     payload = jsonify(response)
#     payload.headers.add('Access-Control-Allow-Origin', '*')
#     return payload
    
if __name__ == '__main__':
   app.run()