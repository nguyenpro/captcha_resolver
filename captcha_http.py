import cv2
import argparse
import numpy as np
import time
from datetime import timedelta
import os
import os.path
import ctypes
import glob
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from os import curdir, sep
from datetime import datetime
import cgi
import re,base64
import io
from PIL import Image
#server
HOST_NAME = ''
PORT_NUMBER = 9000

# detect product
cfgfile = ''
whtfile = ''

SP_CHAR = '_'
#detect logo

classes = []

conf_threshold = 0.5  #Confidence threshold
nms_threshold = 0.4   #Non-maximum suppression threshold

COLORS = [0,255,255]




user32 = ctypes.windll.user32

def InitNet():
    global cfgfile, whtfile, clsfile, classes
    global net
    #detect product
    cfgfile = 'model/yolov3-tiny(captcha).cfg'
    whtfile = 'model/yolov3-tiny(captcha)_7000.weights'

    net = cv2.dnn.readNet(whtfile, cfgfile)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    global classes
    global COLORS
    label = str(class_id)
    color = COLORS
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), [0,255,255], 2)

    l = x
    t = y
    w = img.shape[1]
    h = img.shape[0]
    if x < w / 2:
        l =  x_plus_w
    if y < h / 2:
        t = y_plus_h


    cv2.putText(img, label, (l, t), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def detect_patch(image):

    global net
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    global classes, COLORS, conf_threshold, nms_threshold

    '''feed image to Yolo net'''
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    dets = []

    for index in indices:
        det = []
        i = index[0]
        box = boxes[i]
        x = np.maximum(int(box[0]), 0)
        y = np.maximum(int(box[1]), 0)
        w = int(box[2])
        h = int(box[3])

        if x + w >= Width:
            w = Width - x
        if y + h >= Height:
            h = Height - y

        #draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

        det.append(class_ids[i])
        det.append(x)
        det.append(y)
        det.append(w)
        det.append(h)
        det.append(confidences[i])

        dets.append(det)

    max_score = 0.0
    max_index = -1
    cnt = 0
    for det in dets:
        if det[5] > max_score:
            max_score = det[5]
            max_index = cnt
        cnt += 1


    return dets, max_index


def detect_logo(image):
    global net1

    Width = image.shape[1]
    Height = image.shape[0]

    if Width <= 0 or Height <= 0:
        return []

    scale = 0.00392
    global classes, COLORS, conf_threshold, nms_threshold

    '''feed image to Yolo net'''
    blob = cv2.dnn.blobFromImage(image, scale, (606, 606), (0, 0, 0), True, crop=False)
    net1.setInput(blob)
    outs = net1.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    dets = []
    for i in indices:
        det = []
        i = i[0]
        box = boxes[i]
        x = np.maximum(int(box[0]), 0)
        y = np.maximum(int(box[1]), 0)
        w = int(box[2])
        h = int(box[3])

        if x + w >= Width:
            w = Width - x
        if y + h >= Height:
            h = Height - y
        # draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        det.append(class_ids[i])
        det.append(x)
        det.append(y)
        det.append(w)
        det.append(h)
        det.append(confidences[i])
        dets.append(det)
    return dets


def image_detect(image):
    Width = image.shape[1]
    Height = image.shape[0]

    # make patch images
    rd = int(Height / 2)
    cd = int(Width / 5)

    x = 0
    y = 0
    name = ''
    acc = [0] * 10
    total_dets = []

    start = time.time()
    for i in range(2):
        for j in range(5):
            patch = image[y: y + rd, x:x + cd]
            (dets, max_index) = detect_patch(patch)

            if max_index < 0: # detect failed
                # add failed symbol
                name = '{}{}'.format(name, SP_CHAR)
                # add empty det
                det = []
                det.append(-1)
                det.append(x)
                det.append(y)
                det.append(cd)
                det.append(rd)
                det.append(0.0)
                total_dets.append(det)
                x += cd
                continue

            #draw result
            det = dets[max_index]
            det[1] += x
            det[2] += y
            #draw_prediction(image, det[0], det[5], round(det[1]), round(det[2]), round(det[1] + det[3]), round(det[2] + det[4]))

            total_dets.append(det)
            acc[det[0]] += 1

            # make digit sequence
            name = '{}{}'.format(name,det[0])
            x += cd

        y += rd
        x = 0
    # modify detection result
    #total_dets = modify_dets(total_dets, acc, name)
    name = reconfig_name(total_dets)
    #draw_all_patch(image, total_dets)
    #dt = float(time.time() - start)
    #str_time = "%.3f s"%dt
    #cv2.putText(image, str_time, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255,0,0], 2)
    #cv2.putText(image, name, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0], 1)

    return image, name
def reconfig_name(dets):
    name = ""
    for det in dets:
        id = det[0]
        if id == -1:
            id = '?'
        name = '{}{}'.format(name, id)
    return name
'''
function: draw_all_patch
description: draw all detectioon result on image
'''
def draw_all_patch(image, dets):
    for det in dets:
        draw_prediction(image, det[0], det[5], round(det[1]), round(det[2]), round(det[1] + det[3]),
                        round(det[2] + det[4]))

'''
function: modify_dets
arg0: dets: list of det in each patch
    if detection failed in this patch, classid = -1
arg1: acc: table of accumulate: length = 10
arg2: names: string of detection result.
    SP_CHAR exists in the position of detection failed
'''
def modify_dets(dets, acc, names):

    #find position of detection failed
    failed_index = []
    cnt = 0
    for elem in names:
        if elem != SP_CHAR:
            cnt += 1
            continue
        failed_index.append(cnt)
        cnt+=1
        continue

    #find number of detection failed
    failed_num = []
    dupli_num = []
    for i in range(10):
        if acc[i] == 0:
            failed_num.append(i)
        elif acc[i] > 1:
            for d in range(acc[i] - 1):
                dupli_num.append(i)
    #matching start
    #assert failed_index.__len__() == failed_num.__len__(), "detection string error"
    len = failed_num.__len__()
    matching = [0] * len
    if  len == 0: #detection success
        return dets

    if failed_index.__len__()  > 0: #detection failed
        # generate random matching relation
        len1 = failed_index.__len__()
        for i in range(len1):
            thres = 1.0 / len
            rd = np.maximum(random.random() - 0.001, 0)
            index = int(rd / thres)
            matching[i] = failed_num[index]
            del failed_num[index]
            #if i >= index:
            #   i -= 1
            len = failed_num.__len__()
        # modify dets using matching
        for i in range(len1):
            index = failed_index[i]
            # modify classid
            dets[index][0] = matching[i]
        if len == 0:
            return dets

    #reamin misdetection in some position
    assert (len == dupli_num.__len__()), "error occured in modify function"
    for i in range(len):
        indexes = []
        count = 0
        id_dst = dets[i][0] # dest classid
        for det in dets:
            id = det[0]
            if id != id_dst:
                count += 1
                continue
            indexes.append(count)
            count += 1
        # find all mismatch position , find position of lowest probability
        low_prob = 1.0
        low_index = -1
        count = 0
        for index in indexes:
            prob = dets[index][5]
            if prob > low_prob:
                count += 1
                continue
            low_prob = prob
            low_index = index
            count += 1

        # exchange classid of dets in low_prob position
        thres = 1.0 / (len - i)
        val = np.maximum(random.random()-0.001, 0.0)
        index = int(val / thres)
        # exchange classid
        dets[low_index][0] = failed_num[index]
        # remove it from candidates list
        del failed_num[index]
        if i >= index:
            i -= 1
        len = failed_num.__len__()
    return dets

def images_detect(img_path):

    #win_name = 'detect_result window... press "n" for next, "q" for exit '
    files = glob.glob(os.path.join(img_path, '*.jpg'))

    len = files.__len__()
    cnt_correct = 0
    cnt_total = 0
    index = 0
    win_name = "capture window"
    while index < len:
        file = files[index]
        name = os.path.basename(file).split('.')[0]
        ext = os.path.splitext(file)[1]
        if not ext in ['.JPG', '.jpg', '.bmp']:
            continue
        #ipath = os.path.join(img_path, file)
        img = cv2.imread(file)
        (img, detected_name)=image_detect(img)
        if name == detected_name:
            cnt_correct += 1
            print('correct: ', detected_name)
        else:
            print('error: ', detected_name)

        cnt_total += 1
        accuracy = float(cnt_correct) / cnt_total
        print('accuracy = ', accuracy)
        img = cv2.resize(img, None, fx = 3.0, fy = 3.0)
        cv2.imshow(win_name, img)
        index += 1
        cv2.waitKey(100)
        '''key = cv2.waitKey(0) & 0xFF
        if key == ord('d'):
            index += 1
            continue
        elif key == ord('a'):
            index -= 1
            continue
        elif key == ord('q'):
            break'''
    cv2.destroyAllWindows()

    accuracy = float(cnt_correct) / cnt_total


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--image', help='path of image directory', default='images')
    args = parser.parse_args()
    return args


class MyHandler(BaseHTTPRequestHandler):
  def do_POST(self):
     form = cgi.FieldStorage(fp=self.rfile,headers=self.headers,environ={'REQUEST_METHOD':'POST','CONTENT_TYPE':self.headers['Content-Type'],})
     filename = datetime.now().strftime('%Y%m%d-%H%M%S') + '.jpg'
     im = Image.open(io.BytesIO(base64.b64decode(re.sub('^data:image/.+;base64,', '', form["immg"].value))))
     im.save(filename)
     img = cv2.imread(filename)
     (img, detected_name)=image_detect(img)
     os.remove(filename)
     self.send_response(200)
     self.send_header('Content-type', 'text/html')
     self.end_headers()
     self.wfile.write(detected_name.encode())
     return

  def do_GET(self):
      self.path = 'index.html'
      try:
          f = open(curdir + sep + self.path)
          self.send_response(200)
          self.send_header('Content-type', 'text/html')
          self.end_headers()
          self.wfile.write(f.read().encode())
          f.close()
          return
      except IOError:
          self.send_error(404,'File Not Found: %s' % self.path)

if __name__ == '__main__':
    InitNet()
    try:
        httpd = HTTPServer((HOST_NAME, PORT_NUMBER), MyHandler)
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('^C received, shutting down the web server')
        httpd.socket.close()
    #//
    #images_detect(image_dir)
