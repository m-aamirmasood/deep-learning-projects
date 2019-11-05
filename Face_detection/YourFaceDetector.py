import cv2
import os
import argparse
import json
import numpy as np
from viola_jones import ViolaJones


json_list = []
def image_loading(folder):
    img = []
    for filename in os.listdir(folder):
        if filename.endswith('.ppm') or filename.endswith('.jpg'):
            img.append(filename)
    img.sort()

    return img

def json_parsing(filename,locations):
    for i in range(0,len(locations)):
        x,y,x2,y2 = locations[i]
        element = {"iname": filename, "bbox": [int(x), int(y), int(x2-x), int(y2-y)]}
        json_list.append(element)

def saveJson():
    output_json = "results.json"
    with open(output_json, 'w') as f:
        json.dump(json_list, f)

def parse_args():
    parser = argparse.ArgumentParser(description="lalalala")
    parser.add_argument('string', type=str, default="./data/",help="image folder")
    args = parser.parse_args()
    return args

def non_max_sup(dab, Threshold):
	if len(dab) == 0:
		return []
	p = []
	x1 = dab[:,0]
	y1 = dab[:,1]
	x2 = dab[:,2]
	y2 = dab[:,3]

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		p.append(i)
		suppress = [last]
		for pos in range(0, last):
			j = idxs[pos]
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			overlap = float(w * h) / area[j]
			if overlap > Threshold:
				suppress.append(pos)
		idxs = np.delete(idxs, suppress)
	return dab[p]

if __name__ == '__main__':
    args = parse_args()
    test_image_location = args.string
    test_image = image_loading(test_image_location)
    for i in range(len(test_image)):
            imgtest = cv2.imread(test_image_location+'/'+ test_image[0])
            test_image = cv2.cvtColor(imgtest,cv2.COLOR_RGB2GRAY)
            windowsize_r = 100
            windowsize_c = 100
            listofsegments = []
            filename = "45"
            clf = ViolaJones.load(filename)
            locations = []
            while windowsize_r<(len(test_image)-2):
                for r in range(0,test_image.shape[0] - windowsize_r, 10):
                    for c in range(0,test_image.shape[1] - windowsize_c, 10):
                        window = test_image[r:r+windowsize_r,c:c+windowsize_c]
                        window=cv2.resize(window,dsize=(24,24))
                        prediction = clf.category(window)
                        if prediction ==1:
                            arr = [r,c,r+windowsize_r,c+windowsize_c]
                            locations.append(arr)
                windowsize_r+=50
                windowsize_c+=50

            locations = non_max_sup(np.array(locations), 0.2)
            json_parsing(test_image[i],locations)

            for i in range(0,len(locations)):
                x,y,x2,y2 = locations[i]
                cv2.rectangle(imgtest,(x,y),(x2,y2),(255,0,0),2)
            cv2.imwrite('result.jpg',imgtest)
    print(json_list)
    saveJson()
