import os
import sys
import cv2
import json
import argparse
from skimage import io



def urls_to_images(input, path):
    with open(input, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        url = json.loads(line)['content']
        print(url)
        img = io.imread(url)
        file_name = os.path.join(path, f'gun_{i}.jpg')
        cv2.imwrite(file_name, img)
        print(f'Processed file {i + 1} out of {len(lines)}')



# Adding the keyword arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help='Path to input json file', required=True)
parser.add_argument('--output', '-o', help='Path to output folder', required=True)
args = parser.parse_args()

if __name__ == '__main__':
	# Error handling
	if not os.path.exists(args.input):
		sys.exit('Path given to json file does not exist.')
	elif not os.path.exists(args.output):
		os.makedirs(args.output)
	
	urls_to_images(args.input, args.output)