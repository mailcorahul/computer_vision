import argparse
import cv2
import MSER

if __name__ == '__main__':

	parser = argparse.ArgumentParser() ;
	parser.add_argument('--path', type=str, help='input image path') ;
	args = parser.parse_args() ;

	start_time = time.time() ;
	gray = cv2.imread(args.path, 0) ;
	rgb = cv2.imread(args.path) ;

	words = MSER(gray, rgb, 1) ;
	vis = rgb.copy();

	for br in words:
		cv2.rectangle(vis, (br[0], br[1]), (br[2], br[3]), (0, 0, 255), 2);

	cv2.imwrite('output/detected_text.png', vis);
