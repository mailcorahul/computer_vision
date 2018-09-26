import cv2 ;
import numpy as np ;
from collections import OrderedDict ;

# method to sort map based on 'value' list length
def sort_map(char_map) :

	sorted_map = OrderedDict( {} ) ;
	for i in range(len(char_map)) :
		min_len = 99999 ; min_key = 0 ;
		for k , v in char_map.items() :
			# finding min length key
			if len(v) < min_len and not (k in sorted_map) :
				min_len = len(v) ;
				min_key = k ;

		sorted_map[min_key] = char_map[min_key] ;

	return sorted_map ;


def filters(img ,rect) :

	white = img.copy() ;
	white.fill(255) ;
	
	# remove duplicate rects ---- rects = msers 
	for i in range(len(rect)) :
		for j in range(len(rect)) :
			if i != j and len(rect[i]) > 0 and len(rect[j]) > 0 :				
				if rect[i][0] == rect[j][0] and rect[i][1] == rect[j][1] and rect[i][2] == rect[j][2] and rect[i][3] == rect[j][3] :				
					rect[j] = [] ;

	rect = filter(None , rect) ;
	rectToarea = {} ;

	# area for rects 
	for i in range(len(rect)) :
		rectToarea[i] = (rect[i][2] - rect[i][0]) * (rect[i][3] - rect[i][1]) ;

	# sort rects by area
	ordRects = sorted(rectToarea.items(), key=lambda x: x[1] ,reverse = True) ;

	# char container map
	char_cont = {} ;

	# adding chars falling inside another char
	for i in range(len(ordRects)) :
		rect_i = ordRects[i][0] ;
		box = rect[rect_i] ;

		for j in range(len(rect)) :
			if j != rect_i :
				# checking if char 'rect_i' encloses char j 
				if box[0] <= rect[j][0] and box[1] <= rect[j][1] and box[2] >= rect[j][2] and box[3] >= rect[j][3] :	
					if not (rect_i in char_cont) :
						char_cont[rect_i] = [] ;
					char_cont[rect_i].append(j) ;


	char_cont = sort_map(char_cont) ;

	rect.sort(key=lambda  x:int(x[0])) ;

	# iterate map and remove undesired characters 
	for k , v in char_cont.items() :
		
		if len(rect[k]) == 0 :
			continue ;
		
		w = rect[k][2] - rect[k][0] ;
		h = rect[k][3] - rect[k][1]	;
		sim_char = 0 ;

		# similar to container char , remove it
		for i in range(len(v)) :		
			if len(rect[v[i]]) > 0 and abs( w - (rect[v[i]][2] - rect[v[i]][0]) ) <= 10 and abs( h - (rect[v[i]][3] - rect[v[i]][1]) ) <= 10 :
				rect[v[i]] = [] ;				

		# if the container char encloses only one char, remove the enclosed char --- assuming it is a part of a char
		if len(v) == 1 :
			rect[v[0]] = [] ;
			continue ;

		is_same_height = True ;
		widths_sum = 0 ;

		# sum up the widths of enclosed chars
		for i in range(len(v)) :
			if len(rect[v[i]]) > 0 and abs( h - (rect[v[i]][3] - rect[v[i]][1]) ) > 10 :
				is_same_height = False ;
				break ;
			elif len(rect[v[i]]) > 0 :
				widths_sum += (rect[v[i]][2] - rect[v[i]][0]) ;

		if is_same_height :
			# contains multiple individual chars
			if abs(widths_sum - w) <= 10 :
				rect[k] = [] ;
			else:
				is_same_height = False ;

		if not is_same_height :
			for i in range(len(v)) :
				rect[v[i]] = [] ;


	rect = filter(None , rect) ;

	return rect ;