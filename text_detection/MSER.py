import nms
import cv2
import numpy as np 

def findBRect(img,hulls) :

	rectToarea = {} ;
	rect = [] ;
	boxes = [[],[],[],[]] ;
	i = 0 ;
	imgH , imgW  = img.shape ;
	meanArea = 0 ;

	for hull in hulls :
		#print hull ;
		if len(hull) < 3 :
			continue ;
		#areas.append(cv2.contourArea(hull)) ;
		minX = img.shape[1] + 1 ;
		minY = img.shape[0] + 1 ;
		maxX = -1 ;
		maxY = -1 ;
		for h in hull :
			if h[0][0] < minX :
				minX = h[0][0] ;
			if h[0][0] > maxX :
				maxX = h[0][0] ;
			if h[0][1] < minY :
				minY = h[0][1] ;
			if h[0][1] > maxY :
				maxY = h[0][1] ;
		#print minX , minY , maxX , maxY ;
		#areas.append( cv2.contourArea(np.array([[[minX , minY]] ,[[minX , maxY]],[[ maxX , maxY ]] , [[maxX , minY]]])) );
		rectToarea[i] = (maxX - minX) * (maxY - minY) ;#cv2.contourArea(np.array([[[minX , minY]] ,[[minX , maxY]],[[ maxX , maxY ]] , [[maxX , minY]]])) ;
		if rectToarea[i] > (imgH * imgW) / 3 :
			continue ;

		rect.append([minX,minY,maxX,maxY]) ;
		boxes[0].append(minX) ;
		boxes[1].append(minY) ;
		boxes[2].append(maxX) ;
		boxes[3].append(maxY) ;
		meanArea += rectToarea[i] ;

		i += 1 ;		
	return rect , boxes , rectToarea ;

def delOverlapRects(copy ,rect, ordRects) :

	copy.fill(255) ;
	for i in range(len(ordRects)) :
		pts = rect[ordRects[i][0]] ;
		cropped = copy[pts[1]:pts[3],pts[0]:pts[2]] ;

		# if the area is completely black , then ignore the rect
		if ( np.sum(cropped) == 0 ) :
			rect[ordRects[i][0]] = [] ; # mark rectangle as ignored
		# enclosing / overlapping rectangle
		else :
			copy[pts[1]:pts[3],pts[0]:pts[2]] = 0 ;

	return rect ;



def ARFilter(rect) :

	i = 0 ;
	# remove rectangles based on aspect ratio
	for br in rect :
		if len(br) > 0 :
			tw = br[2] - br[0] ; th = br[3] - br[1] ;
			if tw < th :
				temp = th ;
				th = tw ;
				tw = temp
			if th > 0:
				if (float(tw) / th) > 5 :
				#print float(br[2] - br[0]) / (br[3] - br[1]) ;
					#print 'Wrong : ' , float(tw) / th ;
					rect[i] = []
				i += 1 ;

	return rect ;

# method to find MSERs from a given image and return word boundaries 
def MSER(gray, rgb, scale_factor):

	mser = cv2.MSER_create() ;
	regions = mser.detectRegions(gray) ;
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]

	# find bounding rectangle from msers detected
	rect, boxes, rectToarea = findBRect(gray, hulls) ;

	# rescale points by scale factor
	for j in range(len(rect)) :
		for i in range(len(rect[j])) :
			rect[j][i] =  int(float(rect[j][i]) * scale_factor) ;

	return rect;

