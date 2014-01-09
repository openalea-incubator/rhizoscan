"""
Using opencv descriptors

The code here is taken, or inspired from:
https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_tutorials.html
"""
import cv2
import numpy as np


def detect_sift(filename, output_file=None):
    img = cv2.imread(filename)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray,None)
    
    if output_file:
        img=cv2.drawKeypoints(gray,kp)
        cv2.imwrite(output_file,img)
    
    return kp, des
    
def detect_orb(image_file, output_file=None, max_features=2000):
    img = cv2.imread(image_file,0)
    
    # Initiate STAR detector
    orb = cv2.ORB(nfeatures=max_features)
    
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    
    # draw only keypoints location,not size and orientation
    if output_file:
        img2 = cv2.drawKeypoints(img,kp,color=(255,255,255), flags=0)
        cv2.imwrite(output_file,img)
    
    return kp,des
    

def tracker(image1,image2, output_file=None, alpha=0.75, matching='FLANN'):
    """
    Compute the affine transformation between `image1` and `image2`
    
    `image1` and `image2` can be either:
      - the filenames of a the images to load
      - numpy array or (opencv) iplimage
      - triple (imgs,key-points,descriptors) such as is returned by this function
      
    :Outputs:
      - the affine transformation matrix
      - triple (img1,kp1,desc1)   - (*)
      - triple (img2,kp2,desc2)   - (*)
      
      (*) img#  is the image processed
          kp#   is the detected key-points (position in image)
          desc# is the respective descriptor
          
          Those are returned in order to be easily reused, and not re-computed
    """
    # Initiate SIFT detector
    sift = cv2.SIFT()
    
    # load image and find the keypoints-descriptors with SIFT
    if isinstance(image1,basestring):
        print 'load and compute sift in:', image1
        img1 = cv2.imread(image1,0) # queryImage
        kp1, des1 = sift.detectAndCompute(img1,None)
    elif isinstance(img1,(np.ndarray,cv2.cv.iplimage)):
        kp1, des1 = sift.detectAndCompute(img1,None)
    else:
        img1, kp1, des1 = image1
        
    if isinstance(image2,basestring):
        print 'load and compute sift in:', image2
        img2 = cv2.imread(image2,0) # trainImage
        kp2, des2 = sift.detectAndCompute(img2,None)
    elif isinstance(img2,(np.ndarray,cv2.cv.iplimage)):
        kp2, des2 = sift.detectAndCompute(img2,None)
    else:
        img2, kp2, des2 = image2
    
    
    # BFMatcher with default params
    print 'matching (using ' + matching + ')'
    if matching.lower()=='bf':
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
    else:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < alpha*n.distance:
            good.append(m)
    
    MIN_MATCH_COUNT = 10
    if len(good)<MIN_MATCH_COUNT:
        raise RuntimeError("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    else:
        print 'image transformation'
        pts1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        pts2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC,5.0)

        if output_file:
            print 'saving in', output_file
            wrapImage(img2, M, img1.shape[:2], output_file=output_file)

        return M, (img1, kp1,des2), (img2, kp2,des2)

def wrapImage(img, T, shape, output_file=None):
    """
    Transform gray `img` by transformation matrix `T`
    
    if `output_file` is not None, save transformed image into it
    """
    dst = cv2.warpPerspective(img,T,shape[::-1])
    if output_file is not None:
        cv2.imwrite(output_file,dst)
    return dst

def sequence_tracker(filenames, out_dir=None, matching='BF'):
    """
    Iteratively call `tracker` on the `filenames` image sequence
    """
    from shutil  import copy2
    from os.path import join, split
    
    if isinstance(filenames,basestring):
        with open(filenames) as f:
            filenames = [fname.strip() for fname in f.readlines()]
            
    T = [None]*len(filenames)
    images = filenames[:]
            
    # process 1st pair of images
    # --------------------------
    T[1], images[0], images[1] = tracker(images[0], images[1], matching=matching)
    
    if out_dir is not None:
        cv2.imwrite(join(out_dir,split(filenames[0])[-1]),images[0][0])
        wrapImage(images[1][0], T[1], images[1][0].shape, output_file=join(out_dir,split(filenames[1])[-1]))

    # process all others
    # ------------------
    for i in range(1,len(images)-1):
        j = i+1
        T[j], images[i], images[j] = tracker(images[i], images[j], matching=matching)
        T[j] = np.dot(T[i],T[j])
        if out_dir is not None:
            wrapImage(images[j][0], T[j], images[0][0].shape, output_file=join(out_dir,split(filenames[j])[-1]))
        
    return T        

if __name__ == "__main__":
   import sys
   from ast import literal_eval
   method = sys.argv[1]
   args = sys.argv[2:]
   
   def eval_input(x):
       try:
           return literal_eval(x)
       except:
           return x
   
   if method.lower()=='sift': method=detect_sift
   if method.lower()=='orb':  method=detect_orb
   else:                      method=tracker
   
   method(*map(eval_input,args))#filename=f_input,output_file=f_output)
   