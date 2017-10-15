import cv2

class EyeTracker:
    def __init__(self, facePath, eyePath):
        self.faceCascade = cv2.CascadeClassifier(facePath)
        self.eyeCascade = cv2.CascadeClassifier(eyePath)

    def track(self, image):
        faceRects = self.faceCascade.detectMultiScale(image, scaleFactor=1.1,
            minNeighbors=5, minSize=(30,300), flags=cv2.CASCADE_SCALE_IMAGE)
        rects = []

        for(fX, fY, fW, fH) in faceRects:
            faceROI = image[fY:fY +fH, fX:fX+fW]
            rects.append((fX, fY, fX+fW, fY+fH))

            eyeRects = self.eyeCascade.detectMultiScale(faceROI, scaleFactor=1.1,
                minNeighbors = 10, minSize=(20,20))

            for(eX, eY, eW, eH) in eyeRects:
                rects.append((fX+eX, fY+eY, fX+eX+eW, fY +eY+eH))
    
        return rects
