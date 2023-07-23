import cv2


test = "P1680238.JPG"
test = cv2.imread(test)
cv2.imshow("test", test)
cv2.waitKey(0)
cv2.destroyAllWindows()