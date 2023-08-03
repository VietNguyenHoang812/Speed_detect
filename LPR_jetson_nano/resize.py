import cv2


ori_img = cv2.imread("cropped.jpg")
resized = cv2.resize(ori_img, (640, 480))
cv2.imwrite("resized.jpg", resized)
# cv2.imshow("resized", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()