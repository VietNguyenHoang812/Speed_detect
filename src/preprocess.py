import cv2
import os


def variance_of_Laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

if __name__ == "__main__":
    saved_dir = "images/blurry"
    os.makedirs(saved_dir, exist_ok=True)
    saved_dir_laplacian = "images/edges"
    os.makedirs(saved_dir_laplacian, exist_ok=True)

    threshold = 100

    val_folder = "datasets/type_1_license_plate/images/val"
    # image_name = "P1690088.JPG"
    image_name = "P1690067.JPG"
    image = cv2.imread(f"{val_folder}/{image_name}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # save edge
    half = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
    # cv2.imshow("half", half)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    gray_half = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{saved_dir_laplacian}/{image_name}", cv2.Laplacian(gray_half, cv2.CV_64F))

    fm = variance_of_Laplacian(gray_half)
    text = "Sharp"

    if fm < threshold:
        text = "Blurry"

    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)
    cv2.imwrite(f"{saved_dir}/{image_name}", image)