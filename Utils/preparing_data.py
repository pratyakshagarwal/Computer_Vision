import cv2
import time
import uuid
import os


# function to capture images and name and store them in the specified filepath
def capture_images(labels, images_path, number_imgs):
    for label in labels:
        # connect the webcam 
        cap = cv2.VideoCapture(0)
        print('Collecting images for {}'.format(label))
        time.sleep(5)
        for imgnum in range(number_imgs):
            print('Collecting image {}'.format(imgnum))
            ret, frame = cap.read()
            imgname = os.path.join(images_path, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
            cv2.imwrite(imgname, frame)
            time.sleep(2)
          
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()