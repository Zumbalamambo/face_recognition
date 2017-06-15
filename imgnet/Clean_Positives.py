import cv2
import os

if __name__ == '__main__':

    try:
        os.remove('pos/roi.txt')
    except OSError:
        pass

    for img in os.listdir("pos"):
        if img.endswith(".jpg") or img.endswith(".jpeg"):
            # Read image
            im = cv2.imread(os.path.join('pos', img), cv2.IMREAD_GRAYSCALE)
            display = cv2.imread(os.path.join('pos', img))
            # Select ROI
            rect_pts = []  # Starting and ending points
            win_name = img

            def select_points(event, x, y, flags, param):
                global rect_pts
                if event == cv2.EVENT_LBUTTONDOWN:
                    rect_pts = [(x, y)]
                    cv2.circle(display, (x, y), 1, (0, 0, 255))
                    cv2.imshow(win_name, display)

                if event == cv2.EVENT_LBUTTONUP:
                    rect_pts.append((x, y))

                    # draw a rectangle around the region of interest
                    cv2.rectangle(display, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
                    cv2.imshow(win_name, display)


            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(win_name, select_points)
            loop = True
            while loop:
                # display the image and wait for a keypress
                cv2.putText(display, "press c to save the region", (15, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(display, "and move to next image.", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                cv2.imshow(win_name, display)
                key = cv2.waitKey(0) & 0xFF

                if key == ord("c") and len(rect_pts) > 0:  # Hit 'r' to replot the image

                    # resize the image to be 100px x npx
                    # make the images 100px x ~relative px
                    width, height = im.shape[1::-1]
                    adj = 100 / width
                    print(height * adj)
                    resized_image = cv2.resize(im, (100, int(round(height * adj))))

                    # write roi to file
                    # eg. IMG_1683.JPG 1 327 259 360 360  : filename, top left xy, length, width
                    with open('train/roi.txt', 'a') as f:
                        f.write(img + ' ' +
                                str(int(round(rect_pts[0][0]*adj))) + ' ' +
                                str(int(round(rect_pts[0][1]*adj))) + ' ' +
                                str(int(round(rect_pts[1][0]*adj - rect_pts[0][0]*adj))) + ' ' +
                                str(int(round(rect_pts[1][1]*adj - rect_pts[0][1]*adj))) + '\n')
                    # save the cropped / shrunk image
                    cv2.imwrite(os.path.join('train', img), resized_image)

                    loop = False
            cv2.destroyWindow(win_name)
