import numpy as np
import cv2
import threading
import os
import copy
import sys
import argparse
import time


save = False


def ready_save():
    global save
    save = True


def change_class(photo_interval):
    # schedule the next photo shoot
    threading.Timer(photo_interval, ready_save).start()
    return


def video_feed(args):
    global save
    face_cascade = ""
    cascade_adjust = 25
    if hasattr(args, 'har'):
        face_cascade = cv2.CascadeClassifier(args.har)
        if face_cascade == "":
            raise ValueError('har cascade directory is not valid')
    cnt = 0
    lbl = 0
    loop = True
    cap = cv2.VideoCapture(0)
    screen_txt = '%s : %s / %s' % (args.labels[lbl], cnt+1, args.number_of_samples)

    # to avoid starting a threading timer inside the loop
    threading.Timer(args.photo_interval, ready_save).start()
    # web camera warm up
    time.sleep(2)
    # run the camera loop
    while (loop):
        # exit the camera loop prematurely with the q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            loop = False
        # Capture frame-by-frame
        ret, frame = cap.read()
        # print requested pose
        no_text_frame = copy.copy(frame)
        cv2.rectangle(frame, (0, 0), (720, 50), (255, 255, 255), -1)
        cv2.putText(frame, screen_txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
        cv2.imshow('frame', frame)

        if save:
            photo_output_directory_raw = '%s\\raw\\%s\\%s%s.jpg' % (args.output_directory, args.labels[lbl],
                                                                args.labels[lbl], cnt)
            photo_output_directory_crop = '%s\\cropped\\%s\\%s%s.jpg' % (args.output_directory, args.labels[lbl],
                                                                    args.labels[lbl], cnt)

            if not os.path.exists(os.path.dirname(photo_output_directory_raw)):
                os.makedirs(os.path.dirname(photo_output_directory_raw))
            if not os.path.exists(os.path.dirname(photo_output_directory_crop)):
                os.makedirs(os.path.dirname(photo_output_directory_crop))

            # write file cropped
            if face_cascade != "":
                # detect object (as per input har-cascade)
                faces = face_cascade.detectMultiScale(frame, 1.3, 5)

                # failed to detect a face. Save is still true, so next iteration will try again.
                if len(faces) < 1:
                    continue

                (x, y, w, h) = faces[0]
                roi = no_text_frame[y:y + h, x:x + w]
                roi_with_margins = no_text_frame[(y - cascade_adjust): y + (h + cascade_adjust),
                                                 (x - cascade_adjust): x + (w + cascade_adjust)]
                cv2.imwrite(photo_output_directory_crop, roi_with_margins)
                print(photo_output_directory_crop)
            else:
                cv2.imwrite(photo_output_directory_raw, no_text_frame)
                print(photo_output_directory_raw)
            save = False
            cnt = cnt + 1
            screen_txt = '%s : %s / %s' % (args.labels[lbl], cnt + 1, args.number_of_samples)

            # when all samples are taken, move to the next label and start from the first sample
            if cnt == args.number_of_samples:
                lbl = lbl + 1
                # stop scheduling captures once all labels have had photos captured
                if lbl == len(args.labels):
                    break  # exits the while loop
                else:
                    screen_txt = 'NEW class:%s' % (args.labels[lbl])
                    threading.Timer(args.photo_interval, change_class, args=[args.photo_interval]).start()

                cnt = 0
            else:
                # schedule the next photo shoot
                threading.Timer(args.photo_interval, ready_save).start()

    cap.release()
    cv2.destroyAllWindows()
    return


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('output_directory', type=str, help='Directory to save captured photos.')
    parser.add_argument('labels', type=str, help='Classes of images to create', nargs='+')
    parser.add_argument('--number_of_samples', type=int, help='number of photos per class.', default=5)
    parser.add_argument('--photo_interval', type=float, help='time between photo shoots', default=5)
    parser.add_argument('--har', type=str, help='the har-classifer to use for auto cropping')
    return parser.parse_args(argv)

if __name__ == '__main__':
    video_feed(parse_arguments(sys.argv[1:]))
