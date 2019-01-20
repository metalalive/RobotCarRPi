# the code has been tested with opencv 3.3.0
import cv2
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

# Arrow keys
UP_KEY = 82
DOWN_KEY = 84
RIGHT_KEY = 83
LEFT_KEY = 81
ENTER_KEY = 10
SPACE_KEY = 32
EXIT_KEYS = [113, 27]  # Escape and q
S_KEY = 115 # S key

# used to record label (image coordinate) for each image
labelBuffer = []



def cleanupLabelBuf ():
    global labelBuffer
    for idx in range(len(labelBuffer)):
        del labelBuffer [0]



# determine the 3 central points of the lane in current frame
# then this tool is supposed to switch to next frame .
def evtAddNewPointOnImage(event, x, y, flags, param):
    global  labelBuffer
    if event == cv2.EVENT_LBUTTONDOWN:
        if (len(labelBuffer) < (2*1)):
            labelBuffer.append (x)
            labelBuffer.append (y)
        else:
            print ("[WARN] current frame is already labeled with 3 points, press arrow key to switch to next frame.")






def addNewLabelExmaple (saved_img_path, img_file_current_idx, labelf, labelBuffer, img):
    img_out_path      = str(saved_img_path) +'/'+ str(img_file_current_idx) + '.jpg'
    img_flip_out_path = str(saved_img_path) +'/'+ str(img_file_current_idx) + '_flip.jpg'
    labelf.write(img_out_path + ", ")
    for col in labelBuffer:
        labelf.write(str(col)+", ")
    labelf.write('\n')
    cv2.imwrite(img_out_path, img)

    # augmenting the image example here, we only horizontally flip each labels image, 
    flipped = cv2.flip (img, 1) # flip around y-axis.
    height, width, channel = flipped.shape 
    cv2.imwrite(img_flip_out_path, flipped)
    for idx in range(0, len(labelBuffer)-1, 2):
        labelBuffer[idx] = width - labelBuffer[idx]
    labelf.write(img_flip_out_path + ", ")
    for col in labelBuffer:
        labelf.write(str(col)+", ")
    labelf.write('\n')
    labelf.flush()



def label_imgs_1by1(args):
    #    this function reads the recorded video back then label the frames one by one,
    #    we manually gives a label to each frame by clicking the central point of the lane
    #    in every driving case (fast moving, straight, curve, oscillate)
    #    in dataset this has to be determined by human
    video_in_path = str(args.input_file)
    LabelOutFileName = str(args.output_file)
    labelf = open (LabelOutFileName, "a+")
    
    # 0-based index of the frame to be decoded/captured next
    img0idx_prop  = cv2.CAP_PROP_POS_FRAMES
    numFrame_prop = cv2.CAP_PROP_FRAME_COUNT
    cap = cv2.VideoCapture ( video_in_path )
    curr_idx = cap.get (img0idx_prop)
    numFrame = cap.get (numFrame_prop)
    numLabeled = 0
    print ("[INFO] {} frames, we'll start from index {}".format(numFrame, curr_idx))
    pbar = tqdm(total=numFrame)
    param = []
    
    # check # frames that are already saved in the output folder.
    filelist = os.listdir( args.saved_img_path )
    existing_img_list  = [s for s in filelist if '_flip.jpg' in s]
    img_file_id_offset = len(existing_img_list) + 1

    # start grabbing frames to see if user want to label it & save to image file.
    while curr_idx < numFrame:
        while True:
            flag, frame = cap.read()
            if flag:
                break
            else:
                # the next frame isn't ready, we try to read it again later
                cap.set(img0idx_prop, curr_idx-1)
                cv2.waitKey(1000) # wait 1000 ms then read again
                continue

        try:
            cropped = frame.copy()
            height, width, channels =  frame.shape # supposed to be 320 x 240 in our USB webcam
            #### print ("[DBG] curr_idx : "+ str(curr_idx)) 
            #### if (curr_idx < 3):
            ####     print ("[DBG] (width, height) = ("+ str(width) +", "+ str(height) +") ")
            cropped = cropped [height/2:height, : ]
            cropped = cv2.resize(cropped, (width/2, height/4))
            cv2.imshow("current image", cropped)
            # setMouseCallback () must be set after imshow ()
            # if you want your mouse event detector work
            cleanupLabelBuf ()
            cv2.setMouseCallback ("current image", evtAddNewPointOnImage, param)
            key   = cv2.waitKey(0) & 0xff
            # check if you label exactly 1 point for each frame, after key event is caught
            if (len(labelBuffer) == (2*1)) :
                addNewLabelExmaple (args.saved_img_path, img_file_id_offset + numLabeled,
                    labelf, labelBuffer,  cropped)
                numLabeled += 1
            if key in EXIT_KEYS:
                break
            #### elif key in [RIGHT_KEY, ENTER_KEY]:
            else :
                curr_idx += 1
        except cv2.error as e:
            print ("[ERROR] caught exception when processing image "+ str(numLabeled) +". \r\n")
        pbar.update(1)

    labelf.close()
    pbar.close()
    cap.release()
    cv2.destroyAllWindows()
    print ("[INFO] total number of images labeled : "+ str(numLabeled))
# end of label_imgs_1by1 ()



def record_lanes(args):
    # 1. record the race track
    # be sure to use our USB webcam, that will be used on RPi
    cap = cv2.VideoCapture(str(args.input_file))
    video_out_path = str(args.output_file)
    if (cap.isOpened()==False):
        print("Error opening video stream or file")
        exit()
    cap.set (cv2.CAP_PROP_FRAME_WIDTH ,320)
    cap.set (cv2.CAP_PROP_FRAME_HEIGHT,240)
    width  = int(cap.get (cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get (cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
    outvdo = cv2.VideoWriter(video_out_path, fourcc, 20.0, (width, height))
    while(cap.isOpened()):
        # capture frame by frame
        ret, frame = cap.read()
        if (ret):
            # write the flipped frame
            outvdo.write(frame)
            cv2.imshow('current frame', frame)
            # press Q on keyboard to exit
            if cv2.waitKey(25) & 0xff == ord('q'):
                break
        else: 
            print("[ERROR] failed to capture frame from camera. \r\n")
            break
    cap.release()
    outvdo.release()
    cv2.destroyAllWindows()
# end of record_lanes()






def main():
    parser = argparse.ArgumentParser(description='Create your dataset for lane-line detection, see https://github.com/metalalive/RobotCarRPi/blob/master/README.md for description')
    parser.add_argument('-r', '--record_mode', action='store_true', help='record mode, recording lane line & save the frames into video files' )
    parser.add_argument('-l', '--label_mode',  action='store_true', help='label mode , labeling selected frames of a recorded video.' )
    parser.add_argument('-s', '--saved_img_path', help='specify the path to save labeled images, used only within label mode', default='./', type=str, required=False)
    parser.add_argument('-i', '--input_file',  help='input files. Within record mode (-r), input files must be the path of your video device e.g. /dev/video0;With label mode, input files must be the path of recorded video. ', default='./xxxx', type=str, required=True )
    parser.add_argument('-o', '--output_file', help='output files. Within record mode (-r), the output must be the path of your recording video ; With label mode, the output must be the path of label file (the frames will be saved to image files,). ', default='./xoxo.csv', type=str, required=True )
    parser.set_defaults(flag=True)
    args = parser.parse_args()
    if (args.record_mode):
        record_lanes(args)
    elif (args.label_mode):
        label_imgs_1by1(args)
    else:
        print ("[WARNING] type -h to check usage, you must include either -r (record mode) or -l (label mode)")
# end of main function



if __name__=="__main__":
    main()

