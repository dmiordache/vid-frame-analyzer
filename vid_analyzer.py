import cv2
import numpy
import matplotlib.pyplot as plt

input_vid = cv2.VideoCapture("demo_input.mp4") #generated mp4 file that I am analyzing

if not input_vid.isOpened():
    print ("Error, video file unopened")
    exit()

prev_gs_frame = None
motion_data_list = [] #list of motion pixel counts for each frame
complexity_data_list = [] #list of complexity values for each frame

while True:
    got_frame, frame = input_vid.read() # obtains the frame itself (frame), and whether frame is available (bool).

    # end the loop when there are no more frames left to analyze
    if not got_frame:
        print("video done/no more frames")
        break
    print("Frame obtained")
    flag, encoded_frame_data = cv2.imencode('.jpg', frame) #encode the frame as a jpg image
    frame_size = len(encoded_frame_data) #get the size of the encoded frame in bytes

    gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale for simplicity
    
    if prev_gs_frame is None:
        prev_gs_frame = gs_frame
        continue

    frame_diff = cv2.absdiff(prev_gs_frame, gs_frame) #framediff: absolute difference of pixel prev vs current frame (0-255)
    optimal_thresh, thresh_frame = cv2.threshold(frame_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #convert to binary image (0 or 255)
    #print("Binary Image" + str(optimal_thresh))
    motion_pixels = numpy.count_nonzero(thresh_frame)#counts the white motion-detected pixels
    motion_data_list.append(motion_pixels) #append the motion pixel count to the list
    complexity_data_list.append(frame_size) #append the complexity value to the list
    prev_gs_frame = gs_frame

input_vid.release()
print("Motion data list: ", motion_data_list)
print("Complexity data list: ", complexity_data_list)
plt.plot(motion_data_list)
plt.xlabel("Frame")
plt.ylabel("Motion Pixels")
plt.title("Motion Detection Over Time")
plt.show()

plt.plot(complexity_data_list)
plt.xlabel("Frame")
plt.ylabel("Complexity (Bytes)")
plt.title("Video Complexity Over Time")
plt.show()
