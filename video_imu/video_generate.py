import re
import cv2
import os

def sort_key(s):
    # read the images by the number index
    numbers = re.findall(r'\d+', s)
    return tuple(int(num) for num in numbers)

# 设置输入文件夹和输出视频文件名
input_folder = 'E:/master-2/madgewick_filter/video_imu/img_imu/baseline'  # 替换为你的图片文件夹路径
output_video = 'E:/master-2/madgewick_filter/video_imu/img_imu/baseline_video.avi'     # 你的输出视频文件名

frame_width = 1920
frame_height = 1080

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, 20.0, (frame_width, frame_height))

images = [img for img in os.listdir(input_folder) if img.endswith(".jpg") or img.endswith(".png")]
images.sort(key=sort_key)

for image in images:
    img_path = os.path.join(input_folder, image)
    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (frame_width, frame_height))
    out.write(frame)

out.release()
cv2.destroyAllWindows() 
