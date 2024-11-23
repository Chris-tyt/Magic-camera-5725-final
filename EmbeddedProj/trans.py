import ffmpeg

# input_file = 'face1.mp4'
# output_file = 'face2.mp4'

input_file = 'mytest.MOV'
output_file = 'mytest.mp4'

# 使用ffmpeg对视频进行转码，设置宽高为320x240
ffmpeg.input(input_file).output(output_file, vf='scale=320:240').run()
