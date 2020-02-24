import os
import requests
import time


def download(url, path):
    start = time.clock()
    try:
        headers = {"User-Agent":
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, \
                       like Gecko) Chrome/75.0.3770.142 Safari/537.36"}
        pre_content_length = 0
        
        while True:
            # If the file already exists, the breakpoint is continued 
            # and the location of the data to be received is set
            if os.path.exists(path):
                headers['Range'] = 'bytes=%d-' % os.path.getsize(path)

            res = requests.get(url, stream=True, headers=headers)
            # res = requests.get(url, stream=True, headers=headers)
            content_length = int(res.headers['content-length'])
            # If the length of the current message is less than the previous, 
            # or if the received file is equal to the length of the current message, 
            # the image reception can be considered complete
            if (content_length < pre_content_length) or \
               ((os.path.exists(path) and os.path.getsize(path) >= content_length)):
                break
            pre_content_length = content_length
            # Save the received image data
            with open(path, 'ab') as file:
                file.write(res.content)
                file.flush()
                print('receive data，file size : %d   total size:%d' % \
                	 (os.path.getsize(path), content_length))

    except Exception as e:
        print(e)


if __name__ == '__main__':
    f = open("image_url_t.txt", "r", encoding='UTF-8')
    lines = f.readlines()
    i = 0
    file_name_num = 1

    for line in lines:
        arr_line = line.split("\t")
        print("Current    ID    is    "+str(arr_line[0]))
        if i % 5000 == 0:
            file_name_num = file_name_num + 1
            path = "./image/" + str(file_name_num)+"/"
            print("Current    path  is    ", path)
            if not os.path.exists(path):
                os.makedirs(path)
        i = i + 1
        download(str(arr_line[1]), path + str(i) + "_" + str(arr_line[0]) + ".jpg")
        print("It is the  "+str(i)+"th image"+"\n")

