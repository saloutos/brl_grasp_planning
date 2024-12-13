import os
import sys
import json
import urllib
from urllib.request import urlopen
import subprocess
from utils import get_base_path

# NOTE: need to install obj2mjcf to use this script

output_directory = "./ycb"

# You can either set this to "all" or a list of the objects that you'd like to download.
objects_to_download = "all"
#objects_to_download = ["002_master_chef_can", "003_cracker_box"]

# You can edit this list to only download certain kinds of files.
# 'berkeley_rgbd' contains all of the depth maps and images from the Carmines.
# 'berkeley_rgb_highres' contains all of the high-res images from the Canon cameras.
# 'berkeley_processed' contains all of the segmented point clouds and textured meshes.
# 'google_16k' contains google meshes with 16k vertices.
# 'google_64k' contains google meshes with 64k vertices.
# 'google_512k' contains google meshes with 512k vertices.
# See the website for more details.
files_to_download = ["berkeley_processed"]

# Extract all files from the downloaded .tgz, and remove .tgz files.
# If false, will just download all .tgz files to output_directory
extract = True
base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
objects_url = base_url + "objects.json"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def fetch_objects(url):
    with urlopen(url) as response:
        html = response.read()
    objects = json.loads(html)
    return objects["objects"]

def download_file(url, filename):
    with urlopen(url) as u, open(filename, 'wb') as f:
        meta = u.info()
        file_size = int(meta.get("Content-Length"))

        print("Downloading: %s (%s MB)" % (filename, file_size/1000000.0))

        file_size_dl = 0
        block_sz = 65536
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl/1000000.0, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print(status)

def tgz_url(object, type):
    if type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return base_url + "berkeley/{object}/{object}_{type}.tgz".format(object=object,type=type)
    elif type in ["berkeley_processed"]:
        return base_url + "berkeley/{object}/{object}_berkeley_meshes.tgz".format(object=object,type=type)
    else:
        return base_url + "google/{object}_{type}.tgz".format(object=object,type=type)

def extract_tgz(filename, dir):
    tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename,dir=dir)
    os.system(tar_command)
    os.remove(filename)

def check_url(url):
    try:
        request = urllib.request.Request(url)
        request.get_method = lambda: 'HEAD'

        try:
            urllib.request.urlopen(request)
            return True
        except urllib.error.URLError as e:
            return False
    except Exception as e:
        return False

def convert_ycb_obj(obj_name, blacklist=[]):

    absolute_object_path = os.path.join(get_base_path(), "ycb", obj_name)

    data_type = "poisson"
    absolute_data_path = os.path.join(absolute_object_path, data_type)
    if not os.path.exists(absolute_data_path):
        print("Object not found: ", obj_name)
        blacklist.append(obj_name)
        return

    print(absolute_data_path)
    subprocess.run(["obj2mjcf", "--obj-dir", absolute_data_path, "--overwrite"], timeout=10)
    raw_output_path = os.path.join(absolute_data_path, "textured")

    if not os.path.exists(raw_output_path):
        blacklist.append(obj_name)
        print("Object conversion failed: ", obj_name)
        return

    processed_data_path = os.path.join(absolute_object_path, "processed")
    if os.path.exists(processed_data_path):
        subprocess.run(["rm", "-r", processed_data_path])
    subprocess.run(["mv", raw_output_path, processed_data_path])

def convert_all_ycb_obj():
    absolute_object_path = os.path.join(get_base_path(), "ycb")
    blacklist = []
    for obj_name in os.listdir(absolute_object_path):
        print("Converting: ", obj_name)
        convert_ycb_obj(obj_name, blacklist)

    blacklist_fname = absolute_object_path + "/blacklist.txt"
    with open(blacklist_fname, "w") as f:
        for obj_name in blacklist:
            f.write(obj_name + "\n")



if __name__ == "__main__":

    objects = fetch_objects(objects_url)
    print(objects)
    print(files_to_download)

    for object in objects:
        if objects_to_download == "all" or object in objects_to_download:
            for file_type in files_to_download:
                url = tgz_url(object, file_type)
                print(url)
                if not check_url(url):
                    print("Continued")
                    continue
                filename = "{path}/{object}_{file_type}.tgz".format(path=output_directory,
                                                                    object=object,
                                                                    file_type=file_type)
                download_file(url, filename)
                if extract:
                    extract_tgz(filename, output_directory)

    convert = True
    if convert:
        convert_all_ycb_obj()
