# This file contains google utils: https://cloud.google.com/storage/docs/reference/libraries
# pip install --upgrade google-cloud-storage

import os


# from google.cloud import storage


def gdrive_download(files=(('1HaXkef9z6y5l4vUnCYgdmEAj61c6bfWO', 'coco.zip'))):
    # https://gist.github.com/tanaikech/f0f2d122e05bf5f971611258c22c110f
    # Downloads a file from Google Drive, accepting presented query
    # from utils.google_utils import *; gdrive_download()

    for (id, name) in files:
        if os.path.exists(name):  # remove existing
            os.remove(name)

        # Attempt small file download
        s = 'curl -f -L -o %s https://drive.google.com/uc?export=download&id=%s' % (name, id)
        os.system(s)

        # Attempt large file download
        if not os.path.exists(name):  # file size > 40MB
            print('Google Drive file ''%s'' > 40 MB, attempting large file download...' % name)
            s = ["curl -c ./cookie -s -L \"https://drive.google.com/uc?export=download&id=%s\" > /dev/null" % id,
                 "curl -Lb ./cookie \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=%s\" -o %s" % (
                 id, name),
                 'rm ./cookie']
            [os.system(x) for x in s]  # run commands

        # Unzip if archive
        if name.endswith('.zip'):
            os.system('unzip -q %s' % name)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    # Uploads a file to a bucket
    # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))


def download_blob(bucket_name, source_blob_name, destination_file_name):
    # Uploads a blob from a bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))
