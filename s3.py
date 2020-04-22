import boto3
import os
import json

META_DATA_FILE_NAME = 'gen_info.json'


def retrieve_object_key(meta_data, filename):
    """
    Method for determining S3 object keys.

    :param meta_data: Dictionary containing meta data for the data set.
    :param filename: The name of the object
    :return:
    """

    return '/'.join([f'channels={meta_data["num_channels"]}',
                     f'samples={meta_data["num_instances"]}',
                     f'timesteps={meta_data["num_timesteps"]}',
                     f'max_liquid_modes={meta_data["n_max"]}',
                     f'max_shell_modes={meta_data["n_max_s"]}',
                     f'omega_shift={meta_data["omega_shift"]}',
                     f'delta_gamma={meta_data["dg"]}',
                     f'delta_gamma_s={meta_data["dgs"]}',
                     f'scale={meta_data["scale"]}',
                     filename])


class S3:
    """
    Wrapper class for BOTO 3 S3 client.
    """

    def __init__(self, bucket):
        self.client = boto3.client('s3')
        self.bucket = bucket

    def download_from_meta_data(self, path_to_meta_data, download_location):
        """
        Downloads a dataset that was generated with a specific configuration provided in a meta data file.

        :param path_to_meta_data: The path to the gen_info.json file for the dataset
        :param download_location: The path to the directory where the files will be downloaded
        :return: None
        """
        with open(path_to_meta_data) as meta_data_file:
            meta_data = json.load(meta_data_file)
            base_key = retrieve_object_key('', meta_data)
            return self.download(base_key, download_location)

    def download(self, base_key, download_location):
        """
        Downloads all objects that exist with the base key.
        :param base_key: The object key prefix that we download from.
        :param download_location: The path to the directory where the files will be downloaded.
        :return: None
        """
        data_set_parts = self.client.list_objects_v2(Bucket=self.bucket, Prefix=base_key)

        for part in data_set_parts['Contents']:
            part_key = part['Key']
            part_name = part_key.split('/')[-1]
            self.client\
                .download_file(BUCKET_NAME, part_key, os.path.join(download_location, part_name))

    def upload_json(self, json, meta_data, filename):
        """
        Creates an S3 object from a json string and Meta data.
        :param json: JSON string
        :param meta_data: Meta data
        """
        object_key = retrieve_object_key(meta_data, filename)
        self.client\
            .put_object(Bucket=self.bucket, Key=object_key, Body=json)

    def upload_from_path(self, path_to_data):
        """
        Uploads a dataset to S3 from the local machine.
        :param path_to_data: Path to the directory that contains the meta data and data files for the dataset.
        :return: None
        """
        with open(os.path.join(path_to_data, META_DATA_FILE_NAME)) as meta_data_file:
            meta_data = json.load(meta_data_file)
            for file_name in os.listdir(path_to_data):
                object_key = retrieve_object_key(file_name, meta_data)
                self.client\
                    .upload_file(os.path.join(path_to_data, file_name), self.bucket, object_key)
