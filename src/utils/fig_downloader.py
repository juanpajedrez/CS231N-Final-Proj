"""
Date: 2023-11-14
Original Authors: Kuniaki Iwanami, Juan Pablo Triana Martinez, 
Based Project: CS236 Final Project, GMVAE for X-rays images.

Date: 2024-04-30
Current Authors: Juan Pablo Triana Martinez, Abhishek Kumar
Adapted from: Summers, Ronald (NIH/CC/DRD) [E]
"""

# Import necessary modules
import urllib.request
import os
from tqdm import tqdm

class CXRDownloader:
    """
    Class that would do the following:
    1.) Set the links to download files from.
    2.) Run the files while downloading them.
    """

    def set_data_path(self, data_path:str):
        """
        Setter method that would set the data path
        where you want the zip files to be downloade.
        """
        # Set the data path to be.
        self.data_path = data_path

        #If data path doesnt exist, return "please create it"
        if os.path.exists(self.data_path) == False:
            print("Create data folder as necessary for code")
            return

        # Check if inside the data_path folder, the
        # zip folder container exists, else, create it
        self.download_path = os.path.join(self.data_path, "Zip_files")
        if os.path.exists(self.download_path) == False:
            os.mkdir(self.download_path)

    def set_default_links(self):
        """
        Setter method that would set the default links 
        to get the signals from.
        """

        # URLs for the zip files
        self.links =[
            'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
            'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
            'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
            'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
            'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
            'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
            'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
            'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
            'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
            'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
            'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
            'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
        ]

    def download_files(self):
        """
        Getter method that would download the images in
        zip files into the download path.
        """

        # Iterate through each one of the pre set links
        for idx, link in enumerate(self.links, start=1):

            #Create a path to data folder
            fn = os.path.join(self.download_path, 'images_%02d.tar.gz' % idx)

            #if fn already exists, then continue and skip it (No need to redownload):
            if os.path.exists(fn) == True:
                print('Already downloaded ' + fn + 'PLEASE: Make sure the size and all images are there, erase and execute download again')
                continue
            print('Downloading ' + fn + '')
            with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, ncols=100) as t:
                urllib.request.urlretrieve(link, fn, reporthook=lambda blocknum, blocksize, totalsize: t.update(blocknum * blocksize - t.n))
            print('Download of ' + fn + ' complete.')

        print("All downloads complete. Please check the checksums.")