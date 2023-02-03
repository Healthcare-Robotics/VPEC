import os
import requests
import argparse
import zipfile
from tqdm import tqdm
from recording.util import mkdir

download_zips = {
    'test/soft/test_soft_1.zip':        'https://www.dropbox.com/s/8l0m5wrl7fuc75a/test_soft_1.zip?dl=1',
    'test/soft/test_soft_2.zip':        'https://www.dropbox.com/s/8az9m3ri7hmyi2n/test_soft_2.zip?dl=1',
    'test/stretch/test_stretch_1.zip':  'https://www.dropbox.com/s/9si8ozkf3w48l3z/test_stretch_1.zip?dl=1',
    'test/stretch/test_stretch_2.zip':  'https://www.dropbox.com/s/8u5a21zqifnuuzo/test_stretch_2.zip?dl=1',
    'train/soft/train_soft_1.zip':      'https://www.dropbox.com/s/nm3791lkhyuv79b/train_soft_1.zip?dl=1',
    'train/soft/train_soft_2.zip':      'https://www.dropbox.com/s/ua10m6tloxahr2a/train_soft_2.zip?dl=1',
    'train/soft/train_soft_3.zip':      'https://www.dropbox.com/s/emg4z6k9voquadw/train_soft_3.zip?dl=1',
    'train/soft/train_soft_4.zip':      'https://www.dropbox.com/s/ol9ch4yepwv90qs/train_soft_4.zip?dl=1',
    'train/soft/train_soft_5.zip':      'https://www.dropbox.com/s/flr3mx7kk3kro51/train_soft_5.zip?dl=1',
    'train/soft/train_soft_6.zip':      'https://www.dropbox.com/s/3jfvbh48g4poiym/train_soft_6.zip?dl=1',
    'train/soft/train_soft_7.zip':      'https://www.dropbox.com/s/28cl4oksxrc4imm/train_soft_7.zip?dl=1',
    'train/soft/train_soft_8.zip':      'https://www.dropbox.com/s/u9e2yv5ilfezex0/train_soft_8.zip?dl=1',
    'train/soft/train_soft_9.zip':      'https://www.dropbox.com/s/q4so48ednxoc17w/train_soft_9.zip?dl=1',
    'train/soft/train_soft_10.zip':     'https://www.dropbox.com/s/ixbtswir3s6eygx/train_soft_10.zip?dl=1',
    'train/stretch/train_stretch_1.zip': 'https://www.dropbox.com/s/19k4xcfzsyonjre/train_stretch_1.zip?dl=1',
    'train/stretch/train_stretch_2.zip': 'https://www.dropbox.com/s/d0nmhk0jaoi8s30/train_stretch_2.zip?dl=1',
    'train/stretch/train_stretch_3.zip': 'https://www.dropbox.com/s/ctdgdljiqrg6flg/train_stretch_3.zip?dl=1',
    'train/stretch/train_stretch_4.zip': 'https://www.dropbox.com/s/edmrfnyo0ljyw2h/train_stretch_4.zip?dl=1',
    'train/stretch/train_stretch_5.zip': 'https://www.dropbox.com/s/qpuyexfbqczoz17/train_stretch_5.zip?dl=1',
    'train/stretch/train_stretch_6.zip': 'https://www.dropbox.com/s/cqnepj1ppzpb9pi/train_stretch_6.zip?dl=1',
    'train/stretch/train_stretch_7.zip': 'https://www.dropbox.com/s/f9xucf1qf70blv2/train_stretch_7.zip?dl=1',
    'train/stretch/train_stretch_8.zip': 'https://www.dropbox.com/s/o5joil76wlfceao/train_stretch_8.zip?dl=1',
}

SAVE_ROOT_DIR = 'data'


def download_model_checkpoint():
    save_path = os.path.join(SAVE_ROOT_DIR, 'model/stretch_nopinch_59.pt')
    download_file_from_url('https://www.dropbox.com/s/xmv4iihjg7yl68c/stretch_nopinch_59.pt?dl=1', save_path)

    save_path = os.path.join(SAVE_ROOT_DIR, 'model/soft_nopinch_59.pt')
    download_file_from_url('https://www.dropbox.com/s/64r1ohi2lt0sxo9/soft_nopinch_59.pt?dl=1', save_path)


def download_all_zips():
    for file_path in download_zips:
        save_zip_path = os.path.join(SAVE_ROOT_DIR, file_path)
        final_dir_path = os.path.dirname(save_zip_path)

        download_file_from_url(download_zips[file_path], save_zip_path)     # Download the file

        print('Unzipping:', save_zip_path)
        with zipfile.ZipFile(save_zip_path, 'r') as zip_ref:
            zip_ref.extractall(final_dir_path)

        os.remove(save_zip_path)    # Delete the zip file
        print('Finished unzipping:', save_zip_path)


def download_file_from_url(url, filename):
    """
    Download file from a URL to filename,
    displaying progress bar with tqdm
    taken from https://stackoverflow.com/a/37573701
    and https://github.com/facebookresearch/ContactPose/blob/main/utilities/networking.py
    """

    print('Downloading:', filename)
    mkdir(filename, cut_filename=True)

    try:
        r = requests.get(url, stream=True)
    except ConnectionError as err:
        print(err)
        return False

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)

    done = True
    datalen = 0
    with open(filename, 'wb') as f:
        itr = r.iter_content(block_size)
        while True:
            try:
                try:
                    data = next(itr)
                except StopIteration:
                    break

                t.update(len(data))
                datalen += len(data)
                f.write(data)
            except KeyboardInterrupt:
                done = False
                print('Cancelled')
            except ConnectionError as err:
                done = False
                print(err)

    t.close()

    if (not done) or (total_size != 0 and datalen != total_size):
        print("ERROR, something went wrong")
        try:
            os.remove(filename)
        except OSError as e:
            print(e)
        return False
    else:
        return True


if __name__ == "__main__":
    download_model_checkpoint()
    download_all_zips()

