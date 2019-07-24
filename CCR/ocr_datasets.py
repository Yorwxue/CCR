import os
import glob


class OCRDatasets(object):
    def __init__(self):
        return

    def get_samples(self, split_name, data_dir="./imgs"):
        # sample_pattern = './imgs/{}/*.png'.format(split_name)
        sample_pattern = os.path.join(data_dir, split_name, "*.jpg")
        samples = glob.glob(sample_pattern)
       
        return samples

    def run(self):
        samples = self.get_samples('val', data_dir="./imgs")
        for sample in samples:
            print(sample)
        print(len(samples))
        return


if __name__ == "__main__":   
    obj = OCRDatasets()
    obj.run()
