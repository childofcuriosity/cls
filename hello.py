import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_size', default=(448,448), type=tuple,
                    help='size of input images')
args = parser.parse_args()
print(parser.img_size)