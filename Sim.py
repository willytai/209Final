from uArch import uArch
import argparse

'''
Download model weights from: 'https://drive.google.com/file/d/1ED_2y6CAPgSV4XP-Ytnb-5D8bijMZITr/view?usp=sharing'
'''

def main(args: argparse.Namespace) -> None:
    arch = uArch()
    arch.loadModel(args.model)
    arch.loadWeight(args.weight)
    arch.run(args.input);

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    main(parseArgs())