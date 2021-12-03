from uArch import uArch
from UNet.data import saveResult
import argparse

'''
Download model weights from: 'https://drive.google.com/file/d/1ED_2y6CAPgSV4XP-Ytnb-5D8bijMZITr/view?usp=sharing'
Download model weights from: 'https://drive.google.com/file/d/17v1x95NOmZbmfGegrj8sS3KHwpQDmbYP/view?usp=sharing'

naming convention:
    - class names are capitalized
    - function params are <word1>_<word2>
    - function names and variables are <word1><Word2>
    - class functions meant for solely itself are named with a leading underscore

TODO:
    remove all sanity checks and assertions when finished
'''

def main(args: argparse.Namespace) -> None:
    arch = uArch(pe_array_size=32, verbose=0)
    arch.loadModel(args.model)
    arch.loadWeight(args.weight)
    out = arch.run(args.input)
    saveResult('uArchResult', out)

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    main(parseArgs())
