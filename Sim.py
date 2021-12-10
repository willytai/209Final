from uArch import uArch
from UNet.data import saveResult
import argparse, time

'''
original_unet:
Download model weights from: 'https://drive.google.com/file/d/1ED_2y6CAPgSV4XP-Ytnb-5D8bijMZITr/view?usp=sharing'

naming convention:
    - class names are capitalized
    - function params are <word1>_<word2>
    - function names and variables are <word1><Word2>
    - class functions meant for solely itself are named with a leading underscore
'''

def main(args: argparse.Namespace) -> None:
    starttime = time.time()
    arch = uArch(pe_array_size=args.pe, verbose=args.verbose)
    arch.loadModel(args.model)
    arch.loadWeight(args.weight)
    arch.setComputationMode(args.fixed)
    out = arch.run(args.input)
    arch.showUsage()
    saveResult(npyfile=out, save_path=args.output, save_name=args.input)
    elapsed = int(time.time() - starttime)
    print ('time elapsed: {}m {}s'.format(elapsed // 60, elapsed % 60))

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=False, default='uArchResult')
    parser.add_argument('--pe', type=int, required=False, default=32)
    parser.add_argument('--fixed', type=int, required=False, default=-1)
    parser.add_argument('--verbose', type=int, required=False, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    main(parseArgs())
