import sys
import argparse
from xitorch.debug.modes import enable_debug

def parse_args():
    # parse the argument
    parser = argparse.ArgumentParser("Run python script by enabling xitorch debug mode")
    parser.add_argument("scriptfile", type=str, help="Path to the script to run")
    parser.add_argument("args", type=str, nargs=argparse.REMAINDER,
                        help="The arguments needed to run the script")
    return parser.parse_args()

def main():
    args = parse_args()

    scriptfile = args.scriptfile
    scriptargs = args.args if args.args is not None else []
    scriptargs.insert(0, scriptfile)
    sys.argv[:] = scriptargs[:]

    # compile and run the code with debug mode enabled
    with enable_debug():
        with open(scriptfile, 'rb') as stream:
            code = compile(stream.read(), scriptfile, 'exec')
        globs = {
            '__file__': scriptfile,
            '__name__': '__main__',
            '__package__': None,
            '__cached__': None,
        }
        exec(code, globs, None)


if __name__ == "__main__":
    main()
