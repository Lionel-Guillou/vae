# import general libraries
import sys

# import modules from package
from backend.model import Fashion

# entry point into model inference
def main():
    n_args = len(sys.argv)
    if n_args <= 1:
        print("No arguments passed.")
    else:
        model = Fashion()
        idx = int(sys.argv[1])
        if n_args == 2:
            res = model(idx)
        elif n_args == 3:
            visualize = (sys.argv[2] == "True")
            res = model(idx, visualize)
        else:
            print("Too many arguments passed.")        

if __name__ == "__main__":
    main()