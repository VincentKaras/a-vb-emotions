"""
Training script (+ evaluation of trained model)

"""

from trainer import Trainer
from parsers import Options

if __name__ == "__main__":
    
    opts = Options()
    params = opts.parse()

    print("\n Creating trainer ...")

    tr = Trainer(params=params)

    tr.train()
    tr.test()

    print("\n Training script complete!")