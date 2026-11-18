"""python -m lowflow {preprocess,train,predict,evaluate,troute_flow,drought} --config configs/default.yml [...]"""
import sys

from . import drought, evaluate, predict, preprocess, train, troute_flow

COMMANDS = {"preprocess": preprocess.main, "train": train.main, "predict": predict.main,
            "evaluate": evaluate.main, "drought": drought.main,
            "troute_flow": troute_flow.main}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: python -m lowflow {{{','.join(COMMANDS)}}} --config <yml> [options]")
    COMMANDS[sys.argv[1]](sys.argv[2:])
