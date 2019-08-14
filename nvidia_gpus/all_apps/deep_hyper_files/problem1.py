
# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/yzamora/perf_pred/deephyper_repo/deephyper/')

from deephyper.benchmark import HpProblem

Problem = HpProblem()
#Width of hidden layers
Problem.add_dim('nunits', (1, 1000))
Problem.add_dim('depth', (1,20))
#Problem.add_dim('nunits_l2', (1, 1000))
Problem.add_dim('activation', ['relu', 'elu', 'selu', 'tanh'])
Problem.add_dim('batch_size', (8, 100))
#Problem.add_dim('dropout_l1', (0.0, 1.0))
#Problem.add_dim('dropout_l2', (0.0, 1.0))


Problem.add_starting_point(
    nunits=1,
    activation='relu',
    batch_size=8,
    depth=1
    )


if __name__ == '__main__':
    print(Problem) 
