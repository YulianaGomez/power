## /Users/yzamora/nas_problems/nas_problems/polynome2
import time
from deephyper.benchmark import NaProblem
from deephyper.search.nas.model.preprocessing import minmaxstdscaler
from nas_problems.polynome2.load_data import load_data
from nas_problems.polynome2.architecture import create_search_space

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

#Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_search_space, num_layers=10)

Problem.hyperparameters(
    verbose=0,
    batch_size=100,
    learning_rate=0.001, #lr search: 0.01, lr post: 0.001
    optimizer='adam',
    num_epochs=50,
    callbacks=dict(
        EarlyStopping=dict(
            monitor='val_r2',
            mode='max',
            verbose=0,
            patience=5
        )
    )
)

Problem.loss('mse')

Problem.metrics(['r2'])

Problem.objective('val_r2__last')

Problem.post_training(
    num_epochs=1000,
    metrics=['r2'],
    callbacks=dict(
        ModelCheckpoint={
            'monitor': 'val_r2',
            'mode': 'max',
            'save_best_only': True,
            'verbose': 1
        },
        EarlyStopping={
            'monitor': 'val_r2',
            'mode': 'max',
            'verbose': 1,
            'patience': 50
        },
        TensorBoard=dict(
            log_dir='{}'.format(time.time()),
        ))
)

if __name__ == '__main__':
    print(Problem)
    from pprint import pprint
    pprint(Problem.space)
