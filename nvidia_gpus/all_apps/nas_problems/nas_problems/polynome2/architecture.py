import collections

import tensorflow as tf

from deephyper.search.nas.model.space.architecture import AutoKArchitecture
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.basic import Tensor
from deephyper.search.nas.model.space.op.connect import Connect
from deephyper.search.nas.model.space.op.merge import AddByPadding, AddByProjecting
from deephyper.search.nas.model.space.op.op1d import (Dense, Dropout, Identity)



def add_dense_to_(node):
    node.add_op(Identity()) # we do not want to create a layer in this case

    activations = [None, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
    for units in range(16, 97, 16):
        for activation in activations:
            node.add_op(Dense(units=units, activation=activation))


def create_search_space(input_shape=(116,),
                        output_shape=(116,),
                        *args, **kwargs):

    arch = AutoKArchitecture(input_shape, output_shape, regression=True)
    source = prev_input = arch.input_nodes[0]

    # look over skip connections within a range of 3 nodes
    anchor_points = collections.deque([source], maxlen=3)

    num_layers = 10

    for _ in range(num_layers):
        vnode = VariableNode()
        add_dense_to_(vnode)

        arch.connect(prev_input, vnode)

        # * Cell output
        cell_output = vnode

        cmerge = ConstantNode()
        cmerge.set_op(AddByProjecting(arch, [cell_output], activation='relu'))

        for anchor in anchor_points:
            skipco = VariableNode()
            skipco.add_op(Tensor([]))
            skipco.add_op(Connect(arch, anchor))
            arch.connect(skipco, cmerge)

        # ! for next iter
        prev_input = cmerge
        anchor_points.append(prev_input)


    return arch


def test_create_architecture():
    """'n_parameters': 7349
    """
    from random import random, seed
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    # seed(10)
    architecture = create_search_space()

    ops = [random() for _ in range(architecture.num_nodes)]

    print('num ops: ', len(ops))
    print(ops)
    print('size: ', architecture.size)
    architecture.set_ops(ops)
    architecture.draw_graphviz('architecture_baseline.dot')

    model = architecture.create_model()
    print('depth: ', architecture.depth)
    plot_model(model, to_file='model_baseline.png', show_shapes=True)
    print('n_parameters: ', model.count_params())


if __name__ == '__main__':
    test_create_architecture()
