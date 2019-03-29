#!/usr/bin/env python
"""Chainer example: train a VAE on MNIST
"""
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
import numpy as np

import net


def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--opt',default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mc', type=int, default=1)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Prepare VAE model, defined in net.py
    model = net.VAE(784, args.dimz, 500)

    if args.gpu >= 0:

        chainer.cuda.get_device_from_id(device_id=args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    if args.opt == 'adam' :

        optimizer = chainer.optimizers.Adam(alpha=args.lr)

        # optimizer = chainer.optimizers.Adam()

    else :

        optimizer = chainer.optimizers.AdaGrad(lr=args.lr)

    optimizer.setup(model)

    # # Initialize
    # if args.initmodel:
    #     chainer.serializers.load_npz(args.initmodel, model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(withlabel=False)
    train_label, test_label = chainer.datasets.get_mnist(withlabel=True)

    print('mnist: ', train.shape)

    # if args.test:
    #     train, _ = chainer.datasets.split_dataset(train, 100)
    #     test, _ = chainer.datasets.split_dataset(test, 100)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    train_iter_for_k = chainer.iterators.SerialIterator(train_label, args.batchsize,
                                                 repeat=False, shuffle=False)

    test_iter_for_k = chainer.iterators.SerialIterator(test_label, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up an updater. StandardUpdater can explicitly specify a loss function
    # used in the training with 'loss_func' option
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer,
        device=args.gpu, loss_func=model.get_loss_func(k=args.mc))

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu,
                                        eval_func=model.get_loss_func(k=10)), name='val')
    trainer.extend(net.VAE_Evaluator(train_iter_for_k, model,
                 device=args.gpu, access_name = 'train_acc', k=10), trigger=(20, 'epoch'))
    trainer.extend(net.VAE_Evaluator(test_iter_for_k, model,
                                     device=args.gpu, access_name='test_acc', k=10), trigger=(20, 'epoch'))
    # trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'), trigger=(20, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss',
         'main/rec_loss', 'main/kl', 'elapsed_time', 'val/main/loss', 'val/main/rec_loss', 'val/main/kl',
         'train_acc/acc', 'train_acc/ari', 'train_acc/nmi', 'test_acc/acc', 'test_acc/ari', 'test_acc/nmi']))
    trainer.extend(extensions.PlotReport(['main/loss', 'main/rec_loss', 'main/kl', 'val/main/loss', 'val/main/rec_loss',
                                          'val/main/kl'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['train_acc/acc', 'train_acc/ari', 'train_acc/nmi', 'test_acc/acc', 'test_acc/ari', 'test_acc/nmi'], x_key='epoch',
        file_name='accuracy.png'))
    trainer.extend(net.sample_generate(model,args.out),trigger=(2, 'epoch'))
    trainer.extend(extensions.ProgressBar())

    # if args.resume:
    #     chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    # Visualize the results
    def save_images(x, filename):
        import matplotlib.pyplot as plt
        plt.gray()
        fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
        for ai, xi in zip(ax.flatten(), x):
            ai.imshow(xi.reshape(28, 28))
        fig.savefig(filename)

    model.to_cpu()
    train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
    x = chainer.Variable(np.asarray(train[train_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x)
    save_images(x.array, os.path.join(args.out, 'train'))
    save_images(x1.array, os.path.join(args.out, 'train_reconstructed'))

    test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
    x = chainer.Variable(np.asarray(test[test_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x)
    save_images(x.array, os.path.join(args.out, 'test'))
    save_images(x1.array, os.path.join(args.out, 'test_reconstructed'))

    # draw images from randomly sampled z
    z = chainer.Variable(
        np.random.normal(0, 1, (9, args.dimz)).astype(np.float32))
    x = model.decode(z)
    save_images(x.array, os.path.join(args.out, 'sampled'))


if __name__ == '__main__':
    main()
