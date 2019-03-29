

import six
import os
import copy

import numpy as np
from PIL import Image

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L

from chainer import training
from chainer.dataset import convert
from chainer import reporter
import chainer.dataset.iterator as iterator_module
from chainer import function
from chainer import cuda

from sklearn.cluster import KMeans
from sklearn.metrics import cluster

class VAE(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def represent(self, x):

        means, lnvars = self.encode(x)

        z = F.gaussian(means, lnvars)

        return z

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf


class VAE2(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h, n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        h1 = F.tanh(self.le2(h1))
        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h1 = F.tanh(self.ld2(h1))
        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf



class VAE_BN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE_BN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne = L.BatchNormalization(n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.bnd = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.bne(self.le1(x)))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.bnd(self.ld1(z)))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf

class VAE_BN_R(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE_BN_R, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne = L.BatchNormalization(n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.bnd = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.bne(F.tanh(self.le1(x)))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = self.bnd(F.tanh(self.ld1(z)))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf


class VAE_EBN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE_EBN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne = L.BatchNormalization(n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            # self.bnd = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.bne(self.le1(x)))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf


class VAE2_BN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_BN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.bne2 = L.BatchNormalization(n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.bnd1 = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_h)
            self.bnd2 = L.BatchNormalization(n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):

        h1 = self.le1(x)
        h1 = F.tanh(self.bne1(h1))
        h1 = self.le2(h1)
        h1 = F.tanh(self.bne2(h1))
        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)
        h1 = F.tanh(self.bnd1(h1))
        h1 = self.ld2(h1)
        h1 = F.tanh(self.bnd2(h1))

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf

class VAE2_BN_R(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_BN_R, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.bne2 = L.BatchNormalization(n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.bnd1 = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_h)
            self.bnd2 = L.BatchNormalization(n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):

        h1 = self.le1(x)
        h1 = self.bne1(F.tanh(h1))
        h1 = self.le2(h1)
        h1 = self.bne2(F.tanh(h1))
        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)
        h1 = self.bnd1(F.tanh(h1))
        h1 = self.ld2(h1)
        h1 = self.bnd2(F.tanh(h1))

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf


class VAE2_EBN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_EBN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.bne2 = L.BatchNormalization(n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            # self.bnd1 = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_h)
            # self.bnd2 = L.BatchNormalization(n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):

        h1 = self.le1(x)
        h1 = F.tanh(self.bne1(h1))
        h1 = self.le2(h1)
        h1 = F.tanh(self.bne2(h1))
        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)
        h1 = F.tanh(h1)
        # h1 = F.tanh(self.bnd1(h1))
        h1 = self.ld2(h1)
        h1 = F.tanh(h1)
        # h1 = F.tanh(self.bnd2(h1))

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf

class VAE_Evaluator(training.extensions.Evaluator):

    def __init__(self, iterators, model, converter=convert.concat_examples,
                 device=None, access_name = 'train', k=10):

        if isinstance(iterators, iterator_module.Iterator):

            iterators = {'main': iterators}

        self._iterators = iterators

        self._targets = {'VAE': model}

        self.converter = converter

        self.device = device

        self.access_name = access_name + '/'

        self.k = k

    def evaluate(self):

        iterator = self._iterators['main']

        model = self._targets['VAE']

        it = copy.copy(iterator)

        summary = reporter.DictSummary()

        latent_list = []

        teacher_list = []

        observation = {}

        with reporter.report_scope(observation):

            for batch in it:

                batch_train, batch_teacher = self.converter(batch, self.device)

                # ren = self.converter(batch, self.device)
                #
                # print(len(ren))
                #
                # print(type(ren))
                #
                # print(ren)
                #
                # batch_train = 0
                #
                # batch_teacher = 0

                with function.no_backprop_mode():

                    z = model.represent(batch_train)

                    teacher_list.extend(batch_teacher)

                    latent_list.extend(z.data)

            xp = cuda.get_array_module(batch_train)

            latent_list = xp.stack(latent_list)

            teacher_list = xp.stack(teacher_list)

            latent_list = chainer.cuda.to_cpu(latent_list)

            teacher_list = chainer.cuda.to_cpu(teacher_list)

            kmeans_model = KMeans(n_clusters=self.k, random_state=0).fit(latent_list)

            labels = kmeans_model.labels_

            acc = cluster.completeness_score(labels, teacher_list)

            ari = cluster.adjusted_rand_score(labels, teacher_list)

            nmi = cluster.normalized_mutual_info_score(labels, teacher_list)

            observation[self.access_name + 'acc'] = acc

            observation[self.access_name + 'ari'] = ari

            observation[self.access_name + 'nmi'] = nmi

        summary.add(observation)

        return summary.compute_mean()



class VAE_L(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE_L, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            # self.bn1 = L.BatchNormalization(n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.relu(self.le1(x))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.relu(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf


class VAE_LN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE_LN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            # self.bn1 = L.BatchNormalization(n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.relu(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf




class VAE2_L(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_L, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            # self.bn1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h, n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.le1(x)

        h1 = F.relu(h1)

        h1 = self.le2(h1)

        h1 = F.relu(h1)

        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)

        h1 = F.relu(h1)

        h1 = self.ld2(h1)

        h1 = F.relu(h1)

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf


class VAE2_NL(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_NL, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            # self.bn1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h, n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.le1(x)

        h1 = F.relu(h1)

        h1 = self.le2(h1)

        h1 = F.tanh(h1)

        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)

        h1 = F.tanh(h1)

        h1 = self.ld2(h1)

        h1 = F.relu(h1)

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf


class VAE2_NLN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_NLN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            # self.bn1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h, n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.le1(x)

        h1 = F.relu(h1)

        h1 = self.le2(h1)

        h1 = F.tanh(h1)

        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)

        h1 = F.relu(h1)

        h1 = self.ld2(h1)

        h1 = F.relu(h1)

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf



class VAE_L_BN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE_L_BN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne = L.BatchNormalization(n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.bnd = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.relu(self.bne(self.le1(x)))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.relu(self.bnd(self.ld1(z)))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf


class VAE_L_BN_R(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE_L_BN_R, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne = L.BatchNormalization(n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.bnd = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.bne(F.relu(self.le1(x)))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = self.bnd(F.relu(self.ld1(z)))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf



class VAE_LN_BN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE_LN_BN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne = L.BatchNormalization(n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.bnd = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.bne(self.le1(x)))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.relu(self.bnd(self.ld1(z)))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf


class VAE_L_EBN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE_L_EBN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne = L.BatchNormalization(n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            # self.bnd = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.relu(self.bne(self.le1(x)))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.relu(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf

class VAE_LN_EBN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE_LN_EBN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne = L.BatchNormalization(n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            # self.bnd = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.bne(self.le1(x)))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.relu(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf



class VAE2_L_BN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_L_BN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.bne2 = L.BatchNormalization(n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.bnd1 = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_h)
            self.bnd2 = L.BatchNormalization(n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.le1(x)

        h1 = F.relu(self.bne1(h1))

        h1 = self.le2(h1)

        h1 = F.relu(self.bne2(h1))

        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)

        h1 = F.relu(self.bnd1(h1))

        h1 = self.ld2(h1)

        h1 = F.relu(self.bnd2(h1))

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf

class VAE2_L_BN_R(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_L_BN_R, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.bne2 = L.BatchNormalization(n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.bnd1 = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_h)
            self.bnd2 = L.BatchNormalization(n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.le1(x)

        h1 = self.bne1(F.relu(h1))

        h1 = self.le2(h1)

        h1 = self.bne2(F.relu(h1))

        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)

        h1 = self.bnd1(F.relu(h1))

        h1 = self.ld2(h1)

        h1 = self.bnd2(F.relu(h1))

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf



class VAE2_NL_BN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_NL_BN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.bne2 = L.BatchNormalization(n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.bnd1 = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_h)
            self.bnd2 = L.BatchNormalization(n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.le1(x)

        h1 = F.relu(self.bne1(h1))

        h1 = self.le2(h1)

        h1 = F.tanh(self.bne2(h1))

        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)

        h1 = F.tanh(self.bnd1(h1))

        h1 = self.ld2(h1)

        h1 = F.relu(self.bnd2(h1))

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf




class VAE2_NLN_BN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_NLN_BN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.bne2 = L.BatchNormalization(n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.bnd1 = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_h)
            self.bnd2 = L.BatchNormalization(n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.le1(x)

        h1 = F.relu(self.bne1(h1))

        h1 = self.le2(h1)

        h1 = F.tanh(self.bne2(h1))

        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)

        h1 = F.relu(self.bnd1(h1))

        h1 = self.ld2(h1)

        h1 = F.relu(self.bnd2(h1))

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf




class VAE2_L_EBN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_L_EBN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.bne2 = L.BatchNormalization(n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            # self.bnd1 = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_h)
            # self.bnd2 = L.BatchNormalization(n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.le1(x)

        h1 = F.relu(self.bne1(h1))

        h1 = self.le2(h1)

        h1 = F.relu(self.bne2(h1))

        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)

        h1 = F.relu(h1)

        h1 = self.ld2(h1)

        h1 = F.relu(h1)

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf



class VAE2_NL_EBN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_NL_EBN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.bne2 = L.BatchNormalization(n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            # self.bnd1 = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_h)
            # self.bnd2 = L.BatchNormalization(n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.le1(x)

        h1 = F.relu(self.bne1(h1))

        h1 = self.le2(h1)

        h1 = F.tanh(self.bne2(h1))

        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)

        h1 = F.tanh(h1)

        h1 = self.ld2(h1)

        h1 = F.relu(h1)

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf



class VAE2_NLN_EBN(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE2_NLN_EBN, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.bne1 = L.BatchNormalization(n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.bne2 = L.BatchNormalization(n_h)
            self.le3_mu = L.Linear(n_h, n_latent)
            self.le3_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            # self.bnd1 = L.BatchNormalization(n_h)
            self.ld2 = L.Linear(n_h, n_h)
            # self.bnd2 = L.BatchNormalization(n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.n_latent = n_latent

    def forward(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = self.le1(x)

        h1 = F.relu(self.bne1(h1))

        h1 = self.le2(h1)

        h1 = F.tanh(self.bne2(h1))

        mu = self.le3_mu(h1)
        ln_var = self.le3_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):

        h1 = self.ld1(z)

        h1 = F.relu(h1)

        h1 = self.ld2(h1)

        h1 = F.relu(h1)

        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def make_hidden(self, batchsize):

        return np.random.randn(batchsize, self.n_latent).astype(np.float32)



    def get_loss_func(self, beta=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # kl loss
            kl_loss = F.gaussian_kl_divergence(mu, ln_var) / batchsize
            self.kl_loss = kl_loss

            # reconstruction loss
            mu = F.repeat(mu, repeats=k, axis=0)
            ln_var = F.repeat(ln_var, repeats=k, axis=0)
            x = F.repeat(x, repeats=k, axis=0)
            z = F.gaussian(mu, ln_var)
            rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)

            self.rec_loss = rec_loss
            self.loss = self.rec_loss + self.kl_loss
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss, 'kl': self.kl_loss}, observer=self)
            return self.loss

        return lf








def sample_generate(gen, dst, rows=10, cols=10, seed=0):
    """Visualization of rows*cols images randomly generated by the generator."""
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = chainer.Variable(xp.asarray(gen.make_hidden(n_images)))

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen.decode(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 255.0 , 0.0, 255.0), dtype=np.uint8)
        # _, _, h, w = x.shape
        h , w = 28, 28
        x = x.reshape((rows, cols, h, w))
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape((rows * h, cols * w))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x,mode='L').save(preview_path)

    return make_image
