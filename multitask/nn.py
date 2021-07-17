import taichi as ti
import pickle as pkl
import math

from multitask.utils import scalar, vec, mat
# from config import learning_rate, adam_a, adam_b1, adam_b2, dim

@ti.kernel
def compute_TNS(w: ti.template(), s: ti.template()):
    for I in ti.grouped(w):
        s[None] += w.grad[I] ** 2

# @ti.kernel
# def adam_update(w: ti.template(), m: ti.template(), v: ti.template(), iter: ti.i32):
#     for I in ti.grouped(w):
#         m[I] = adam_b1 * m[I] + (1 - adam_b1) * w.grad[I]
#         v[I] = adam_b2 * v[I] + (1 - adam_b2) * w.grad[I] * w.grad[I]
#         m_cap = m[I] / (1 - adam_b1 ** (iter + 1))
#         v_cap = v[I] / (1 - adam_b2 ** (iter + 1))
#         w[I] -= (adam_a * m_cap) / (ti.sqrt(v_cap) + 1e-8)
#
# @ti.kernel
# def sgd_update(w: ti.template()):
#     for I in ti.grouped(w):
#         w[I] -= w.grad[I] * learning_rate


@ti.data_oriented
class Model:

    def weights_allocate(self):
        self.weights1 = scalar()
        self.bias1 = scalar()

        self.weights2 = scalar()
        self.bias2 = scalar()

        ti.root.dense(ti.ij, (self.n_hidden, self.n_input)).place(self.weights1)
        ti.root.dense(ti.i, self.n_hidden).place(self.bias1)
        ti.root.dense(ti.ij, (self.n_output, self.n_hidden)).place(self.weights2)
        ti.root.dense(ti.i, self.n_output).place(self.bias2)

        self.weights = [self.weights1, self.weights2, self.bias1, self.bias2]

    def adam_weights_allocate(self):
        self.m_weights1, self.v_weights1 = scalar(), scalar()
        self.m_bias1, self.v_bias1 = scalar(), scalar()
        self.m_weights2, self.v_weights2 = scalar(), scalar()
        self.m_bias2, self.v_bias2 = scalar(), scalar()

        ti.root.dense(ti.ij, (self.n_hidden, self.n_input)).place(self.m_weights1, self.v_weights1)
        ti.root.dense(ti.i, self.n_hidden).place(self.m_bias1, self.v_bias1)
        ti.root.dense(ti.ij, (self.n_output, self.n_hidden)).place(self.m_weights2, self.v_weights2)
        ti.root.dense(ti.i, self.n_output).place(self.m_bias2, self.v_bias2)

        self.adam_weights = [self.m_weights1, self.v_weights1, self.m_bias1, self.v_bias1, \
                             self.m_weights2, self.v_weights2, self.m_bias2, self.v_bias2]

    @ti.kernel
    def weights_init(self):
        q1 = math.sqrt(6 / self.n_input)
        for i, j in ti.ndrange(self.n_hidden, self.n_input):
            self.weights1[i, j] = (ti.random() * 2 - 1) * q1

        q2 = math.sqrt(6 / self.n_hidden)
        for i, j in ti.ndrange(self.n_output, self.n_hidden):
            self.weights2[i, j] = (ti.random() * 2 - 1) * q2
        '''
        for i, j in ti.ndrange(n_hidden, n_input):
            weights1[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_input)) * 2

        for i, j in ti.ndrange(n_springs, n_hidden):
            # TODO: n_springs should be n_actuators
            weights2[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_springs)) * 2
        '''

    def __init__(self, config, steps, batch_size, n_input, n_output, \
                 input, output, n_hidden = 64, method = "adam"):

        self.adam_b1 = config.get_config()["nn"]["adam_b1"]
        self.adam_b2 = config.get_config()["nn"]["adam_b2"]
        self.learning_rate = config.get_config()["nn"]["learning_rate"]
        self.dim = config.get_config()["robot"]["dim"]
        self.steps = steps
        self.batch_size = batch_size
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.method = method

        self.input = input

        self.TNS = scalar()
        
        self.hidden = scalar()
        self.hidden_act = scalar()
        self.output = scalar()
        self.output_act = output

        ti.root.place(self.TNS)

        ti.root.dense(ti.ijk, (self.steps, self.batch_size, self.n_hidden)).place(self.hidden, self.hidden_act)
        ti.root.dense(ti.ijk, (self.steps, self.batch_size, self.n_output)).place(self.output)

        self.weights_allocate()
        if self.method == "adam":
            self.adam_weights_allocate()

    def clear_adam(self):
        if self.adam_weights is not None:
            for w in self.adam_weights:
                w.fill(0)

    @ti.kernel
    def clear(self):
        for I in ti.grouped(self.hidden):
            self.hidden[I] = 0.
        for I in ti.grouped(self.output):
            self.output[I] = 0.

    @ti.kernel
    def clear_single(self, t: ti.i32):
        for k, i in ti.ndrange(self.batch_size, self.n_hidden):
            self.hidden[t, k, i] = 0.
        for k, i in ti.ndrange(self.batch_size, self.n_output):
            self.output[t, k, i] = 0.

    @ti.kernel
    def nn1(self, t: ti.i32):
        if ti.static(self.dim == 2):
            for k, i, j in ti.ndrange(self.batch_size, self.n_hidden, self.n_input):
                self.hidden[t, k, i] += self.weights1[i, j] * self.input[t, k, j]
        else:
            for k, i, j in ti.ndrange(self.batch_size, self.n_hidden, self.n_input):
                self.hidden[t, k, i] += self.weights1[i, j] * self.input[t, k, j] * 30.
        for k, i in ti.ndrange(self.batch_size, self.n_hidden):
            self.hidden_act[t, k, i] = ti.sin(self.hidden[t, k, i] + self.bias1[i])

    @ti.kernel
    def nn2(self, t: ti.i32):
        for k, i, j in ti.ndrange(self.batch_size, self.n_output, self.n_hidden):
            self.output[t, k, i] += self.weights2[i, j] * self.hidden_act[t, k, j]
        for k, i in ti.ndrange(self.batch_size, self.n_output):
            self.output_act[t, k, i] = ti.sin(self.output[t, k, i] + self.bias2[i])

    def forward(self, t):
        self.nn1(t)
        self.nn2(t)

    def dump_weights(self, name = "save.pkl"):
        w_val = []
        for w in self.weights:
            w_val.append(w.to_numpy())
        pkl.dump(w_val, open(name, "wb"))

    def load_weights(self, name = "save.pkl"):
        w_val = pkl.load(open(name, 'rb'))
        for w, val in zip(self.weights, w_val):
            w.from_numpy(val)

    def gradient_update(self, iter = 0):
        if self.method == "adam":
            self.adam_update(self.weights1, self.m_weights1, self.v_weights1, iter)
            self.adam_update(self.bias1, self.m_bias1, self.v_bias1, iter)
            self.adam_update(self.weights2, self.m_weights2, self.v_weights2, iter)
            self.adam_update(self.bias2, self.m_bias2, self.v_bias2, iter)
        else:
            for w in self.weights:
                self.sgd_update(w)

    def get_TNS(self):
        self.TNS[None] = 0.
        for w in self.weights:
            self.compute_TNS(w, self.TNS)
        return self.TNS[None]

    @ti.kernel
    def compute_TNS(self, w: ti.template(), s: ti.template()):
        for I in ti.grouped(w):
            s[None] += w.grad[I] ** 2

    @ti.kernel
    def adam_update(self, w: ti.template(), m: ti.template(), v: ti.template(), iter: ti.i32):
        for I in ti.grouped(w):
            m[I] = self.adam_b1 * m[I] + (1 - self.adam_b1) * w.grad[I]
            v[I] = self.adam_b2 * v[I] + (1 - self.adam_b2) * w.grad[I] * w.grad[I]
            m_cap = m[I] / (1 - self.adam_b1 ** (iter + 1))
            v_cap = v[I] / (1 - self.adam_b2 ** (iter + 1))
            w[I] -= (self.learning_rate * m_cap) / (ti.sqrt(v_cap) + 1e-8)

    @ti.kernel
    def sgd_update(self, w: ti.template()):
        for I in ti.grouped(w):
            w[I] -= w.grad[I] * self.learning_rate
