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


@ti.func
def relu(x):
    return ti.max(x, 0.0)

@ti.func
def sigmoid(x):
    return 1 / (1 + ti.exp(-x))

@ti.func
def gelu(x):
    return 0.5 * x * (1 + ti.tanh(ti.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


@ti.data_oriented
class Model:

    def weights_allocate(self):
        self.weights1 = scalar()
        self.bias1 = scalar()

        self.weights2 = scalar()
        self.bias2 = scalar()

        self.weights1_node.place(self.weights1)
        self.n_hidden_node.place(self.bias1)
        self.weights2_node.place(self.weights2)
        self.n_output_node.place(self.bias2)

        self.weights = [self.weights1, self.weights2, self.bias1, self.bias2]

    def adam_weights_allocate(self):
        self.m_weights1, self.v_weights1 = scalar(), scalar()
        self.m_bias1, self.v_bias1 = scalar(), scalar()
        self.m_weights2, self.v_weights2 = scalar(), scalar()
        self.m_bias2, self.v_bias2 = scalar(), scalar()

        self.weights1_node.place(self.m_weights1, self.v_weights1)
        self.n_hidden_node.place(self.m_bias1, self.v_bias1)
        self.weights2_node.place(self.m_weights2, self.v_weights2)
        self.n_output_node.place(self.m_bias2, self.v_bias2)

        self.adam_weights = [self.m_weights1, self.v_weights1, self.m_bias1, self.v_bias1, \
                             self.m_weights2, self.v_weights2, self.m_bias2, self.v_bias2]

    @ti.kernel
    def weights_init(self):
        q1 = math.sqrt(6 / self.n_input)
        for model_id, i, j in ti.ndrange(self.n_models, self.n_hidden, self.n_input):
            self.weights1[model_id, i, j] = (ti.random() * 2 - 1) * q1

        q2 = math.sqrt(6 / self.n_hidden)
        for model_id, i, j in ti.ndrange(self.n_models, self.n_output, self.n_hidden):
            self.weights2[model_id, i, j] = (ti.random() * 2 - 1) * q2
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
                 input, output, n_models, n_hidden = 64, method = "adam", activation="sin"):

        self.adam_b1 = config.get_config()["nn"]["adam_b1"]
        self.adam_b2 = config.get_config()["nn"]["adam_b2"]
        self.learning_rate = config.get_config()["nn"]["learning_rate"]
        self.dim = config.get_config()["robot"]["dim"]
        self.activation = config.get_config()["nn"]["activation"] if "activation" in config.get_config()["nn"] else activation
        self.activation_output = config.get_config()["nn"]["activation_output"] if "activation_output" in config.get_config()[
            "nn"] else activation

        self.n_models = n_models
        
        self.default_model_id = 0

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

        # array of structs
        self.batch_node = ti.root.dense(ti.i, self.n_models)
        self.n_hidden_node = self.batch_node.dense(ti.j, self.n_hidden)
        self.weights1_node = self.n_hidden_node.dense(ti.k, self.n_input)
        self.n_output_node = self.batch_node.dense(ti.j, self.n_output)
        self.weights2_node = self.n_output_node.dense(ti.k, self.n_hidden)

        ti.root.place(self.TNS)

        self.batch_node.dense(ti.axes(1, 2, 3), (self.steps, self.batch_size, self.n_hidden)).place(self.hidden, self.hidden_act)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.steps, self.batch_size, self.n_output)).place(self.output)

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
        for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
            self.hidden[model_id, t, k, i] = 0.
        for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_output):
            self.output[model_id, t, k, i] = 0.

    @ti.kernel
    def nn1(self, t: ti.i32):
        if ti.static(self.dim == 2):
            for model_id, k, i, j in ti.ndrange(self.n_models, self.batch_size, self.n_hidden, self.n_input):
                self.hidden[model_id, t, k, i] += self.weights1[model_id, i, j] * self.input[model_id, t, k, j]
        else:
            for model_id, k, i, j in ti.ndrange(self.n_models, self.batch_size, self.n_hidden, self.n_input):
                self.hidden[model_id, t, k, i] += self.weights1[model_id, i, j] * self.input[model_id, t, k, j]
        if ti.static(self.activation == "sin"):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.hidden_act[model_id, t, k, i] = ti.sin(self.hidden[model_id, t, k, i] + self.bias1[model_id, i])
        elif ti.static(self.activation == "tanh"):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.hidden_act[model_id, t, k, i] = ti.tanh(self.hidden[model_id, t, k, i] + self.bias1[model_id, i])
        elif ti.static(self.activation == "relu"):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.hidden_act[model_id, t, k, i] = relu(self.hidden[model_id, t, k, i] + self.bias1[model_id, i])
        elif ti.static(self.activation == "sigmoid"):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.hidden_act[model_id, t, k, i] = sigmoid(self.hidden[model_id, t, k, i] + self.bias1[model_id, i])
        elif ti.static(self.activation == "gelu"):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.hidden_act[model_id, t, k, i] = gelu(self.hidden[model_id, t, k, i] + self.bias1[model_id, i])

    @ti.kernel
    def nn2(self, t: ti.i32):
        for model_id, k, i, j in ti.ndrange(self.n_models, self.batch_size, self.n_output, self.n_hidden):
            self.output[model_id, t, k, i] += self.weights2[model_id, i, j] * self.hidden_act[model_id, t, k, j]

        if ti.static(self.activation_output == "sin"):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_output):
                self.output_act[model_id, t, k, i] = ti.sin(self.output[model_id, t, k, i] + self.bias2[model_id, i])
        elif ti.static(self.activation_output == "tanh"):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_output):
                self.output_act[model_id, t, k, i] = ti.tanh(self.output[model_id, t, k, i] + self.bias2[model_id, i])
        elif ti.static(self.activation_output == "relu"):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_output):
                self.output_act[model_id, t, k, i] = relu(self.output[model_id, t, k, i] + self.bias2[model_id, i]) - 1.0
        elif ti.static(self.activation_output == "sigmoid"):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_output):
                self.output_act[model_id, t, k, i] = sigmoid(self.output[model_id, t, k, i] + self.bias2[model_id, i]) * 2 - 1.0
        elif ti.static(self.activation_output == "gelu"):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_output):
                self.output_act[model_id, t, k, i] = gelu(self.output[model_id, t, k, i] + self.bias2[model_id, i]) - 1.0



    def forward(self, t):
        self.nn1(t)
        self.nn2(t)

    def dump_weights(self, name="save.pkl"):
        w_val = []
        for w in self.weights:
            w = w.to_numpy()
            w_val.append(w[self.default_model_id])
        pkl.dump(w_val, open(name, "wb"))
    
    @ti.kernel
    def copy_from_numpy(self, dst: ti.template(), src: ti.ext_arr(), model_id: ti.i32):
        for I in ti.grouped(src):
            dst[model_id, I] = src[I]

    def load_weights(self, name="save.pkl", model_id=0):
        w_val = pkl.load(open(name, 'rb'))
        self.load_weights_from_value(w_val, model_id)

    def load_weights_from_value(self, w_val, model_id=0):
        for w, val in zip(self.weights, w_val):
            if val.shape[0] == 1:
                val = val[0]
            self.copy_from_numpy(w, val, model_id)

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
