# import numpy as np
# import matplotlib.pyplot as plt
#
#
# class ReluActivator(object):
#     def forward(self, weighted_input):
#         # return weighted_input
#         return max(0, weighted_input)
#
#     def backward(self, output):
#         return 1 if output > 0 else 0
#
#
# class IdentityActivator(object):
#     def forward(self, weighted_input):
#         return weighted_input
#
#     def backward(self, output):
#         return 1
#
#
# class SigmoidActivator(object):
#     def forward(self, weighted_input):
#         return 1.0 / (1.0 + np.exp(-weighted_input))
#
#     def backward(self, output):
#         return output * (1 - output)
#
#
# class TanhActivator(object):
#     def forward(self, weighted_input):
#         return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0
#
#     def backward(self, output):
#         return 1 - output * output
#
#
# def element_wise_op(array, op):
#     for i in np.nditer(array,
#                        op_flags=['readwrite']):
#         i[...] = op(i)
#
#
# class LstmLayer(object):
#     def __init__(self, input_width, state_width,
#                  learning_rate):
#         self.input_width = input_width
#         self.state_width = state_width
#         self.learning_rate = learning_rate
#         # 门的激活函数
#         self.gate_activator = SigmoidActivator()
#         # 输出的激活函数
#         self.output_activator = TanhActivator()
#         # 当前时刻初始化为t0
#         self.times = 0
#         # 各个时刻的单元状态向量c
#         self.c_list = self.init_state_vec()
#         # 各个时刻的输出向量h
#         self.h_list = self.init_state_vec()
#         # 各个时刻的遗忘门f
#         self.f_list = self.init_state_vec()
#         # 各个时刻的输入门i
#         self.i_list = self.init_state_vec()
#         # 各个时刻的输出门o
#         self.o_list = self.init_state_vec()
#         # 各个时刻的即时状态c~
#         self.ct_list = self.init_state_vec()
#         # 遗忘门权重矩阵Wfh, Wfx, 偏置项bf
#         self.Wfh, self.Wfx, self.bf = (
#             self.init_weight_mat())
#         # 输入门权重矩阵Wfh, Wfx, 偏置项bf
#         self.Wih, self.Wix, self.bi = (
#             self.init_weight_mat())
#         # 输出门权重矩阵Wfh, Wfx, 偏置项bf
#         self.Woh, self.Wox, self.bo = (
#             self.init_weight_mat())
#         # 单元状态权重矩阵Wfh, Wfx, 偏置项bf
#         self.Wch, self.Wcx, self.bc = (
#             self.init_weight_mat())
#
#     def init_state_vec(self):
#         '''
#         初始化保存状态的向量
#         '''
#         state_vec_list = []
#         state_vec_list.append(np.zeros(
#             (self.state_width, 1)))
#         return state_vec_list
#
#     def init_weight_mat(self):
#         '''
#         初始化权重矩阵
#         '''
#         Wh = np.random.uniform(-1e-4, 1e-4,
#                                (self.state_width, self.state_width))
#         Wx = np.random.uniform(-1e-4, 1e-4,
#                                (self.state_width, self.input_width))
#         b = np.zeros((self.state_width, 1))
#         return Wh, Wx, b
#
#     def forward(self, x):
#         '''
#         根据式1-式6进行前向计算
#         '''
#         self.times += 1
#         # 遗忘门
#         fg = self.calc_gate(x, self.Wfx, self.Wfh,
#                             self.bf, self.gate_activator)
#         self.f_list.append(fg)
#         # 输入门
#         ig = self.calc_gate(x, self.Wix, self.Wih,
#                             self.bi, self.gate_activator)
#         self.i_list.append(ig)
#         # 输出门
#         og = self.calc_gate(x, self.Wox, self.Woh,
#                             self.bo, self.gate_activator)
#         self.o_list.append(og)
#         # 即时状态
#         ct = self.calc_gate(x, self.Wcx, self.Wch,
#                             self.bc, self.output_activator)
#         self.ct_list.append(ct)
#         # 单元状态
#         c = fg * self.c_list[self.times - 1] + ig * ct
#         self.c_list.append(c)
#         # 输出
#         h = og * self.output_activator.forward(c)
#         self.h_list.append(h)
#
#     def calc_gate(self, x, Wx, Wh, b, activator):
#         '''
#         计算门
#         '''
#         h = self.h_list[self.times - 1]  # 上次的LSTM输出
#         net = np.dot(Wh, h) + np.dot(Wx, x) + b
#         gate = activator.forward(net)
#         return gate
#
#     def backward(self, x, delta_h, activator):
#         '''
#         实现LSTM训练算法
#         '''
#         self.calc_delta(delta_h, activator)
#         self.calc_gradient(x)
#
#     def update(self):
#         '''
#         按照梯度下降，更新权重
#         '''
#         self.Wfh -= self.learning_rate * self.Whf_grad
#         self.Wfx -= self.learning_rate * self.Whx_grad
#         self.bf -= self.learning_rate * self.bf_grad
#         self.Wih -= self.learning_rate * self.Whi_grad
#         self.Wix -= self.learning_rate * self.Whi_grad
#         self.bi -= self.learning_rate * self.bi_grad
#         self.Woh -= self.learning_rate * self.Wof_grad
#         self.Wox -= self.learning_rate * self.Wox_grad
#         self.bo -= self.learning_rate * self.bo_grad
#         self.Wch -= self.learning_rate * self.Wcf_grad
#         self.Wcx -= self.learning_rate * self.Wcx_grad
#         self.bc -= self.learning_rate * self.bc_grad
#
#     def calc_delta(self, delta_h, activator):
#         # 初始化各个时刻的误差项
#         self.delta_h_list = self.init_delta()  # 输出误差项
#         self.delta_o_list = self.init_delta()  # 输出门误差项
#         self.delta_i_list = self.init_delta()  # 输入门误差项
#         self.delta_f_list = self.init_delta()  # 遗忘门误差项
#         self.delta_ct_list = self.init_delta()  # 即时输出误差项
#
#         # 保存从上一层传递下来的当前时刻的误差项
#         self.delta_h_list[-1] = delta_h
#
#         # 迭代计算每个时刻的误差项
#         for k in range(self.times, 0, -1):
#             self.calc_delta_k(k)
#
#     def init_delta(self):
#         '''
#         初始化误差项
#         '''
#         delta_list = []
#         for i in range(self.times + 1):
#             delta_list.append(np.zeros(
#                 (self.state_width, 1)))
#         return delta_list
#
#     def calc_delta_k(self, k):
#         '''
#         根据k时刻的delta_h，计算k时刻的delta_f、
#         delta_i、delta_o、delta_ct，以及k-1时刻的delta_h
#         '''
#         # 获得k时刻前向计算的值
#         ig = self.i_list[k]
#         og = self.o_list[k]
#         fg = self.f_list[k]
#         ct = self.ct_list[k]
#         c = self.c_list[k]
#         c_prev = self.c_list[k - 1]
#         tanh_c = self.output_activator.forward(c)
#         delta_k = self.delta_h_list[k]
#
#         # 根据式9计算delta_o
#         delta_o = (delta_k * tanh_c *
#                    self.gate_activator.backward(og))
#         delta_f = (delta_k * og *
#                    (1 - tanh_c * tanh_c) * c_prev *
#                    self.gate_activator.backward(fg))
#         delta_i = (delta_k * og *
#                    (1 - tanh_c * tanh_c) * ct *
#                    self.gate_activator.backward(ig))
#         delta_ct = (delta_k * og *
#                     (1 - tanh_c * tanh_c) * ig *
#                     self.output_activator.backward(ct))
#         delta_h_prev = (
#                 np.dot(delta_o.transpose(), self.Woh) +
#                 np.dot(delta_i.transpose(), self.Wih) +
#                 np.dot(delta_f.transpose(), self.Wfh) +
#                 np.dot(delta_ct.transpose(), self.Wch)
#         ).transpose()
#
#         # 保存全部delta值
#         self.delta_h_list[k - 1] = delta_h_prev
#         self.delta_f_list[k] = delta_f
#         self.delta_i_list[k] = delta_i
#         self.delta_o_list[k] = delta_o
#         self.delta_ct_list[k] = delta_ct
#
#     def calc_gradient(self, x):
#         # 初始化遗忘门权重梯度矩阵和偏置项
#         self.Wfh_grad, self.Wfx_grad, self.bf_grad = (
#             self.init_weight_gradient_mat())
#         # 初始化输入门权重梯度矩阵和偏置项
#         self.Wih_grad, self.Wix_grad, self.bi_grad = (
#             self.init_weight_gradient_mat())
#         # 初始化输出门权重梯度矩阵和偏置项
#         self.Woh_grad, self.Wox_grad, self.bo_grad = (
#             self.init_weight_gradient_mat())
#         # 初始化单元状态权重梯度矩阵和偏置项
#         self.Wch_grad, self.Wcx_grad, self.bc_grad = (
#             self.init_weight_gradient_mat())
#
#         # 计算对上一次输出h的权重梯度
#         for t in range(self.times, 0, -1):
#             # 计算各个时刻的梯度
#             (Wfh_grad, bf_grad,
#              Wih_grad, bi_grad,
#              Woh_grad, bo_grad,
#              Wch_grad, bc_grad) = (
#                 self.calc_gradient_t(t))
#             # 实际梯度是各时刻梯度之和
#             self.Wfh_grad += Wfh_grad
#             self.bf_grad += bf_grad
#             self.Wih_grad += Wih_grad
#             self.bi_grad += bi_grad
#             self.Woh_grad += Woh_grad
#             self.bo_grad += bo_grad
#             self.Wch_grad += Wch_grad
#             self.bc_grad += bc_grad
#
#         # 计算对本次输入x的权重梯度
#         xt = x.transpose()
#         self.Wfx_grad = np.dot(self.delta_f_list[-1], xt)
#         self.Wix_grad = np.dot(self.delta_i_list[-1], xt)
#         self.Wox_grad = np.dot(self.delta_o_list[-1], xt)
#         self.Wcx_grad = np.dot(self.delta_ct_list[-1], xt)
#
#     def init_weight_gradient_mat(self):
#         '''
#         初始化权重矩阵
#         '''
#         Wh_grad = np.zeros((self.state_width,
#                             self.state_width))
#         Wx_grad = np.zeros((self.state_width,
#                             self.input_width))
#         b_grad = np.zeros((self.state_width, 1))
#         return Wh_grad, Wx_grad, b_grad
#
#     def calc_gradient_t(self, t):
#         '''
#         计算每个时刻t权重的梯度
#         '''
#         h_prev = self.h_list[t - 1].transpose()
#         Wfh_grad = np.dot(self.delta_f_list[t], h_prev)
#         bf_grad = self.delta_f_list[t]
#         Wih_grad = np.dot(self.delta_i_list[t], h_prev)
#         bi_grad = self.delta_f_list[t]
#         Woh_grad = np.dot(self.delta_o_list[t], h_prev)
#         bo_grad = self.delta_f_list[t]
#         Wch_grad = np.dot(self.delta_ct_list[t], h_prev)
#         bc_grad = self.delta_ct_list[t]
#         return Wfh_grad, bf_grad, Wih_grad, bi_grad, \
#                Woh_grad, bo_grad, Wch_grad, bc_grad
#
#     def reset_state(self):
#         # 当前时刻初始化为t0
#         self.times = 0
#         # 各个时刻的单元状态向量c
#         self.c_list = self.init_state_vec()
#         # 各个时刻的输出向量h
#         self.h_list = self.init_state_vec()
#         # 各个时刻的遗忘门f
#         self.f_list = self.init_state_vec()
#         # 各个时刻的输入门i
#         self.i_list = self.init_state_vec()
#         # 各个时刻的输出门o
#         self.o_list = self.init_state_vec()
#         # 各个时刻的即时状态c~
#         self.ct_list = self.init_state_vec()
#
#
# def data_set():
#     x = [np.array([[1], [2], [3]]),
#          np.array([[2], [3], [4]])]
#     d = np.array([[1], [2]])
#     return x, d
#
#
# def test():
#     l = LstmLayer(3, 2, 1e-3)
#     x, d = data_set()
#     l.forward(x[0])
#     l.forward(x[1])
#     l.backward(x[1], d, IdentityActivator())
#     return l
# import torch
# import torch.nn as nn
# import math
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
#
# print('Using PyTorch version:', torch.__version__, ' Device:', device)
#
#
# class mylstm(nn.Module):
#     def __init__(self,input_size,hidden_size):
#         super(mylstm, self).__init__()
#         self.input_size=input_size
#         self.hidden_size=hidden_size
#         # 输入门：声明可进行梯度计算的参数矩阵
#         self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
#         self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_i = nn.Parameter(torch.Tensor(hidden_size))
#         # 遗忘门
#         self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
#         self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_f = nn.Parameter(torch.Tensor(hidden_size))
#         # 啥东西？
#         self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
#         self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_c = nn.Parameter(torch.Tensor(hidden_size))
#         # 输出门
#         # o_t
#         self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
#         self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_o = nn.Parameter(torch.Tensor(hidden_size))
#
#         self.init_weights()
#
#     def init_weights(self):
#         # 参数的初始化
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)
#
#     def forward(self,x,init_state=None):
#         batch_size, seq_size= x.size()
#         hidden_seq = []
#         if init_state is None:
#             h_t, c_t = (
#                 torch.zeros(batch_size, self.hidden_size).to(device),
#                 torch.zeros(batch_size, self.hidden_size).to(device)
#             )
#         else:
#             h_t, c_t = init_state
#
#         for t in range(seq_size):
#             x_t = x[:, t, :]
#             i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
#             f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
#             g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
#             o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
#             c_t = f_t * c_t + i_t * g_t
#             h_t = o_t * torch.tanh(c_t)
#
#             hidden_seq.append(h_t.unsqueeze(0))
#         hidden_seq = torch.cat(hidden_seq, dim=0)
#         hidden_seq = hidden_seq.transpose(0, 1).contiguous()
#         return hidden_seq, (h_t, c_t)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmoidLayer(nn.Module):
    def __init__(self,input_size,output_size):
        super(SigmoidLayer, self).__init__()
        self.fc=nn.Linear(input_size,output_size)

    def forward(self,input):
        output=self.fc(input)
        return torch.sigmoid(output)

class TanhLayer(nn.Module):
    def __init__(self,input_size,output_size):
        super(TanhLayer, self).__init__()
        self.fc=nn.Linear(input_size,output_size)

    def forward(self,input):
        output=self.fc(input)
        return torch.tanh(output)

class mylstm(nn.Module):
    def __init__(self):
        super(mylstm, self).__init__()
        self.input_size=57
        self.hidden_size=128
        self.output_size=18

        self.f_gate=SigmoidLayer(self.input_size+self.hidden_size,self.hidden_size)

        self.i_gate_1=SigmoidLayer(self.input_size+self.hidden_size,self.hidden_size)
        self.i_gate_2=TanhLayer(self.input_size+self.hidden_size,self.hidden_size)

        self.o_gate=SigmoidLayer(self.input_size+self.hidden_size,self.hidden_size)

        self.classifier=nn.Linear(self.hidden_size,self.output_size)


    def forward(self,input,hidden,cell):
        # 遗忘门的处理
        before_gate=torch.cat((input,hidden),dim=1)
        f_activate=self.f_gate(before_gate)
        cell_next=f_activate*cell

        # 记忆门
        i_t=self.i_gate_1(before_gate)
        C_t=self.i_gate_2(before_gate)
        cell_next+=i_t*C_t

        # 输出门
        o_t=self.o_gate(before_gate)
        h_t=o_t*torch.tanh(cell_next)

        output=self.classifier(h_t)
        output=F.log_softmax(output,dim=1)
        return output,h_t,o_t


if __name__=='__main__':
    net=mylstm()
    input=torch.zeros(1,57)
    hidden=torch.zeros(1,128)
    cell=torch.zeros(1,128)

    output,hidden,cell=net(input,hidden,cell)
    print(output.size())

