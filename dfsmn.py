import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from collections import OrderedDict

class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

def compute_memory_block(inputs, stride, memory_weight, l_memory_size, r_memory_size):
  """
  :param inputs: 3D, [batch, length, d_hidden]
  :param stride:
  :param memory_weight:
  :param l_memory_size:
  :param r_memory_size:
  :return:
  """
  memory_size = l_memory_size + r_memory_size + 1
  for i in range(memory_size):
    l_pad = max((l_memory_size - i) * stride, 0)
    l_index = max((i - l_memory_size) * stride, 0)
    r_pad = max((i - l_memory_size) * stride, 0)
    r_index = min((i - l_memory_size) * stride, 0)

    if r_index != 0:
      pad_inputs = F.pad(inputs[:, l_index:r_index, :], (0, 0, l_pad, r_pad, 0, 0))
    else:
      pad_inputs = F.pad(inputs[:, l_index:, :], (0, 0, l_pad, r_pad, 0, 0))

    #print(pad_inputs.shape, memory_weight[i, :].shape, i, l_index, l_pad, r_index, r_pad)
    if i == 0:
      p_hatt = torch.einsum('bld,d->bld', pad_inputs, memory_weight[i, :])
    else:
      p_hatt += torch.einsum('bld,d->bld', pad_inputs, memory_weight[i, :])

  return p_hatt

class cfsmn_cell(nn.Module):
  def __init__(self, name, input_size, output_size, hidden_size, l_memory_size, r_memory_size, stride):
    super(cfsmn_cell, self).__init__()
    self.__name__ = name
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.l_memory_size = l_memory_size
    self.r_memory_size = r_memory_size
    self.stride = stride
    self.memory_size = l_memory_size + r_memory_size + 1

    self.wx_v = nn.Linear(input_size, hidden_size, bias=False)
    self.wx_u = nn.Sequential(
        SequenceWise(nn.BatchNorm1d(hidden_size)),
        nn.Linear(hidden_size, output_size),
        nn.ReLU(output_size)
    )
    #self.memory_weights = torch.ones([self.memory_size, self.hidden_size], dtype=torch.float32, requires_grad=True)
    self.memory_weights = Parameter(torch.Tensor(self.memory_size, self.hidden_size))
    nn.init.xavier_uniform_(self.memory_weights)


  def forward(self, inputs):
    # liner transformer
    p = self.wx_v(inputs)
    p = p.transpose(0, 1) # NxTxH
    # memory compute v2
    p_hatt = compute_memory_block(inputs=p, stride=self.stride, memory_weight=self.memory_weights,
                    l_memory_size=self.l_memory_size, r_memory_size=self.r_memory_size)

    # liner transformer
    p_hatt = p + p_hatt
    h = self.wx_u(p_hatt)

    return [h, p_hatt]

class dfsmn_cell(nn.Module):
  def __init__(self, name, input_size, output_size, hidden_size, l_memory_size, r_memory_size, stride):
    super(dfsmn_cell, self).__init__()
    self.__name__ = name
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.l_memory_size = l_memory_size
    self.r_memory_size = r_memory_size
    self.stride = stride
    self.memory_size = l_memory_size + r_memory_size + 1
    self.wx_v = nn.Linear(input_size, hidden_size, bias=False)
    self.wx_u = nn.Sequential(
        SequenceWise( nn.BatchNorm1d(hidden_size) ),
        nn.Linear(hidden_size, output_size),
        nn.ReLU(output_size)
    )
    self.memory_weights = Parameter(torch.Tensor(self.memory_size, self.hidden_size))
    nn.init.xavier_uniform_(self.memory_weights)


  def forward(self, inputs, last_p_hatt):
    # liner transformer
    p = self.wx_v(inputs)

    # memory compute v2
    p_hatt = compute_memory_block(inputs=p, stride=self.stride, memory_weight=self.memory_weights,
                    l_memory_size=self.l_memory_size, r_memory_size=self.r_memory_size)

    # liner transformer
    p_hatt = last_p_hatt + p + p_hatt
    h = self.wx_u(p_hatt)

    return [h, p_hatt]


class DFSMN(nn.Module):
  def __init__(self, feat_input_size, num_classes, n_dfsmn_layers, fsmn_input_dim,
      fsmn_projection_dim, fsmn_output_dim,
             l_memory_size, r_memory_size, stride):
    super(DFSMN, self).__init__()
    self.version = '0.0.1'
    self.feat_input_size = feat_input_size
    self.num_classes = num_classes
    self.n_dfsmn_layers = n_dfsmn_layers
    self.fsmn_input_dim = fsmn_input_dim
    self.fsmn_projection_dim = fsmn_projection_dim
    self.fsmn_output_dim = fsmn_output_dim
    self.l_mem = l_memory_size
    self.r_mem = r_memory_size
    self.stride = stride

    self.fc1 = nn.Sequential(
        nn.Linear(feat_input_size, fsmn_input_dim),
        nn.ReLU(fsmn_input_dim)
    )

    dfsmn_cells = []
    for i in range(self.n_dfsmn_layers):
      if i < 1:
        cell = cfsmn_cell('cfsmn_cell', self.fsmn_input_dim, self.fsmn_output_dim, self.fsmn_projection_dim,
                        self.l_mem, self.r_mem, self.stride)
      else:
        cell = dfsmn_cell('dfsmn_cell', self.fsmn_input_dim, self.fsmn_output_dim, self.fsmn_projection_dim,
                        self.l_mem, self.r_mem, self.stride)
      dfsmn_cells.append(('%d' % (i), cell))

    self.dfsmn_cells = nn.Sequential(OrderedDict(dfsmn_cells))
    fully_connected = nn.Sequential(
      nn.BatchNorm1d(fsmn_output_dim),
      nn.Linear(fsmn_output_dim, num_classes, bias=False)
    )
    self.fc2 = nn.Sequential(
      SequenceWise(fully_connected),
    )
    self.inference_softmax = InferenceBatchSoftmax()

  def forward(self, x, lengths):
    lengths = lengths.cpu().int()
    output_lengths = lengths

    sizes = x.size()
    x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension, NxHxT
    x = x.transpose(1, 2).transpose(0,1).contiguous()  # TxNxH
    x = self.fc1(x)
    outputs = x

    for i in range(self.n_dfsmn_layers):
      if i == 0:
        outputs, p_hatt = self.dfsmn_cells[i](outputs)
      else:
        outputs, p_hatt = self.dfsmn_cells[i](outputs, p_hatt)
    x = self.fc2(outputs) # NxTxH
    x = self.inference_softmax(x)
    return x, output_lengths

  @staticmethod
  def get_param_size(model):
    params = 0
    for p in model.parameters():
      tmp = 1
      for x in p.size():
        tmp *= x
      params += tmp
    return params

  @classmethod
  def load_model(cls, path):
    package = torch.load(path, map_location=lambda storage, loc: storage)
    model = cls(feat_input_size=package['feat_input_size'],
                num_classes=package['num_classes'],
                n_dfsmn_layers=package['n_dfsmn_layers'],
                fsmn_input_dim=package['fsmn_input_dim'],
                fsmn_projection_dim=package['fsmn_projection_dim'],
                fsmn_output_dim=package['fsmn_output_dim'],
                l_memory_size=package['l_memory_size'],
                r_memory_size=package['r_memory_size'],
                stride=package['stride']
        )
    model.load_state_dict(package['state_dict'])
    return model

  @classmethod
  def load_model_package(cls, package):
    model = cls(feat_input_size=package['feat_input_size'],
                num_classes=package['num_classes'],
                n_dfsmn_layers=package['n_dfsmn_layers'],
                fsmn_input_dim=package['fsmn_input_dim'],
                fsmn_projection_dim=package['fsmn_projection_dim'],
                fsmn_output_dim=package['fsmn_output_dim'],
                l_memory_size=package['l_memory_size'],
                r_memory_size=package['r_memory_size'],
                stride=package['stride']
        )
    model.load_state_dict(package['state_dict'])
    return model

  @staticmethod
  def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
    cer_results=None, wer_results=None, avg_loss=None, meta=None):
    package = {
        'version': model.version,
        'feat_input_size': model.feat_input_size,
        'num_classes': model.num_classes,
        'n_dfsmn_layers': model.n_dfsmn_layers,
        'fsmn_input_dim': model.fsmn_input_dim,
        'fsmn_projection_dim': model.fsmn_projection_dim,
        'fsmn_output_dim': model.fsmn_output_dim,
        'l_memory_size': model.l_mem,
        'r_memory_size': model.r_mem,
        'stride': model.stride,
        'state_dict': model.state_dict()
    }

    if optimizer is not None:
      package['optim_dict'] = optimizer.state_dict()
    if avg_loss is not None:
      package['avg_loss'] = avg_loss
    if epoch is not None:
      package['epoch'] = epoch + 1  # increment for readability
    if iteration is not None:
      package['iteration'] = iteration
    if loss_results is not None:
      package['loss_results'] = loss_results
      package['cer_results'] = cer_results
      package['wer_results'] = wer_results
    if meta is not None:
      package['meta'] = meta
    return package

if __name__ == '__main__':
  feat_input_size = 560
  num_classes = 2155
  n_dfsmn_layers = 5
  fsmn_input_dim = 1024
  fsmn_projection_dim = 256
  fsmn_output_dim = 1024
  l_memory_size = 10
  r_memory_size = 1
  stride = 1

  model = DFSMN(feat_input_size, num_classes, n_dfsmn_layers, fsmn_input_dim,
                fsmn_projection_dim, fsmn_output_dim, l_memory_size, r_memory_size, stride)
  #torch.save(DFSMN.serialize(model), './debug.pth')
  print(model)
  print(DFSMN.get_param_size(model))
  size = [16, 1, 560, 10]
  x = torch.ones(size)
  lengths = torch.tensor([187, 143, 136, 125, 115, 110, 110, 102,  98,  93,  93,  91,  78,  69, 64,  60])
  print(x.shape)
  y, output_lengths = model(x, lengths)
  print(y, output_lengths)
  print("work done!")
