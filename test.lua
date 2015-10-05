-- Tester:
local _ = require 'moses'
local totem = require 'totem'
local nnfunc = require 'nnfunc'
local tester = totem.Tester()

-- List of tests:
local tests = {
   Tanh = function()
      -- Simple module:
      local layer = nnfunc.nn.Tanh()

      -- Fprop:
      local res = layer({input = torch.randn(10)})

      -- Asserts:
      tester:asserteq(_.count(res), 1, 'expected 1 value in response')
      tester:asserteq(torch.typename(res.output), 'torch.DoubleTensor', 'incorrect return type')
      tester:asserteq(res.output:dim(), 1, 'incorrect nb of dims')
      tester:asserteq(res.output:size(1), 10, 'incorrect size')

      -- Bprop:
      local res = layer({input = torch.randn(10), gradOutput = torch.randn(10)})

      -- Asserts:
      tester:asserteq(_.count(res), 1, 'expected 1 value in response')
      tester:asserteq(torch.typename(res.gradInput), 'torch.DoubleTensor', 'incorrect return type')
      tester:asserteq(res.gradInput:dim(), 1, 'incorrect nb of dims')
      tester:asserteq(res.gradInput:size(1), 10, 'incorrect size')
   end,

   Linear = function()
      -- Simple module:
      local layer = nnfunc.nn.Linear(10,100)

      -- Fprop:
      local res = layer({
         input = torch.randn(10),
         weight = torch.randn(100,10), bias = torch.randn(100)
      })

      -- Asserts:
      tester:asserteq(_.count(res), 1, 'expected 1 value in response')
      tester:asserteq(torch.typename(res.output), 'torch.DoubleTensor', 'incorrect return type')
      tester:asserteq(res.output:dim(), 1, 'incorrect nb of dims')
      tester:asserteq(res.output:size(1), 100, 'incorrect size')

      -- Bprop:
      local res = layer({
         input = torch.randn(10),
         weight = torch.randn(100,10), bias = torch.randn(100),
         gradOutput = torch.randn(100),
      })

      -- Asserts:
      tester:asserteq(_.count(res), 3, 'expected 3 values in response')
      tester:asserteq(torch.typename(res.gradInput), 'torch.DoubleTensor', 'incorrect return type')
      tester:asserteq(res.gradInput:dim(), 1, 'incorrect nb of dims')
      tester:asserteq(res.gradInput:size(1), 10, 'incorrect gi size')
      tester:asserteq(res.gradWeight:size(1), 100, 'incorrect gw size')
      tester:asserteq(res.gradWeight:size(2), 10, 'incorrect gb size')
      tester:asserteq(res.gradBias:size(1), 100, 'incorrect gb size')
   end,

   SpatialConvolution = function()
      -- Simple module:
      local layer = nnfunc.nn.SpatialConvolution(4,16,5,5)

      -- Fprop:
      local res = layer({
         input = torch.randn(4,10,10),
         weight = torch.randn(16,4,5,5), bias = torch.randn(16)
      })

      -- Asserts:
      tester:asserteq(_.count(res), 1, 'expected 1 value in response')
      tester:asserteq(torch.typename(res.output), 'torch.DoubleTensor', 'incorrect return type')
      tester:asserteq(res.output:dim(), 3, 'incorrect nb of dims')
      tester:asserteq(res.output:size(1), 16, 'incorrect size')
      tester:asserteq(res.output:size(2), 6, 'incorrect size')
      tester:asserteq(res.output:size(3), 6, 'incorrect size')

      -- Bprop:
      local res = layer({
         input = torch.randn(4,10,10),
         weight = torch.randn(16,4,5,5), bias = torch.randn(16),
         gradOutput = torch.randn(16,6,6),
      })

      -- Asserts:
      tester:asserteq(_.count(res), 3, 'expected 3 values in response')
      tester:asserteq(torch.typename(res.gradInput), 'torch.DoubleTensor', 'incorrect return type')
      tester:asserteq(res.gradInput:dim(), 3, 'incorrect nb of dims')
      tester:asserteq(res.gradInput:size(1), 4, 'incorrect gi size')
      tester:asserteq(res.gradInput:size(2), 10, 'incorrect gi size')
      tester:asserteq(res.gradInput:size(3), 10, 'incorrect gi size')
      tester:asserteq(res.gradWeight:size(1), 16, 'incorrect gw size')
      tester:asserteq(res.gradWeight:size(2), 4, 'incorrect gb size')
      tester:asserteq(res.gradWeight:size(3), 5, 'incorrect gb size')
      tester:asserteq(res.gradWeight:size(4), 5, 'incorrect gb size')
      tester:asserteq(res.gradBias:size(1), 16, 'incorrect gb size')
   end,

   SeparateGrads = function()
      -- Simple module:
      local layer,gradLayer = nnfunc.nn.Tanh()

      -- Given a layer, one can also lookup its associated gradients like this:
      tester:asserteq(nnfunc.gradsOf[layer], gradLayer, 'incorrect grads lookup')

      -- Fprop:
      local res = layer({input = torch.randn(10)})

      -- Asserts:
      tester:asserteq(_.count(res), 1, 'expected 1 value in response')
      tester:asserteq(torch.typename(res.output), 'torch.DoubleTensor', 'incorrect return type')
      tester:asserteq(res.output:dim(), 1, 'incorrect nb of dims')
      tester:asserteq(res.output:size(1), 10, 'incorrect size')

      -- Bprop:
      local res = gradLayer({input = torch.randn(10), gradOutput = torch.randn(10)})

      -- Asserts:
      tester:asserteq(_.count(res), 1, 'expected 1 value in response')
      tester:asserteq(torch.typename(res.gradInput), 'torch.DoubleTensor', 'incorrect return type')
      tester:asserteq(res.gradInput:dim(), 1, 'incorrect nb of dims')
      tester:asserteq(res.gradInput:size(1), 10, 'incorrect size')
   end,

   ProvideOutput = function()
      -- Simple module:
      local layer = nnfunc.nn.Linear(10,100)

      -- Bprop:
      local res = layer({
         input = torch.randn(10),
         output = torch.Tensor(100),
         weight = torch.randn(100,10), bias = torch.randn(100),
      })

      -- Asserts:
      tester:asserteq(_.count(res), 1, 'expected 1 value in response')
      tester:asserteq(torch.typename(res.output), 'torch.DoubleTensor', 'incorrect return type')
      tester:asserteq(res.output:dim(), 1, 'incorrect nb of dims')
      tester:asserteq(res.output:size(1), 100, 'incorrect size')
   end,

   ProvideGrads = function()
      -- Simple module:
      local layer,gradLayer = nnfunc.nn.Linear(10,100)

      -- Bprop:
      local res = gradLayer({
         input = torch.randn(10),
         weight = torch.randn(100,10), bias = torch.randn(100),
         gradWeight = torch.zeros(100,10), bias = torch.zeros(100),
         gradInput = torch.zeros(10),
         gradOutput = torch.randn(100),
      })

      -- Asserts:
      tester:asserteq(_.count(res), 3, 'expected 3 values in response')
      tester:asserteq(torch.typename(res.gradInput), 'torch.DoubleTensor', 'incorrect return type')
      tester:asserteq(res.gradInput:dim(), 1, 'incorrect nb of dims')
      tester:asserteq(res.gradInput:size(1), 10, 'incorrect gi size')
      tester:asserteq(res.gradWeight:size(1), 100, 'incorrect gw size')
      tester:asserteq(res.gradWeight:size(2), 10, 'incorrect gb size')
      tester:asserteq(res.gradBias:size(1), 100, 'incorrect gb size')
   end,

   CheckLinearGrads = function()
      -- Groundtruth states:
      local layer = nn.Linear(10,100)
      local gradOutput = torch.randn(100)
      local input = torch.randn(10)
      layer:zeroGradParameters()
      local output = layer:forward(input)
      local gradInput = layer:backward(input,gradOutput)

      -- Simple module:
      local layerTest = nnfunc.nn.Linear(10,100)

      -- Fprop:
      local res = layerTest({
         input = input,
         weight = layer.weight, bias = layer.bias,
      })

      -- Asserts:
      tester:asserteq((res.output - output):abs():max(), 0, 'incorrect output state')

      -- Bprop:
      local res = layerTest({
         input = input,
         weight = layer.weight, bias = layer.bias,
         gradOutput = gradOutput,
      })

      -- Asserts:
      tester:asserteq((res.gradInput - gradInput):abs():max(), 0, 'incorrect gradInput state')
      tester:asserteq((res.gradWeight - layer.gradWeight):abs():max(), 0, 'incorrect gradWeight state')
      tester:asserteq((res.gradBias - layer.gradBias):abs():max(), 0, 'incorrect gradBias state')

      -- Repeat tests above, but now user provides all tensors:
      local layerTest2 = nnfunc.nn.Linear(10,100)

      -- Fprop:
      local userOutput = torch.Tensor(100)
      layerTest2({
         input = input,
         weight = layer.weight, bias = layer.bias,
         output = userOutput
      })

      -- Asserts:
      tester:asserteq((userOutput - output):abs():max(), 0, 'incorrect output state')

      -- Bprop:
      local userGradInput = torch.Tensor(10)
      local userGradWeight = torch.Tensor(100,10)
      local userGradBias = torch.Tensor(100)
      layerTest2({
         input = input,
         weight = layer.weight, bias = layer.bias,
         gradOutput = gradOutput,
         gradInput = userGradInput,
         gradWeight = userGradWeight, gradBias = userGradBias,
      })

      -- Asserts:
      tester:asserteq((userGradInput - gradInput):abs():max(), 0, 'incorrect gradInput state')
      tester:asserteq((userGradWeight - layer.gradWeight):abs():max(), 0, 'incorrect gradWeight state')
      tester:asserteq((userGradBias - layer.gradBias):abs():max(), 0, 'incorrect gradBias state')
   end,

   Float = function()
      -- Simple module:
      local layer = nnfunc.nn.Linear(10,100)

      -- Fprop:
      local res = layer({
         input = torch.randn(10):float(),
         weight = torch.randn(100,10):float(), bias = torch.randn(100):float()
      })
      local res = layer({
         input = torch.randn(10):float(),
         weight = torch.randn(100,10):float(), bias = torch.randn(100):float()
      })

      -- Asserts:
      tester:asserteq(torch.typename(res.output), 'torch.FloatTensor', 'incorrect return type')

      -- Bprop:
      local res = layer({
         input = torch.randn(10):float(),
         weight = torch.randn(100,10):float(), bias = torch.randn(100):float(),
         gradOutput = torch.randn(100):float(),
      })

      -- Asserts:
      tester:asserteq(torch.typename(res.gradInput), 'torch.FloatTensor', 'incorrect return type')
      tester:asserteq(torch.typename(res.gradWeight), 'torch.FloatTensor', 'incorrect return type')
      tester:asserteq(torch.typename(res.gradBias), 'torch.FloatTensor', 'incorrect return type')
   end,

}

-- Run tests:
tester:add(tests):run()
