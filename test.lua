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
}

-- Run tests:
tester:add(tests):run()
