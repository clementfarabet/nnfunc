# nnfunc

Functionalize nn modules: the goal of this package is to make it
easy to develop 3rd-party frameworks, by re-exposing nn modules
as functions. Basically provide a functional API to nn.

Every instantiated module becomes a simple state-less function:
input data and parameters must be provided as inputs to this function;
same thing for gradients. For convenience and efficiency, the state
of the underlying nn module is still relied on for caching (every function
returned by nnfunc is a closure relying on an instantiated nn module).

## API

### Expose packages

Any package that provides `nn.Module` children can be exposed.

```lua
nnfunc.functionalize 'nn'   -- done by default by nnfunc
nnfunc.functionalize 'nnx'  -- bundle new package...
```

Once called, every module in the source package is available to
use as a function; see examples below.

### API #1

A single function that evaluates the module, and automatically
computes gradients if `gradOutput` is provided.

```lua
-- this returns a function that can be used to eval this module and
-- its gradients:
layer = nnfunc.nn.Linear(10,100)

-- compute module's output:
prediction = layer({
   input = torch.randn(10),
   weight = torch.randn(100,10), bias = torch.randn(100),
})
-- prediction looks like this:
-- {
--    output = torch.Tensor(100)
-- }

-- output can be user-provided, optionally:
prediction = layer({
   input = torch.randn(10),
   weight = torch.randn(100,10), bias = torch.randn(100),
   output = torch.Tensor(100),
})
-- output is now valid

-- compute gradients (backprop) - this happens automatically
-- because gradOutput is provided:
grads = layer({
   input = torch.randn(10),
   weight = torch.randn(100,10), bias = torch.randn(100),
   gradOutput = torch.randn(100),
})
-- grads looks like this:
-- {
--    gradInput = torch.Tensor(10),
--    gradWeight = torch.Tensor(100,10),
--    gradBias = torch.Tensor(100),
-- }

-- the user can also provide all the tensors for computed gradients,
-- if her application requires that they be owned externally:
grads = layer({
   input = torch.randn(10),
   weight = torch.randn(100,10), bias = torch.randn(100),
   gradOutput = torch.randn(100),
   gradWeight = torch.zeros(100,10), bias = torch.zeros(100),
   gradInput = torch.zeros(10),
})
-- user-provided gradInput, gradWeight and gradBias are now
-- valid!
```

### API #2

Two separate functions: one for eval, one for gradients. This
can be useful when separate function pointers need to be used
to register gradients.

```lua
-- two separate functions:
layer,gradLayer = nnfunc.nn.Linear(10,100)

-- compute module's output [same as API #1]:
prediction = layer({
   input = torch.randn(10),
   weight = torch.randn(100,10), bias = torch.randn(100),
})

-- compute gradients (backprop) [separate function for grads]:
grads = gradLayer({
   input = torch.randn(10),
   weight = torch.randn(100,10), bias = torch.randn(100),
   gradOutput = torch.randn(100),
})
```

A hash table is also maintained to retrieve gradients associated
to any object created:

```lua
-- two separate functions:
layer,gradLayer = nnfunc.nn.Linear(10,100)

-- gradLayer could be retrieve like this:
gradLayer2 = nnfunc.gradsOf[layer]
assert(gradLayer2 == gradLayer)
```
