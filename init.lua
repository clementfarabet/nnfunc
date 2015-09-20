-- What? This file provides a functionalize utility that
-- turns every nn Module into a simple function.

-- Package
local nnfunc = {}

-- Grads lookup
nnfunc.gradsOf = {}
local gradsOf = nnfunc.gradsOf

-- Functionalize any nn-like package:
function nnfunc.functionalize(mod)
   -- mod is the package name:
   assert(type(mod) == 'string', 'mod should be a string (e.g. nn, ...')

   -- populate hash of methods:
   nnfunc[mod] = {}
   local map = nnfunc[mod]

   -- lookup every module in source package:
   mod = require(mod)
   for k,v in pairs(mod) do
      local mt = getmetatable(v)
      if mt then
         local mmt = getmetatable(mt)
         if mmt then
            map[k] = function(...)
               -- Construct object:
               local o = v(...)

               -- Gradients:
               local g = function(data)
                  -- compute gradients
                  o.weight = data.weight -- weight+bias are always user-provided
                  o.bias = data.bias
                  o.gradWeight = data.gradWeight or o.gradWeight -- grads are optionally user-provided
                  o.gradBias = data.gradBias or o.gradBias
                  o.gradInput = data.gradInput or o.gradInput
                  if o.gradWeight then
                     o.gradWeight:zero()
                  end
                  if o.gradBias then
                     o.gradBias:zero()
                  end
                  local input = data.input
                  local gradOutput = data.gradOutput
                  local gradInput = o:backward(input, gradOutput)
                  assert(not data.gradInput or gradInput == data.gradInput, 'module ['..k..'] replaces self.gradInput; fix it')
                  return {
                     gradInput = gradInput,
                     gradWeight = o.gradWeight,
                     gradBias = o.gradBias,
                  }
               end

               -- Fprop:
               local f = function(data)
                  if data.gradOutput then
                     -- compute gradients
                     return g(data)
                  else
                     -- compute output
                     o.weight = data.weight -- weight+bias are always user-provided
                     o.bias = data.bias
                     o.output = data.output or o.output -- ouput is optionally user-provided
                     local input = data.input
                     local output = o:forward(input)
                     assert(not data.output or output == data.output, 'module ['..k..'] replaces self.output; fix it')
                     return {
                        output = output,
                     }
                  end
               end

               -- Register:
               gradsOf[f] = g

               -- Return both:
               return f,g
            end
         end
      end
   end
end

-- Functinoalize nn by default:
nnfunc.functionalize 'nn'

-- Tests
nnfunc.test = function()
   require('./test')
end

-- Return package:
return nnfunc
