package = "nnfunc"
version = "scm-1"

source = {
   url = "git@github.com:clementfarabet/nnfunc.git",
   branch = "master",
}

description = {
   summary = "NN primitives for functional programming.",
   homepage = "https://github.com/clementfarabet/nnfunc",
   license = "MIT",
}

dependencies = {
   "torch >= 7.0",
   "totem",
   "moses",
   "nn",
}

build = {
   type = "builtin",
   modules = {
      ['nnfunc.init'] = 'init.lua',
      ['nnfunc.test'] = 'test.lua',
   },
}
