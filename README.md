# minFuncSO

minFuncSO is a Julia package for unconstained optimization of differentiable real-valued multivariate functions. It is based on a Mark Schmidt's minFunc Matlab code. MinFuncSO, however, is optimized for linear composition problems and allows for subspace optimization.

julia ./example_minFuncSO.jl --help

usage: example_minFuncSO.jl [--ds DS] [--linReg] [--linRegLHalf]
                        [--logReg] [--logRegL2] [--hh] [-h]

optional arguments:
  --ds DS        a comma-separated list of datasets to run (default:"") 
  --linReg       set objective to linear regression 
  --linRegLHalf  set objective to linear regression using a p=1/2 norm 
  --logReg       set objective to logistic regression 
  --logRegL2     set objective to logistic regression with L2 regularization 
  --hh           set objective to huberized hinge 
  -h, --help     show this help message and exit 
