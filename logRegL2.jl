# Logistic regression with L2 regularization
# notation:
#  X = features matrix, w = solution, y = labels {-1,+1}
#  f = objective, g = gradient, f' = the (non-linear) part of the gradient
#  nMM = number of matrix-vector multiplications, z = Xw, k = constants
#  addComp = additive component for f' evaluation for the version optimized
#   for linear composition problems. same dimension as w and the optimizer
#   code adds this to account for any regularization term. see example in 
#   the directional derivative calculation in Wolfe in linesearch.jl 
include("misc.jl")

# objective evaluation for linear composition problems
function logisticL2ObjLinear(z,w,y;k=nothing)
    nMM = 0
    if k == nothing
        f = sum(log.(1 .+ exp.(-y.*z)))
    else
        f = sum(log.(1 .+ exp.(-y.*(z.+k))))
    end
    f += 0.5*norm(w,2)
    return (f,nMM)
end


# f' evaluation for linear composition problems
# arguments for gradient wrt iterates (outer loop):
#  z = Xw --> g(w) = X^T * logisticGradLinear
# arguments for gradient wrt stepsizes (inner loop): 
#  z = XDt, k = Xw --> g(t) = (XD)^T * logisticGrad
function logisticL2FPrimeLinear(z,w,y;k=nothing)
    nMM = 0
    if k == nothing
        yz = y .* z
    else
        yz = y .* (z+k)
    end
    fPrime = -y./(1 .+ exp.(yz))
    addComp = w
    return (fPrime,addComp,nMM)
end

# gradient evaluation for linear composition problems
function logisticL2GradLinear(z,w,X,y;k=nothing)
    fPrime,addComp,nMM = logisticL2FPrimeLinear(z,w,y,k=k)
    g = X'*fPrime .+ addComp; nMM += 1
    return (g,nMM)
end

# objective and gradient evaluations for minFuncNonOpt to minimize f wrt t
function logisticL2ObjAndGrad(t,D,w,X,y)
    nMM = 0
    w_new = w + D*t
    z = y.*(X*w_new); nMM += 1
	f = sum(log.(1 .+ exp.(-z))) 
	g = -(X*D)'*(y./(1 .+ exp.(z))); nMM += 1 #regularization does not affect g here
	return (f,g,nMM)
end

# objective and gradient evaluations for minFuncNonOpt to minimize f wrt w
function logisticL2ObjAndGrad(w,X,y)
    nMM = 0
	yXw = y.*(X*w); nMM += 1
	f = sum(log.(1 .+ exp.(-yXw))) + 0.5*norm(w,2)
	g = -X'*(y./(1 .+ exp.(yXw))) .+ w; nMM += 1
	return (f,g,nMM)
end

# function value and zeros for gradient for minFuncNonOpt
# (called by lsArmijoNonOpt)
function logisticL2ObjAndNoGrad(w,X,y)
    nMM = 0
    (m,n) = size(X)
	yXw = y.*(X*w); nMM += 1
	f = sum(log.(1 .+ exp.(-yXw))) + 0.5*norm(w,2)
	g = zeros(m,1)
	return (f,g,nMM)
end