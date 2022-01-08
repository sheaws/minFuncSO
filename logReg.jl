# Logistic regression
# notation:
#  X = features matrix, w = solution, y = labels {-1,+1}
#  f = objective, g = gradient, f' = the (non-linear) part of the gradient
#  nMM = number of matrix-vector multiplications, z = Xw, k = constants
include("misc.jl")

# objective evaluation for linear composition problems
function logisticObjLinear(z,w,y;k=nothing)
    nMM = 0
    if k == nothing
        f = sum(log.(1 .+ exp.(-y.*z)))
    else
        f = sum(log.(1 .+ exp.(-y.*(z.+k))))
    end
    return (f,nMM)
end

# f' evaluation for linear composition problems
# arguments for gradient wrt iterates (outer loop):
#  z = Xw --> g(w) = X^T * logisticGradLinear
# arguments for gradient wrt stepsizes (inner loop): 
#  z = XDt, k = Xw --> g(t) = (XD)^T * logisticGrad
function logisticFPrimeLinear(z,w,y;k=nothing)
    nMM = 0
    if k == nothing
        yz = y .* z
    else
        yz = y .* (z+k)
    end
    fPrime = -y./(1 .+ exp.(yz))
    return (fPrime,nothing,nMM)
end

# gradient evaluation for linear composition problems
#  note: w is not used
function logisticGradLinear(z,w,X,y;k=nothing)
    fPrime,_,nMM = logisticFPrimeLinear(z,w,y,k=k)
    g = X'*fPrime; nMM += 1
    return (g,nMM)
end

# objective and gradient evaluations for minFuncNonOpt to minimize f wrt t
function logisticObjAndGrad(t,D,w,X,y)
    nMM = 0
    w_new = w + D*t
    z = y.*(X*w_new); nMM += 1
	f = sum(log.(1 .+ exp.(-z))) 
	g = -(X*D)'*(y./(1 .+ exp.(z))); nMM += 1
	return (f,g,nMM)
end

# objective and gradient evaluations for minFuncNonOpt to minimize f wrt w
function logisticObjAndGrad(w,X,y)
    nMM = 0
	yXw = y.*(X*w); nMM += 1
	f = sum(log.(1 .+ exp.(-yXw))) 
	g = -X'*(y./(1 .+ exp.(yXw))); nMM += 1
	return (f,g,nMM)
end

# function value and zeros for gradient for minFuncNonOpt
# (called by lsArmijoNonOpt)
function logisticObjAndNoGrad(w,X,y)
    nMM = 0
    (m,n) = size(X)
	yXw = y.*(X*w); nMM += 1
	f = sum(log.(1 .+ exp.(-yXw)))
	g = zeros(m,1)
	return (f,g,nMM)
end