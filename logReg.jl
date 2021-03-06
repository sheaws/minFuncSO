# Logistic regression
# notation:
#  X = features matrix, w = solution, y = labels {-1,+1}
#  f = objective, g = gradient, f' = the (non-linear) part of the gradient
#  nMM = number of matrix-vector multiplications, z = Xw, k = constants
include("misc.jl")

# objective evaluation for linear composition problems
#  note: w is not used for logistic regression but framework
#   provides w for cases where it is used for regularization, etc.
function objLinear(z,w,y;k=nothing)
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
#  z = Xw --> g(w) = X^T * fPrime
# arguments for gradient wrt stepsizes (inner loop): 
#  z = XDt, k = Xw --> g(t) = (XD)^T * fPrime
#  note: w is not used for logistic regression but framework
#   provides w for cases where it is used for regularization, etc.
function fPrimeLinear(z,w,y;k=nothing,epsilon=1e-12)
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
function gradLinear(z,w,X,y;k=nothing)
    fPrime,_,nMM = fPrimeLinear(z,w,y,k=k)
    g = X'*fPrime; nMM += 1
    return (g,nMM)
end

#f'' evaluation for linear composition problems
function fDoublePrimeLinear(z,w,X,y;k=nothing)
    #fPrime,_,nMM = logisticFPrimeLinear(z,w,y,k=k)
    nMM = 0

    if k==nothing
        expZ=exp.(-y.*z)
    else
        expZ=exp.(-y.*(z+k))
    end
    sumExpZ = sum(expZ)
    mult = 1/ sumExpZ

    fPrimePrime = mult .* diagm(vec(expZ)) + mult^2 .* expZ.*expZ'; nMM += 2
    return (fPrimePrime,nMM)
end

function hessianLinear(z,w,X,y;k=nothing)
    fDoublePrime,nMM = fDoublePrimeLinear(z,w,X,y,k=k)
    H = X'*fDoublePrime*X; nMM += 2
    return (H,nMM)
end

# objective and gradient evaluations for minFuncNonOpt to minimize f wrt t
function objAndGrad(t,D,w,X,y)
    nMM = 0
    w_new = w + D*t
    z = y.*(X*w_new); nMM += 1
	f = sum(log.(1 .+ exp.(-z))) 
	g = -(X*D)'*(y./(1 .+ exp.(z))); nMM += 1
	return (f,g,nMM)
end

# objective and gradient evaluations for minFuncNonOpt to minimize f wrt w
function objAndGrad(w,X,y)
    nMM = 0
	yXw = y.*(X*w); nMM += 1
	f = sum(log.(1 .+ exp.(-yXw))) 
	g = -X'*(y./(1 .+ exp.(yXw))); nMM += 1
	return (f,g,nMM)
end

# function value and zeros for gradient for minFuncNonOpt
# (called by lsArmijoNonOpt)
function objAndNoGrad(w,X,y)
    nMM = 0
    (m,n) = size(X)
	yXw = y.*(X*w); nMM += 1
	f = sum(log.(1 .+ exp.(-yXw)))
	g = zeros(m,1)
	return (f,g,nMM)
end