# Linear regression
# notation:
#  X = features matrix, w = solution, y = labels {-1,+1}
#  f = objective, g = gradient, f' = the (non-linear) part of the gradient
#  nMM = number of matrix-vector multiplications, z = Xw, k = constants
include("misc.jl")

# objective evaluation for linear composition problems
#  note: w is not used for linear regression but framework
#   provides w for cases where it is used for regularization, etc.
function objLinear(z,w,y;k=nothing)
    nMM = 0
    m = length(z)
    if k == nothing
        f = 0.5*norm(z-y,2)^2
    else
        f = 0.5*norm((z+k)-y,2)^2
    end
    f = f ./ m
    return (f,nMM)
end

# f' evaluation for linear composition problems
# arguments for gradient wrt iterates (outer loop):
#  z = Xw --> g(w) = X^T * fPrime
# arguments for gradient wrt stepsizes (inner loop): 
#  z = XDt, k = Xw --> g(t) = (XD)^T * fPrime
#  note: w is not used for linear regression but framework
#   provides w for cases where it is used for regularization, etc.
function fPrimeLinear(z,w,y;k=nothing,epsilon=1e-12)
    nMM = 0
    m = length(z)
    if k == nothing
        fPrime = z-y
    else
        fPrime = (z+k)-y
    end
    fPrime = fPrime ./ m
    return (fPrime,nothing,nMM)
end

# gradient evaluation for linear composition problems
function gradLinear(z,w,X,y;k=nothing)
    fPrime,_,nMM = fPrimeLinear(z,w,y,k=k)
    g = X'*fPrime; nMM += 1
    return (g,nMM)
end

#f'' evaluation for linear composition problems
function gDoublePrimeLinear(z,w,X,y;k=nothing)
    (m,n) = size(X)
    nMM = 0
    fPrimePrime = Matrix(I(m))
    fPrimePrime ./ m
    return (fPrimePrime,nMM)
end

function hessianLinear(z,w,X,y;k=nothing)
    nMM = 0
    (m,n) = size(X)
    H = X'*X; nMM += m
    return (H,nMM)
end

# objective and gradient evaluations for minFuncNonOpt to minimize f wrt t
function objAndGrad(t,D,w,X,y)
    nMM = 0
    m = length(y)
    w_new = w + D*t
    z = X*w_new; nMM += 1
	f = 0.5/m*norm(z-y,2)^2
	g = -(X*D)'*norm(z-y,2)./m; nMM += 1
	return (f,g,nMM)
end

#objective and gradient evaluations for minFuncNonOpt to minimize f wrt w
function objAndGrad(w,X,y)
    return (0.0,nothing,0)
end

# function value and zeros for gradient for minFuncNonOpt
# (called by lsArmijoNonOpt)
function objAndNoGrad(w,X,y)
    return (0.0,nothing,0)
end