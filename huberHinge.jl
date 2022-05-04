# Hinge loss 
# notation:
#  X = features matrix, w = solution, y = labels {-1,+1}
#  f = objective, g = gradient, f' = the (non-linear) part of the gradient
#  nMM = number of matrix-vector multiplications, z = Xw, k = constants
include("misc.jl")

huberEps = 1e-2 

function hingeFunc(y,z,k=nothing)
    if k == nothing
        hinge = max.(0, 1 .- y.*z)
    else
        hinge = max.(0, 1 .- y.*(z.+k))
    end
    return hinge    
end

function huberFunc(losses)
    return [ abs(x) <= huberEps ? 0.5*x*x : huberEps*(abs(x)-huberEps/2) for x in losses ]
end

function huberFPrimeFunc(losses)
    return [ abs(x) <= huberEps ? x : (x > 0 ? huberEps : -huberEps) for x in losses ]
end

# objective evaluation for linear composition problems
function objLinear(z,w,y;k=nothing)
    nMM = 0
    hinge = hingeFunc(y,z,k)
    hh = huberFunc(hinge)
    f = sum(hh)
    return (f,nMM)
end

# f' evaluation for linear composition problems
# arguments for gradient wrt iterates (outer loop):
#  z = Xw --> g(w) = X^T * fPrime
# arguments for gradient wrt stepsizes (inner loop): 
#  z = XDt, k = Xw --> g(t) = (XD)^T * fPrime
function fPrimeLinear(z,w,y;k=nothing)
    nMM = 0
    hinge = hingeFunc(y,z,k)
    fPrime = -y .* huberFPrimeFunc(hinge) 
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
    fPrime,_,nMM = fPrimeLinear(z,w,y,k=k)
    fPrimePrime = fPrime .* fPrime'
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
    z = X*w_new; nMM += 1
    hinge = hingeFunc(y,z,nothing)
    hh = huberFunc(hinge)
    f = sum(hh)
	fPrime = -1 .* huberFPrimeFunc(hinge)
	g = (X*D)'*fPrime; nMM += 1 
	return (f,g,nMM)
end

# objective and gradient evaluations for minFuncNonOpt to minimize f wrt w
function objAndGrad(w,X,y)
    nMM = 0
	z = X*w; nMM += 1
    hinge = hingeFunc(y,z,nothing)
    hh = huberFunc(hinge)
    f = sum(hh)
	fPrime = -1 .* huberFPrimeFunc(hinge)
	g = X'*fPrime; nMM += 1 
	return (f,g,nMM)
end

# function value and zeros for gradient for minFuncNonOpt
# (called by lsArmijoNonOpt)
function objAndNoGrad(w,X,y)
    nMM = 0
    (m,n) = size(X)
	z = X*w; nMM += 1
    hinge = hingeFunc(y,z,nothing)
    hh = huberFunc(hinge)
    f = sum(hh)
	g = zeros(m,1)
	return (f,g,nMM)
end