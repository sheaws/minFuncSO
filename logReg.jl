# Logistic regression objective function evaluations and gradient evaluations
include("misc.jl")

# z = Xw
function logisticObjLinear(z,y;k=nothing)
    if k == nothing
        f = sum(log.(1 .+ exp.(-y.*z)))
    else
        f = sum(log.(1 .+ exp.(-y.*(z.+k))))
    end
    return f
end

#  for gradient wrt iterates (outer loop):
#   z = Xw --> g(w) = X^T * logisticGradLinear
#  for gradient wrt stepsizes (inner loop): 
#   z = XDt, k = Xw --> g(t) = (XD)^T * logisticGrad
function logisticGradLinear(z,y;k=nothing)
    if k == nothing
        yz = y .* z
    else
        yz = y .* (z+k)
    end
    logGrad = -y./(1 .+ exp.(yz))
    
    return logGrad
end

# function value and gradient for minFuncNonOpt to minimize f wrt t
#  (called by minFuncNonOpt)
function logisticObjAndGrad(t,D,w,X,y)
    w_new = w + D*t
    z = y.*(X*w_new)
	f = sum(log.(1 .+ exp.(-z)))
	g = -(X*D)'*(y./(1 .+ exp.(z)))
	return (f,g)
end

# function value and gradient for minFuncNonOpt to minimize f wrt w
#  (called by minFuncNonOpt)
function logisticObjAndGrad(w,X,y)
	yXw = y.*(X*w)
	f = sum(log.(1 .+ exp.(-yXw)))
	g = -X'*(y./(1 .+ exp.(yXw)))
	return (f,g)
end

# function value and zeros for gradient for minFuncNonOpt
# (called by lsArmijoNonOpt)
function logisticObjAndNoGrad(w,X,y)
    (m,n) = size(X)
	yXw = y.*(X*w)
	f = sum(log.(1 .+ exp.(-yXw)))
	g = zeros(m,1)
	return (f,g)
end