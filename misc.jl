# Miscellaneous supporting functions

# Numerical gradient for minFuncSO. Call with optimized loss function eval.
function numGrad(objFunc,w,X,Xw;k=nothing)
    (m,n) = size(X)
    delta = 2*sqrt(1e-12)*(1+norm(w))
    Xwplus = Xw
    if k!= nothing
        Xwplus = Xwplus + k
    end
	g = zeros(n,1)
	for i in 1:n
    	pertCol = delta*X[:,i]
    	fxp = objFunc(Xwplus + pertCol)
    	fxm = objFunc(Xwplus - pertCol)
		g[i,1] = (fxp - fxm)/(2*delta)
	end
	return g
end

# Numerical gradient for minFuncNonOpt. Call with non-optimized loss+gradient
#  function that does not calculate gradient
function numGrad(funObj,w,X)
    (m,n) = size(X)
    delta = 2*sqrt(1e-12)*(1+norm(w))
	g = zeros(n,1)
	e_i = zeros(n)
	for i in 1:n
    	e_i[i] = 1
		(fxp,) = funObj(w + delta*e_i,X)
		(fxm,) = funObj(w - delta*e_i,X)
		g[i,1] = (fxp - fxm)/(2*delta)
		e_i[i] = 0
	end
	return g
end

# Numerical gradient for minFuncSO (forward diff)
function forwardDiffGrad(objFunc,w,X,fXw)
    (m,n) = size(X)
    h= 2*sqrt(1e-12)*(1+norm(w))
	g_fd = zeros(n,1)
	for i in 1:n
    	hXe_i=h*X[:,i]
    	fhXe_i=objFunc(hXe_i)
		g_fd[i,1] = (fhXe_i - fXw)/h
	end
	return g_fd
end

# Check if number is a real-finite number
function isfinitereal(x)
	return (imag(x) == 0) & (!isnan(x)) & (!isinf(x))
end

# Updates memory for minFuncSO
function updateDiffs(i,lBfgsSize,g_prev,g,w_prev,w,X,DiffIterates,DiffGrads,
 XDiffIterates;normalizeColumns=false,calcXDiffIterates=false)
    j = mod(i,lBfgsSize)
    if j==0
        j=lBfgsSize
    end
    DiffIterates[j] = w.-w_prev
    DiffGrads[j] = g.-g_prev
    
    if normalizeColumns
        colNorm = norm(DiffIterates[j],2)
        if colNorm > 1e-4
            DiffIterates[j] = DiffIterates[j]./colNorm
        end
    end
    
    if calcXDiffIterates
        XDiffIterates[j] = X*DiffIterates[j]
    end
end

# updates memory for minFuncNonOpt
#=
function updateDiffs(i,lBfgsSize,g_prev,g,w_prev,w,DiffIterates,DiffGrads)
    j = mod(i,lBfgsSize)
    if j==0
        j=lBfgsSize
    end
    DiffIterates[j] = w.-w_prev
    DiffGrads[j] = g.-g_prev
end
=#