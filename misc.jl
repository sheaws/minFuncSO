# Miscellaneous supporting functions

# Numerical gradient for minFuncSO. Call with optimized loss function eval.
function numGrad(objFunc,w,X,Xw;k=nothing)
    nMM = 0
    (m,n) = size(X)
    delta = 2*sqrt(1e-12)*(1+norm(w))
    Xwplus = Xw
    if k!= nothing
        Xwplus = Xwplus + k
    end
	g = zeros(n,1)
	for i in 1:n
    	pertCol = delta*X[:,i]
    	fxp,nmm = objFunc(Xwplus + pertCol,w); nMM += nmm
    	fxm,nmm = objFunc(Xwplus - pertCol,w); nMM += nmm
		g[i,1] = (fxp - fxm)/(2*delta)
	end
	return (g,nMM)
end

# Numerical gradient for minFuncNonOpt. Call with non-optimized loss+gradient
#  function that does not calculate gradient
function numGrad(funObj,w,X)
    nMM = 0
    (m,n) = size(X)
    delta = 2*sqrt(1e-12)*(1+norm(w))
	g = zeros(n,1)
	e_i = zeros(n)
	for i in 1:n
    	e_i[i] = 1
		(fxp,_,nmm) = funObj(w + delta*e_i,X); nMM += nmm
		(fxm,_,nmm) = funObj(w - delta*e_i,X); nMM += nmm
		g[i,1] = (fxp - fxm)/(2*delta)
		e_i[i] = 0
	end
	return (g,nMM)
end

# Numerical gradient for minFuncSO (forward diff)
function forwardDiffGrad(objFunc,w,X,fXw)
    nMM = 0
    (m,n) = size(X)
    h = 2*sqrt(1e-12)*(1+norm(w))
	g = zeros(n,1)
	for i in 1:n
    	hXe_i = h*X[:,i]
    	fhXe_i,nmm = objFunc(hXe_i,w); nMM += nmm
		g[i,1] = (fhXe_i - fXw)/h
	end
	return (g,nMM)
end

# Check if number is a real-finite number
function isfinitereal(x)
	return (imag(x) == 0) & (!isnan(x)) & (!isinf(x))
end

# Updates memory for minFuncSO
function updateDiffs(i,lBfgsSize,g_prev,g,w_prev,w,X,DiffIterates,DiffGrads,
 XDiffIterates;normalizeColumns=false,calcXDiffIterates=false)
    nMM = 0
    j = mod(i,lBfgsSize)
    if j==0
        j = lBfgsSize
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
        XDiffIterates[j] = X*DiffIterates[j]; nMM += 1
    end
    return nMM
end