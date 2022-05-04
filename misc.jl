# Miscellaneous supporting functions
using LinearAlgebra, Printf, SparseArrays

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
 XDiffIterates;XDiffGrads=nothing,normalizeColumns=false,calcXDiffIterates=false,calcXDiffGrads=false)
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

    if calcXDiffGrads && XDiffGrads!=nothing
        XDiffGrads[j] = X*DiffGrads[j]; nMM += 1
    end

    return nMM
end

# 2: multidimension Wolfe
# 3: solver optimized for linear composition problems, momentum as secondary directions
# 4: non-opt solver, momentum as secondary directions
# 5: solver optimized for linear composition problems, TN as secondary directions
# 7: solver optimized for linear composition problems, Adagrad as secondary directions
function lsTypeIsSO(lsType)
    if (2<= lsType && lsType <= 5) || (7<=lsType && lsType<=10)
        return true
    else
        return false
    end
end

function methodHasPrecond(method)
    if method== 3 || method==5
        return true
    end
    return false
end

# returns a string description of iterative method
function methodToLabel(method;abbreviate=false)
    methodLabel=""
    if method==0
        methodLabel="GD"
    elseif method==1
        methodLabel="BB"
    elseif method==2
        methodLabel="CG"
    elseif method==3
        if abbreviate
            methodLabel="lQN"
        else
            methodLabel="lBFGS"
        end
    elseif method==4
        methodLabel="TN"
    elseif method==5
        if abbreviate
            methodLabel="NEW"
        else
            methodLabel="Newton"
        end
    end
    return methodLabel
end

function lsTypeToLabel(lsType)
    lsLabel=""
    if lsType==0
        lsLabel = "Arm"
    elseif lsType==1
        lsLabel = "lsW"
    elseif lsType==2
        lsLabel = "mdW"
    elseif lsType==6
        lsLabel = "Lip"
    elseif lsType==7
        lsLabel = "Adg"
    elseif lsType==8
        lsLabel = "NAG"
    elseif lsType==9
        lsLabel = "GD"
    elseif lsType==10
        lsLabel = "GDm"
    end
    return lsLabel
end

# returns a string description of line/ subspace search 
function lsTypeToLabel(lsType,lsInterp,nMomDirs,ssMethod,ssLS,nonOpt)
    lsLabel=""
    if lsType==0
        lsLabel = "Armijo"
    elseif lsType==1
        lsLabel = "lsWolfe"
    elseif lsType==2
        lsLabel = "mdWolfe"
    elseif lsType==3 || lsType==4
        lsLabel="2"
        if nMomDirs!=0
            lsLabel=string(nMomDirs+1)
        end
        lsLabel=string(lsLabel,"d-Mm")
    elseif lsType==5
        lsLabel = "2d-TN"
    elseif lsType==6
        lsLabel = "Lipschitz"
    elseif lsType==7
        lsLabel="2"
        if nMomDirs!=0
            lsLabel=string(nMomDirs+1)
        end
        lsLabel=string(lsLabel,"d-Ag")
    elseif lsType==8
        lsLabel = "NAG"
    elseif lsType==9
        lsLabel = "2d-GD"
    elseif lsType==10
        lsLabel = "3d-GDm"
    end
    
    if  lsTypeIsSO(lsType) && lsType!=2
        lsLabel=string(lsLabel,"-",methodToLabel(ssMethod,abbreviate=true),"-",
        lsTypeToLabel(ssLS))
    end
    
    if lsType==4 || nonOpt==1
        lsLabel=string(lsLabel,"-")
    end

    if lsInterp==2
        lsLabel=string(lsLabel,"2")
    end
    
    return lsLabel
end

