# Decent/ search directions for the main iterative method (i.e. outer loop)
#  Algorithms from J. Nocedal and S.J. Wright, Numerical Optimization, 2nd ed.
#  and from M. Schmidt, minFunc (2005)

using Printf, LinearAlgebra
include("misc.jl")

# Raydan 1997 GBB algo
function bb(g,g_prev,w,w_prev,i,alpha)
    eps = 1e-10
    oneOverEps = 1e10
    if i==1
        return (-g,alpha)
    end

    deltaW = w .- w_prev 
    deltaG = g .- g_prev
    oldAlpha = alpha
    
    alpha = oldAlpha*dot(-g_prev,deltaG)/dot(g_prev,g_prev)
            
    # Sanity check on the step-size
	if (!isfinitereal(alpha)) | (alpha < eps) | (alpha > oneOverEps)
		alpha = 1.0
	end

    return (-g*1.0/alpha,alpha)
end

# Non-linear conjugate gradient
function cg(g,g_prev,i,d_prev,optTol)
    d = -g
    if i==1
        return d
    end
    
    # non-linear Hestenes Steifel update
    diffG = (g-g_prev)
    beta = dot(g,diffG)/dot(diffG,d_prev)
    d += beta*d_prev
    
    # check for reset
    if dot(g,d) > -optTol
        d = -g
    end
    
    return d
end

# limited memory BFGS
# could choose H_k^0 to be \gamma_k I where \gamma_k=(s_{k-1}^Ty_{k-1})/(y_{k-1}^Ty_{k-1}) 
function lbfgs(g,i,DiffIterates,DiffGrads)
    if i==1
        return -g
    end    

    lBfgsSize=length(DiffIterates)
    m = min(lBfgsSize,i-1)
    rhos = zeros(m)
    alphas = zeros(m)
    q = g
    for j in 1:m
        k = mod(i - j,m)
        if k < 1
            k = k + m
        end
        rhos[k] = 1 ./ dot(DiffGrads[k]',DiffIterates[k])
        alphas[k] = rhos[k] * dot(DiffIterates[k],q)
        q = q .- alphas[k] * DiffGrads[k]
    end
    
    n = length(g)
    j = mod(i-1,m)
    if j < 1
        j = j + m
    end
    gamma = dot(DiffIterates[j]',DiffGrads[j])/(dot(DiffGrads[j]',DiffGrads[j]))
    r = gamma*q # Hk0 = gamma* I
    
    for j in 1:m
        k = mod(i + j,m)
        if k < 1
            k = k + m
        end
        beta = rhos[k] * dot(DiffGrads[k]',r) 
        r = r + DiffIterates[k]*(alphas[k]-beta)
    end
    
    return -r
end

# Line Search Newton-CG/ truncated Newton
#  i.e., truncated CG for an approximation of Newton directions
# to run version that is not optimized for linear structure, set nonOpt=true and
#  pass in funObj (preferably version with no gradient calculation) to objFunc
#  pass in funObj (with gradient calculation) to gradFunc
function newtonCG(g,maxIter,objFunc,gradFunc,numDiff,w,X;k=nothing,nonOpt=false)
    (m,n) = size(X)
    nOE = 0
    nGE = 0
    nMM = 0
    Xw_s = fill(0.0,m)
    z = zeros(length(g))
    
    normg = norm(g)
    eps = min(0.5,sqrt(normg))*normg # forcing sequence * norm(g)
    r = g
    d = -g
    for j in 1:maxIter
        s = 2*sqrt(1e-12)*(1+norm(w))/norm(d)
        w_s = w + s*d

        if nonOpt
            if numDiff
                g_v = numGrad(objFunc,w_s,X)
                nOE += 2*n
                nMM += 2*n
            else
                _,g_v = gradFunc(w_s,X)
                nOE += 1
                nGE += 1
                nMM += 2
            end
        else
            Xw_s = X*w_s
            nMM += 1
            if numDiff
                g_v = numGrad(objFunc,w_s,X,Xw_s,k=k)
                nOE += 2*n
            else
                g_v = X'*gradFunc(Xw_s,konst=k)
                nMM += 1
                nGE += 1
            end
        end
        
        Bd = (g_v-g)/s 
        dTBd = dot(d',Bd)
        if dTBd <= 1e-16 # B has negative eigenvalues
            if j==1
                return (-g,j,nOE,nGE,nMM)
            else
                return (z,j,nOE,nGE,nMM)
            end
        end
        
        normr = dot(r',r)
        alpha = normr/dTBd
        z = z + alpha*d
        r_new = r+ alpha*Bd
        
        norm2r_new = dot(r_new',r_new)
        if sqrt(norm2r_new) < eps
            return (z,j,nOE,nGE,nMM)
        end
        
        beta = norm2r_new/normr
        d = -r_new + beta*d
        r = r_new
    end
    return (z,maxIter,nOE,nGE,nMM)
end