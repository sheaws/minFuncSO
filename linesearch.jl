# functions for linesearches and subspace searches (given by lsType)
#  and supporting functions (initialization and interpolation)

include("misc.jl")

### get initial step size to try ###
function getLSInitStep(lsInit,i,f,f_prev,gTd;verbose=false)
    t = 1.0
    if i>1
        if lsInit==0 # Newton step
            t = 1.0
        elseif lsInit==1 # Quadratic initialization based on f, g and prev f
            t = 2.0*(f-f_prev)/gTd
            if(t>1.0)
                t = 1.0
            end
        end
    end
    if verbose
        @printf("getLSInitStep returns t=%f: lsInit=%d, i=%d, f=%f, fprev=%f, gTd=%.0f\n",
            t, lsInit, i, f, f_prev, gTd)
    end
    return t    
end

### get and set fref ###
# update the list of the most recent fVals. returns the largest f value and its index in recent memmory
#  (and its index into oldFVals)
function getAndSetLSFref(nFref,oldFvals,i,f_prev)
    j = mod(i,nFref)
    if j== 0
        j = nFref
    end
    oldFvals[j]=f_prev
    return findmax(oldFvals)    
end

# update the list of the largest fVals. returns the largest f value ever encountered and its index
function getAndSetLSFref2(nFref,oldFvals,i,f_prev)
    if i <= nFref
        oldFvals[i] = f_prev
    else
        oldMax,j = findmax(oldFvals)
        if oldMax > f_prev
            oldFvals[j] = f_prev
        end
    end
    return findmax(oldFvals)    
end

### interpolate new step size ###
# f1, t1 and gTD1 are the 1-prev values
function lsInterpolate(lsInterpType,f,f1,t,g,gTd;t1=nothing,gTd1=nothing,mult=0.5,verbose=false)
    oldT = t
    if lsInterpType==0
        t = t*mult
    elseif lsInterpType==1 
        #gg=dot(g,g)
        if(isfinitereal(f))
        	t= t^2*gTd/(2(f - f1 - t*gTd))
        else
            t=lsInterpolate(lsInterpType-1,f,f1,t,g,gTd,mult=mult,verbose=verbose)
        end
    elseif lsInterpType==2
        if gTd1==nothing || t1==nothing
            t=lsInterpolate(lsInterpType-1,f,f1,t,g,gTd,mult=mult,verbose=verbose)
        else
            d1 = gTd1 + gTd - 3*(f1-f)/(t1-t)
            d2 = d1^2 - gTd1*gTd
            if isfinitereal(d2) && d2>0
                d2 = sqrt(d2)
            else
                d2 = 0
            end
            d2 = sign(t-t1)*d2
            t = t - (t-t1) * (gTd+d2-d1)/(gTd-gTd1+2*d2)
        end
    end
    if verbose
        @printf("lsInterpolate (%d): oldT=%f, t=%f, f=%f, f1=%f",lsInterpType,oldT,t,f,f1)
        if gTd==nothing || gTd1==nothing
            @printf("\n")
        else
            @printf(", gTd=%f, gTd1=%f\n",gTd,gTd1)
        end
    end
    return t
end

function mdInterpolate(mdInterpType,t;mult=0.5,g=nothing,f=nothing,f1=nothing)
    if mdInterpType == 0
        t = t .* mult
    elseif mdInterpType == 1
        if f1!=nothing && f!=nothing && g!=nothing
            gg=dot(g,g)
            t = t.^2 .* gg./ (2(f .- f1 .+ t .* gg))
            t /= norm(t)
        else
            t = mdIterpolate(mdIterpType-1,t)
        end
    end
    return t
end

### backtracking Armijo ###
# to run version that is not optimized for linear structure, set nonOpt=true and
#  pass in funObj (preferably version with no gradient calculation) to objFunc
#  Xw_prev and Xd are not used but need to be passed in with the correct dims
# TODO: add numDiff
# TODO: switch between different interpolation methods
function lsArmijo(objFunc,fPrimeFunc,X,Xw_prev,Xd,w_prev,t0,c1,f_prev,fref,g,gTd,d;
 konst=nothing,verbose=false,maxLsIter=25,nonOpt=false)
    lsFailed = false
    nLsIter = 1
    nObjEvals = 0
    nGradEvals = 0
    nMatMult = 0
    t = t0
    t_lsPrev = t
    f = f_prev
    w = w_prev
    Xw = Xw_prev
    gTd_lsPrev = gTd
    gTd_new = gTd

    if verbose
        @printf("lsArmijo: fref=%.4f, f_prev=%.4f, norm(w,2)=%f, norm(w_prev,2)=%f, t0=%f\n",
            fref,f_prev,norm(w,2),norm(w_prev,2),t0)	
    end
    
    c1gTd = c1*gTd
    thresh = fref + t*c1gTd    
    while f > thresh && nLsIter <= maxLsIter # sufficient decrease not yet satisfied
        # calculate f at new point
        w = w_prev + t*d
        f_lsPrev = f

        gTd_lsPrev = gTd_new
        if nonOpt
            f,g_new,nmm = objFunc(w,X); nMatMult += nmm
            gTd_new = dot(g_new,d)
        else
            Xw = Xw_prev + t*Xd
            f,nmm = objFunc(Xw,w,konst=konst); nMatMult += nmm
            fPrime,addComp,nmm = fPrimeFunc(Xw,w); nMatMult += nmm
            gTd_new = dot(fPrime,Xd)
            if addComp!= nothing
                gTd_new += dot(addComp,d)
            end
        end
        nObjEvals += 1
        nGradEvals += 1
        
        thresh = fref + t*c1gTd 
        
        if verbose
            @printf("nLsIter=%d, t=%f, t_lsPrev=%f, f=%f, f_lsPrev=%f, fref=%f, thresh=%f\n",
                nLsIter,t,t_lsPrev,f,f_lsPrev,fref,thresh)
        end

        t = lsInterpolate(0,f,f_lsPrev,t,g,gTd_new) 
        #t = lsInterpolate(1,f,f_lsPrev,t,g,gTd_new)
        #t_new = lsInterpolate(2,f,f_lsPrev,t,g,gTd_new,t1=t_lsPrev,gTd1=gTd_lsPrev)
        #t_lsPrev = t
        #t = t_new

        nLsIter += 1
    end

    if nLsIter > maxLsIter
        lsFailed = true
    end

    if verbose
        @printf("lsArmijo returns t=%f, f=%f, nLsIter=%d, size(w)=%s\n",t,f,nLsIter,size(w))
    end
    return (t,f,w,Xw,lsFailed,nLsIter,nObjEvals,nGradEvals,nMatMult)
end

### strong Wolfe conditions - line search ###
# to run version that is not optimized for linear structure, set nonOpt=true and
#  pass in funObj (preferably version with no gradient calculation) to objFunc
#  pass in funObj (with gradient calculation) to fPrimeFunc. unlike the optimized version,
#   this calculates gradient and not fPrime. Wolfe conditions do not require gradients,
#   only directional derivatives. thus optimized version uses fPrimeFunc and not gradFunc.
#  Xw_prev and Xd are not used but need to be passed in with the correct dims
function lsWolfe(objFunc,fPrimeFunc,numDiff,X,Xw_prev,Xd,w_prev,c1,c2,f_prev,g,gTd,d;
 konst=nothing,verbose=false,maxLsIter=25,nonOpt=false)
    (m,n) = size(X)
    epsilon = 1e-6
    lsFailed = false
    nLsIter = 1
    nObjEvals = 0
    nGradEvals = 0
    nMatMult = 0

    c1gTd = c1*gTd
    c2gTd = c2*gTd
    Xw = Xw_prev
    w = w_prev
    f = f_prev # f_prev is phi(0), i.e. from prev outer loop iteration

    t = epsilon
    t_lsPrev = 0
    gTd_lsPrev = gTd 
    gTd_new = gTd

    alphaLo = t
    alphaHi = t

    if verbose
        @printf("lsWolfe: f_prev=%f\n",f_prev)	
    end

    # bracketing phase
    while nLsIter <= maxLsIter  
        # evaluate phi(alpha_i)
        w = w_prev + t*d
        f_lsPrev = f
        
        gTd_lsPrev = gTd_new
        if nonOpt
            if numDiff
                f,_,nmm = objFunc(w,X); nMatMult += nmm
                g_new,nmm = numGrad(objFunc,w,X); nMatMult += nmm
                nObjEvals += 2*n+1
            else
                f,g_new,nmm = fPrimeFunc(w,X); nMatMult += nmm
                nObjEvals += 1
                nGradEvals += 1
            end
            gTd_new = dot(g_new,d)
        else
            Xw = Xw_prev + t*Xd
            f,nmm = objFunc(Xw,w,konst=konst); nMatMult += nmm
            nObjEvals += 1
            if numDiff
                g_new,nmm = numGrad(objFunc,w,X,Xw,k=konst)
                nObjEvals += 2*n
                gTd_new = dot(g_new,d)
            else
                fPrime,addComp,nmm = fPrimeFunc(Xw,w)
                nGradEvals += 1
                gTd_new = dot(fPrime,Xd)
                if addComp!= nothing
                    gTd_new += dot(addComp,d) #g^Td=((X^TfPrime)+addComp)^Td = gradF^T(Xd)+addComp^Td
                end
            end
            nMatMult += nmm
        end
        
        # condition 1 for exiting bracketing phase: 
        #   phi(alpha_i) > phi(0)+c1*alpha_i*phi'(0) OR [phi(alpha_i) >= phi(alpha_{i-1}) and i>1]
        thresh = f_prev + t*c1gTd
        if verbose
            @printf("nLsIter=%d, t=%f, t_lsPrev=%f, f=%f, f_lsPrev=%f, fref=%f, thresh=%f\n",
                nLsIter,t,t_lsPrev,f,f_lsPrev,f_prev,thresh)
        end
                
        if f > thresh || (nLsIter>1 && f >= f_lsPrev)
            alphaLo = t_lsPrev
            alphaHi = t
            if verbose
                @printf("Exiting bracket phase from condition 1: alphaLo=%f, t=alphaHi=%f\n",alphaLo,alphaHi)
            end
            break #zoom(t_lsPrev,t)
        end
    
        # evaluate phi'(alpha_i)  
        if abs(gTd_new) <= -c2gTd
            if verbose
                @printf("Exiting bracket phase from condition 2: returning stepsize %f\n",t)
            end
            return (t,f,w,Xw,lsFailed,nLsIter,nObjEvals,nGradEvals,nMatMult)            
        elseif gTd_new >= 0
            alphaLo = t
            alphaHi = t_lsPrev
            if verbose
                @printf("Exiting bracket phase from condition 3: t=alphaLo=%f, alphaHi=%f\n",
                    alphaLo,alphaHi)
            end
            break #zoom(t,t_lsPrev)
        end
        
        # choose alpha_{i+1} in (alpha_i,alpha_max)
        #t_new=lsInterpolate(0,f,f_prev,t,g,mult=10.0)
        t_new=lsInterpolate(2,f,f_lsPrev,t,g,gTd_new,t1=t_lsPrev,gTd1=gTd_lsPrev)
        t_lsPrev = t
        t = t_new
        nLsIter += 1
    end

    if nLsIter > maxLsIter
        lsFailed = true
        if verbose
            @printf("Maximum number of LS iter reached in bracketing phase.\n")
        end
        return (t,f,w,Xw,lsFailed,nLsIter,nObjEvals,nGradEvals,nMatMult)
    end
    
    # zoom phase
    foundIt = false
    while nLsIter <= maxLsIter    
        t = (alphaLo+alphaHi)/2.0 # just bisection to find next trial point
        
        # evaluate phi(alpha_j)
        w = w_prev + t*d
        if nonOpt
            if numDiff
                f,_,nmm = objFunc(w,X); nMatMult += nmm
                g_new,nmm = numGrad(objFunc,w,X); nMatMult += nmm
                nObjEvals += 2*n+1
            else
                f,g_new,nmm = fPrimeFunc(w,X); nMatMult += nmm
                nObjEvals += 1
                nGradEvals += 1
            end
            gTd_new = dot(g_new,d)
        else
            Xw = Xw_prev + t*Xd
            f,nmm = objFunc(Xw,w,konst=konst); nMatMult += nmm
            if numDiff
                g_new,nmm = numGrad(objFunc,w,X,Xw,k=konst)
                nObjEvals += 2*n
                gTd_new = dot(g_new,d)           
            else
                gradF,addComp,nmm = fPrimeFunc(Xw,w)
                nGradEvals += 1
                gTd_new = dot(gradF,Xd)
                if addComp!=nothing
                    gTd_new += dot(addComp,d)        
                end
            end
            nObjEvals += 1
            nMatMult += nmm
        end
        thresh = f_prev + t*c1gTd
        
        # condition 1
        wLo = w_prev+alphaLo*d
        if nonOpt
            fLo,_,nmm = objFunc(wLo,X)
        else
            fLo,nmm = objFunc(Xw_prev+alphaLo*Xd,wLo,konst=konst)
        end
        nMatMult += nmm
        nObjEvals += 1
        if f > thresh || f >= fLo
            if verbose
                @printf("Zoom phase condition 1: old=(%f,%f), new=(%f,%f)\n",alphaLo,
                    alphaHi,alphaLo,t)
            end
            alphaHi = t
        else
            # evaluate phi'(alpha_j)     
            if abs(gTd_new) <= -c2gTd # stopping condition
                foundIt = true
                if verbose
                    @printf("Zoom phase condition 2: t=%f works\n",t)
                end
                break
            elseif gTd_new*(alphaHi-alphaLo) >= 0.0
                if verbose
                    @printf("Zoom phase condition 3: old=(%f,%f), new=(%f,%f)\n",alphaLo,
                        alphaHi,t,alphaLo)
                end
                alphaHi = alphaLo
            else
                if verbose
                    @printf("Zoom phase otherwise: old=(%f,%f), new=(%f,%f)\n",alphaLo,
                        alphaHi,t,alphaHi)
                end
            end
            alphaLo = t  
        end
        nLsIter += 1
        
        if abs(alphaHi-alphaLo)<epsilon
            if verbose
                @printf("Bracket too small. Exiting zoom phase.\n")
            end
            break
        end
    end
    
    if !foundIt
        if verbose
            @printf("Maximum number of LS iter reached in zoom phase.\n")
        end
        lsFailed=true
    end

    return (t,f,w,Xw,lsFailed,nLsIter,nObjEvals,nGradEvals,nMatMult)
end

# TODO: add nonOpt
# TODO: add numDiff
# TODO: make sure that mult used in lsArmijo is 0.5
# t = 1/L, c1=1/2, initialize each linesearch with t=t_prev
function lsLipschitz(objFunc,fPrimeFunc,X,Xw_prev,Xd,w_prev,t_prev,f_prev,fref,g,gTd,d;
 konst=nothing,verbose=false,maxLsIter=25,nonOpt=false)
    lsFailed = false
    nLsIter = 0
    nObjEvals = 0
    nGradEvals = 0
    nMatMult = 0   

    (t,f,w,Xw,lsFailed,nLsIter,nObjEvals,nGradEvals,nMatMult) = 
        lsArmijo(objFunc,fPrimeFunc,X,Xw_prev,Xd,w_prev,t_prev,0.5,f_prev,fref,g,gTd,d,
        konst=konst,verbose=verbose,maxLsIter=maxLsIter,nonOpt=nonOpt)

    return (t,f,w,Xw,lsFailed,nLsIter,nObjEvals,nGradEvals,nMatMult)
end

### strong Wolfe conditions - subspace search ###
function mdWolfe(objFunc,fPrimeFunc,numDiff,X,Xw_prev,XD,w_prev,c1,c2,f,f_prev,g,gTdVec,D;
 konst=nothing,verbose=false,maxLsIter=25,nonOpt=false)
    (m,n) = size(X)
    (_,k) = size(XD)
    epsilon = 1e-6
    lsFailed = false
 
    nLsIter = 0
    nObjEvals = 0
    nGradEvals = 0
    nMatMult = 0   
    
    c1gTdVec = c1 .* gTdVec
    c2gTdVec = c2 .* gTdVec

    w = w_prev
    Xw = Xw_prev
    f = f_prev

    t = ones(k,1).*epsilon
    t_lsPrev = zeros(k,1)

    alphaLo = t
    alphaHi = t
    
    if verbose
        @printf("mdWolfe: f_prev=%f, norm(g)=%f\n",f_prev,norm(g))	
    end

    # bracketing phase
    while nLsIter <= maxLsIter
        # evaluate phi(alpha_i)
        w = w_prev + D*t
        f_lsPrev = f

        if nonOpt
            if numDiff
                f,_,nmm = objFunc(w,X); nMatMult += nmm
                g_new,nmm = numGrad(objFunc,w,X); nMatMult += nmm
                nObjEvals += 2*n+1
            else
                f,g_new,nmm = fPrimeFunc(w,X); nMatMult += nmm
                nObjEvals += 1
                nGradEvals += 1
            end
            gTdVec_new = transpose(g_new'*D)
        else
            Xw = Xw_prev + XD*t
            f,nmm = objFunc(Xw,w,konst=konst); nMatMult += nmm
            nObjEvals += 1
            if numDiff
                g_new,nmm = numGrad(objFunc,w,X,Xw,k=konst)
                nObjEvals += 2*n
                gTdVec_new = transpose(g_new'*D)
            else
                fPrime,addComp,nmm = fPrimeFunc(Xw,w)
                nGradEvals += 1
                gTdVec_new = transpose(fPrime'*XD)
                if addComp!= nothing
                    gTdVec_new += transpose(addComp'*D) #g^TD=((X^TfPrime)+addComp)^TD = gradF^T(XD)+addComp^Td
                end
            end
            nMatMult += nmm
        end

        # condition 1 for exiting bracketing phase: 
        #   phi(alpha_i) > phi(0)+c1*alpha_i*phi'(0) OR [phi(alpha_i) >= phi(alpha_{i-1}) and i>1]
        thresh = f_prev + dot(t,c1gTdVec)
        if verbose
            @printf("  nLsIter=%d, t=%f, t_lsPrev=%f, f=%f, f_lsPrev=%f, fref=%f, thresh=%f\n",
                nLsIter,t,t_lsPrev,f,f_lsPrev,f_prev,thresh)
        end

        if f > thresh || (nLsIter>1 && f >= f_lsPrev)
            alphaLo = t_lsPrev
            alphaHi = t
            if verbose
                @printf("Exiting bracket phase from condition 1: alphaLo=%f, t=alphaHi=%f\n",alphaLo,alphaHi)
            end
            break #zoom(t_lsPrev,t)
        end
      
        t_new = mdInterpolate(0,t,mult=10)
        #t_new = mdInterpolate(1,t,g=g,f=f,f1=f_lsPrev)
        t_lsPrev = t
        t = t_new
        nLsIter += 1
    end

    if nLsIter > maxLsIter
        lsFailed = true
        if verbose
            @printf("Maximum number of LS iter reached in bracketing phase.\n")
        end
        return (t,f,w,Xw,lsFailed,nLsIter,nObjEvals,nGradEvals,nMatMult)
    end

    # zoom phase
    foundIt = false
    while nLsIter <= maxLsIter
        t = (alphaLo .+ alphaHi) ./2.0 # TODO: Newtons!

        # evaluate phi(alpha_j) TODO: this block exactly the same as block above
        w = w_prev + D*t
        if nonOpt
            if numDiff
                f,_,nmm = objFunc(w,X); nMatMult += nmm
                g_new,nmm = numGrad(objFunc,w,X); nMatMult += nmm
                nObjEvals += 2*n+1
            else
                f,g_new,nmm = fPrimeFunc(w,X); nMatMult += nmm
                nObjEvals += 1
                nGradEvals += 1
            end
            gTdVec_new = transpose(g_new'*D)
        else
            Xw = Xw_prev + XD*t
            f,nmm = objFunc(Xw,w,konst=konst); nMatMult += nmm
            if numDiff
                g_new,nmm = numGrad(objFunc,w,X,Xw,k=konst)
                nObjEvals += 2*n
                gTdVec_new = transpose(g_new'*D)
            else
                fPrime,addComp,nmm = fPrimeFunc(Xw,w)
                nGradEvals += 1
                gTdVec_new = transpose(fPrime'*XD)
                if addComp!= nothing
                    gTdVec_new += transpose(addComp'*D) 
                end
            end
            nObjEvals += 1
            nMatMult += nmm
        end
        thresh = f_prev + dot(t,c1gTdVec)

        # condition 1
        wLo = w_prev + D*alphaLo
        if nonOpt
            fLo,_,nmm = objFunc(wLo,X)
        else
            fLo,nmm = objFunc(Xw_prev+XD*alphaLo,wLo,konst=konst)
        end
        nMatMult += nmm
        nObjEvals += 1
        if f > thresh || f >= fLo
            if verbose
                @printf("Zoom phase condition 1: old=(%f,%f), new=(%f,%f)\n",alphaLo,
                    alphaHi,alphaLo,t)
            end
            alphaHi = t
        else
            # evaluate phi'(alpha_j)   TODO: CHECK THESE CONDITIONS!!! avg vs individual directions?!!
            #if all(abs.(gTdVec_new) .<= -c2gTdVec) # stopping condition
            if abs(dot(gTdVec_new,t)) <= -abs(dot(c2gTdVec,t)) # stopping condition
                foundIt = true
                if verbose
                    @printf("Zoom phase condition 2: t=%f works\n",t)
                end
                break
            elseif dot(gTdVec_new,(alphaHi-alphaLo)) >= 0.0
                if verbose
                    @printf("Zoom phase condition 3: old=(%f,%f), new=(%f,%f)\n",alphaLo,
                        alphaHi,t,alphaLo)
                end
                alphaHi = alphaLo
            else
                if verbose
                    @printf("Zoom phase otherwise: old=(%f,%f), new=(%f,%f)\n",alphaLo,
                        alphaHi,t,alphaHi)
                end
            end
            alphaLo = t  
        end
        nLsIter += 1

        if all(abs.(alphaHi.-alphaLo).<epsilon)
            if verbose
                @printf("Bracket too small. Exiting zoom phase.\n")
            end
            break
        end
    end

    if !foundIt
        if verbose
            @printf("Maximum number of LS iter reached in zoom phase.\n")
        end
        lsFailed=true
    end

    return (t,f,w,Xw,lsFailed,nLsIter,nObjEvals,nGradEvals,nMatMult)
end


# find t in R^k, k is the number of directions, that minimizes f(Xw_new)=f(X(w+D*t))
#  D in R^{n x k) where the first column is descent direction from the chosen method 
#   and the next k-1 columns are other search directions (e.g. momentum) 
# to run version that is not optimized for linear structure, set nonOpt=true and
#  pass in funObjForT to funObjForT
#  pass in funObj (preferably version with no gradient calculation) to objFunc
#  pass in funObj (with gradient calculation) to gradFunc
#  Xw_prev and Xd are not used but need to be passed in with the correct dims
function lsDDir(objFunc,fPrimeFunc,gradFunc,XD,D,X,Xw_prev,w_prev,f_prev,g,outerIter,gradNorm,gTd,c1,c2;
 konst=nothing,ssMethod=0,ssLS=0,verbose=false,maxLsIter=25,progTol=1e-9,optTol=1e-9,
 relativeStopping=true,oneDInit=true,nonOpt=false,funObjForT=nothing,fPrimePrimeFunc=nothing,hessFunc=nothing)
    lsFailed = false
    nLsIter = 1
    nObjEvals = 0
    nGradEvals = 0
    nMatMult = 0
    (m,n) = size(X)
    (_,k) = size(XD)
    
    if verbose
        if nonOpt
            @printf("lsDDir(nonOpt) at %dth outer iter. initial f_prev=%f. norm(w_prev,2)=%f, 
                k=%d, size(w_prev)=%s\n",outerIter,f_prev,norm(w_prev,2),k,size(w_prev))
        else
            normKonst = -1.0
            f_prev,nmm = objFunc(Xw_prev,w_prev,konst=konst); nMatMult += nmm
            if konst != nothing
                normKonst = norm(konst,2)
            end
            @printf("lsDDir(opt) at %dth outer iter. initial f_prev=%f. norm(w_prev,2)=%f, 
                k=%d, norm(konst,2)=%f\n",outerIter,f_prev,norm(w_prev,2),k,normKonst)
        end
    end
    
    # call minFunc on f(X(w_prev+Dt)) 
    if konst == nothing
        konst2 = Xw_prev
    else
        konst2 = Xw_prev .+ konst
    end
    if relativeStopping
        progTol = progTol*gradNorm
        optTol = optTol*gradNorm
    end
    
    max1dInitIter = maxLsIter/2 #min(10,maxLsIter)
    t = zeros(k,1) 
    if oneDInit
        if ssLS==1            
            (t0,f,_,_,lsFailed,nLsI,nOE,nGE,nMM) = 
                lsWolfe(objFunc,fPrimeFunc,false,X,Xw_prev,XD[:,1],w_prev,c1,c2,f_prev,g,gTd,D[:,1],
                konst=konst,verbose=verbose,maxLsIter=max1dInitIter,nonOpt=nonOpt)
        else
            (t0,f,_,_,lsFailed,nLsI,nOE,nGE,nMM) = 
                lsArmijo(objFunc,fPrimeFunc,X,Xw_prev,XD[:,1],w_prev,1.0,c1,f_prev,f_prev,g,
                gTd,D[:,1],konst=konst,verbose=verbose,maxLsIter=max1dInitIter,nonOpt=nonOpt)
        end
        
        if !lsFailed
            t[1] = t0
        end
        nLsIter += nLsI 
        nObjEvals += nOE
        nGradEvals += nGE
        nMatMult += nMM         
    end
    
    maxIterRemaining = maxLsIter-nLsIter
    if maxIterRemaining < 1
        lsFailed = true
        nIter = 0
        @printf("lsDDir: maxIterRemaining=%d\n",maxIterRemaining)
    else
        if nonOpt
            # function value and gradient wrt step sizes used for non-opt minFuncSO
            funObj(t,X) = funObjForT(t,D,w_prev,X)
            (t,f,nOE,nGE,nIter,nLsI,nMM,_,_,_,_,_,_,_) = minFuncSO(funObj,funObj,funObj,t,X,
                method=ssMethod,maxIter=max(maxLsIter-nLsIter,1),nonOpt=true,
                funObj=funObj,funObjNoGrad=funObj,
                lsInit=0,lsType=ssLS,c1=c1,verbose=verbose,progTol=progTol,optTol=optTol)
        else
            (t,f,nOE,nGE,nIter,nLsI,nMM,_,_,_,_,_,_,_) = minFuncSO(objFunc,fPrimeFunc,gradFunc,t,XD,
                konst=konst2,method=ssMethod,maxIter=max(maxLsIter-nLsIter,1),
                lsInit=0,lsType=ssLS,c1=c1,verbose=verbose,progTol=progTol,optTol=optTol,
                fPrimePrimeFunc=fPrimePrimeFunc,hessFunc=hessFunc)
        end
    end

    nIter = nIter + nLsIter + nLsI
    nObjEvals += nOE
    nGradEvals += nGE
    nMatMult += nMM
    w = w_prev + D*t 
    Xw = X*w; nMatMult += 1
    
    if verbose
        @printf("  after calling minFunc. f=%f, norm(Xw,2)=%f\n t: %s\n",f,norm(Xw,2),string(t))
    end
    
    return (t,f,w,Xw,lsFailed,nIter,nObjEvals,nGradEvals,nMatMult)
end