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
function getAndSetLSFref(nFref,oldFvals,i,f_prev)
    j = mod(i,nFref)
    if j== 0
        j = nFref
    end

    oldFvals[j]=f_prev
    return findmax(oldFvals)    
end

### interpolate new step size ###
function lsInterpolate(lsInterpType,f,fref,t,g;verbose=false)
    oldT = t
    if lsInterpType==0
        return (t*0.5)
    elseif lsInterpType==1 # Fit a degree-2 polynomial to set step-size
        gg=dot(g,g)
        if(isfinitereal(f))
        	t= t^2*gg/(2(f - fref + t*gg))
        	if verbose
                @printf("lsInterpolate, deg-2 poly: f=%f, fref=%f, oldT=%f, t=%f, gTg=%f\n",
                    f,fref,oldT,t,gg)
            end
        else
            return (t*0.5)
        end
    end
    
    return t
end

### backtracking Armijo ###
function lsArmijo(objFunc,X,Xw_prev,Xd0,w_prev,t0,c1,f0,f_prev,fref,g_prev,gTd,d;
 konst=nothing,verbose=false,maxLsIter=25)
    lsFailed = false
    nLsIter = 1
    nObjEvals = 0
    t = t0
    f = f0
    Xw = Xw_prev + t0*Xd0
    w = w_prev + t0*d

    if verbose
        @printf("minFuncLinear:lsArmijo: f0=%.4f, f_prev=%.4f, norm(w0,2)=%f, norm(w_prev,2)=%f, t0=%f\n",
            f0,f_prev,norm(w,2),norm(w_prev,2),t0)	
    end
    
    c1gTd = c1*gTd
    thresh = fref + t*c1gTd    
    while f > thresh && nLsIter <= maxLsIter # sufficient decrease not yet satisfied
        t_lsPrev = t
        # use interpolation that does not need gradient at new point
        t = lsInterpolate(1,f,f_prev,t,g_prev)
            
        # calculate f at new point
        w = w_prev + t*d
        Xw = Xw_prev + t*Xd0
        f = objFunc(Xw,konst=konst)
        nObjEvals += 1
        thresh = fref + t*c1gTd 
        
        if verbose
            @printf("nLsIter=%d, t=%f, t_lsPrev=%f, f=%f, f_prev=%f, fref=%f, thresh=%f\n",
                nLsIter,t,t_lsPrev,f,f_prev,fref,thresh)
        end
        
        nLsIter += 1
    end

    if nLsIter > maxLsIter
        lsFailed = true
    end

    if verbose
        @printf("lsArmijo returns t=%f, f=%f, nLsIter=%d\n",t,f,nLsIter)
    end
    return (t,f,w,Xw,lsFailed,nLsIter,nObjEvals)
end

### backtracking Armijo ###
function lsArmijoNonOpt(funObjNoGrad,X,w_prev,t0,c1,f0,f_prev,fref,g_prev,gTd_prev,d;
 verbose=false,maxLsIter=25)
    (m,n)=size(X)
    lsFailed = false
    nLsIter = 1
    nObjEvals = 0
    nMatMult = 0
    t = t0
    f = f0
    w = w_prev + t0*d

    if verbose
        @printf("minFunc:lsArmijoNonOpt: f=%f, f_prev=%f, t0=%f\n",f,f_prev,t0)	
    end
    
    c1gTd = c1*gTd_prev
    thresh = fref + t*c1gTd    
    while f > thresh  && nLsIter <= maxLsIter # sufficient decrease not yet satisfied
        t_lsPrev = t
        # use interpolation that does not need gradient at new point
        t = lsInterpolate(1,f,f_prev,t,g_prev)
              
        # calculate f at new point
        w = w_prev + t*d
        f,_ = funObjNoGrad(w,X)
        nObjEvals += 1
        nMatMult += 1
        thresh = fref + t*c1gTd 
        
        if verbose
            @printf("nLsIter=%d, t=%f, t_lsPrev=%f, f=%f, f_prev=%f, fref=%f, thresh=%f\n",
                nLsIter,t,t_lsPrev,f,f_prev,fref,thresh)
        end
        nLsIter += 1
    end

    if nLsIter > maxLsIter
        lsFailed = true
    end

    if verbose
        @printf("lsArmij returns t=%f, f=%f, nLsIter=%d\n",t,f,nLsIter)
    end
    return (t,f,w,lsFailed,nLsIter,nObjEvals,0,nMatMult)
end

### strong Wolfe conditions ###
function lsWolfe(objFunc,gradFunc,numDiff,X,Xw_prev,Xd,w_prev,c1,c2,f,f_prev,g,gTd,d;
 konst=nothing,verbose=false,maxLsIter=25)
    epsilon = 1e-6
    lsFailed = false
    nLsIter = 1
    nObjEvals = 0
    nGradEvals = 0
    (m,n) = size(X)

    c1gTd = c1*gTd
    c2gTd = c2*gTd

    Xw = Xw_prev
    w = w_prev # CHECK THIS!!

    t = epsilon
    t_lsPrev = 0 

    alphaLo = t
    alphaHi = t

    if verbose
        @printf("lsWolfe: f=%f, f_prev=%f, t=%f\n",f,f_prev,t)	
    end

    # bracketing phase
    while nLsIter <= maxLsIter  
        # evaluate phi(alpha_i)
        w = w_prev + t*d
        Xw = Xw_prev + t*Xd
        f_lsPrev = f
        f = objFunc(Xw,konst=konst)
        nObjEvals += 1 
        
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
        if numDiff
            g_new = numGrad(objFunc,w,X,Xw,k=konst)
            nObjEvals += 2*n
            gTd_new = dot(g_new,d)
        else
            gradF = gradFunc(Xw)
            nGradEvals += 1
            gTd_new = dot(gradF,Xd) #g^Td=(X^TgradF)^Td = gradF^T(Xd)
        end
        
        if abs(gTd_new) <= -c2gTd
            if verbose
                @printf("Exiting bracket phase from condition 2: returning stepsize %f\n",t)
            end
            return (t,f,w,Xw,lsFailed,nLsIter,nObjEvals,nGradEvals)
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
        t_lsPrev = t
        t *= 10 
        
        nLsIter += 1
    end

    if nLsIter > maxLsIter
        lsFailed = true
        if verbose
            @printf("Maximum number of LS iter reached in bracketing phase.\n")
        end
        return (t,f,w,Xw,lsFailed,nLsIter,nObjEvals,nGradEvals)
    end
    
    # zoom phase
    foundIt = false
    while nLsIter <= maxLsIter    
        t = (alphaLo+alphaHi)/2.0 # just bisection to find next trial point
        
        # evaluate phi(alpha_j)
        w = w_prev + t*d
        Xw = Xw_prev + t*Xd
        f = objFunc(Xw,konst=konst)
        nObjEvals += 1
        thresh = f_prev + t*c1gTd
        
        # condition 1
        fLo = objFunc(Xw_prev+alphaLo*Xd,konst=konst)
        nObjEvals += 1
        if f > thresh || f >= fLo
            if verbose
                @printf("Zoom phase condition 1: old=(%f,%f), new=(%f,%f)\n",alphaLo,
                    alphaHi,alphaLo,t)
            end
            alphaHi = t
        else
            # evaluate phi'(alpha_j)
            if numDiff
                g_new = numGrad(objFunc,w,X,Xw,k=konst)
                nObjEvals += 2*n
                gTd_new = dot(g_new,d)           
            else
                gradF = gradFunc(Xw)
                nGradEvals += 1
                gTd_new = dot(gradF,Xd)        
            end
            
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

    return (t,f,w,Xw,lsFailed,nLsIter,nObjEvals,nGradEvals)
end

function lsWolfeNonOpt(funObj,funObjNoGrad,numDiff,X,w_prev,c1,c2,f,f_prev,g,gTd_prev,d;
 verbose=false,maxLsIter=25)
    (m,n)=size(X)
    epsilon = 1e-6
    lsFailed = false
    nLsIter = 1
    nObjEvals = 0
    nGradEvals = 0
    nMatMult = 0

    c1gTd_prev = c1*gTd_prev
    c2gTd_prev = c2*gTd_prev
    t = epsilon
    f = f_prev
    w = w_prev # CHECK THIS!!

    alphaLo = t
    alphaHi = t

    if verbose
        @printf("lsWolfeNonOpt: f=%f, f_prev=%f\n",f,f_prev)	
    end

    # bracketing phase
    t_lsPrev = 0    
    while nLsIter <= maxLsIter 
        # evaluate phi(alpha_i)
        w = w_prev + t*d
        f_lsPrev = f
        f,g = funObj(w,X) # the g that is overwritten was calced w/o ls step
        if numDiff
            g = numGrad(funObj,w,X) 
            nObjEvals += 2*n+1 
            nMatMult += 2*n+1
        else
            nObjEvals += 1 
            nGradEvals += 1
            nMatMult += 2
        end
        
        # condition 1 for exiting bracketing phase: 
        #   phi(alpha_i) > phi(0)+c1*alpha_i*phi'(0) OR [phi(alpha_i) >= phi(alpha_{i-1}) and i>1]
        thresh = f_prev + t*c1gTd_prev
        if verbose
            @printf("nLsIter=%d, t=%f, t_lsPrev=%f, f_prev=%f, f_lsPrev=%f, f=%f, thresh=%f\n",
                nLsIter,t,t_lsPrev,f_prev,f_lsPrev,f,thresh)
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
        gTd_new = dot(g,d)
        
        if abs(gTd_new) <= -c2gTd_prev
            if verbose
                @printf("Exiting bracket phase from condition 2: returning stepsize %f\n",t)
            end
            return (t,f,w,lsFailed,nLsIter,nObjEvals,nGradEvals,nMatMult)
        elseif gTd_new >= 0
            alphaLo = t
            alphaHi = t_lsPrev
            if verbose
                @printf("Exiting bracket phase from condition 3: t=alphaLo=%f, alphaHi=%f\n",alphaLo,alphaHi)
            end
            break #zoom(t,t_lsPrev)
        end
        
        # choose alpha_{i+1} in (alpha_i,alpha_max)
        t_lsPrev = t
        t *= 10 
        
        nLsIter += 1
    end

    if nLsIter > maxLsIter
        lsFailed = true
        if verbose
            @printf("Maximum number of LS iter reached in bracketing phase. Returning.\n")
        end
        return (t,f,w,lsFailed,nLsIter,nObjEvals,nGradEvals,nMatMult)
    end
    
    # zoom phase
    foundIt = false
    while nLsIter <= maxLsIter    
        t = (alphaLo+alphaHi)/2.0 # just bisection to find next trial point
        
        # evaluate phi(alpha_j)
        w = w_prev + t*d
        f,g = funObj(w,X) 
        if numDiff
            g = numGrad(funObj,w,X)
            nObjEvals += 2*n+1
            nMatMult += 2*n+1
        else
            nObjEvals += 1
            nGradEvals += 1
            nMatMult += 2
        end
        thresh = f_prev + t*c1gTd_prev
               
        # condition 1
        wLo = w_prev+alphaLo*d
        fLo,_ = funObjNoGrad(wLo,X)
        nObjEvals += 1
        nMatMult += 1
   
        if f > thresh || f >= fLo
            if verbose
                @printf("Zoom phase condition 1: old=(%f,%f), new=(%f,%f)\n",alphaLo,
                    alphaHi,alphaLo,t)
            end
            alphaHi = t
        else
            # evaluate phi'(alpha_j)
            gTd_new = dot(g,d)
            
            if abs(gTd_new) <= -c2gTd_prev # stopping condition
                foundIt = true
                if verbose
                    @printf("Zoom phase condition 2: t=%f works\n",t)
                end
                break
            elseif gTd_new*(alphaHi-alphaLo) >= 0.0
                if verbose
                    @printf("Zoom phase condition 3: old=(%f,%f), new=(%f,%f)\n",alphaLo,alphaHi,t,alphaLo)
                end
                alphaHi = alphaLo
            else
                if verbose
                    @printf("Zoom phase otherwise: old=(%f,%f), new=(%f,%f)\n",alphaLo,alphaHi,t,alphaHi)
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
    end

    return (t,f,w,lsFailed,nLsIter,nObjEvals,nGradEvals,nMatMult)
end

# find t in R^k, k is the number of directions, that minimizes f(Xw_new)=f(X(w+D*t))
#  D in R^{n x k) where the first column is descent direction from the chosen method 
#   and the next k-1 columns are other search directions (e.g. momentum) 
#  solves this with minFuncNonOpt
function lsDDir(funObjForT,XD,D,X,Xw_prev,w_prev,f_prev,g,outerIter,gradNorm,gTd,c1,c2;
 konst=nothing,ssMethod=0,ssLS=0,verbose=false,maxLsIter=25,progTol=1e-6,optTol=1e-6,
 relativeStopping=true,oneDInit=true)
    lsFailed = false
    nLsIter = 1
    nObjEvals = 0
    nGradEvals = 0
    nMatMult = 0
    (m,n) = size(X)
    (_,k) = size(D)
    
    t = zeros(k,1)
    if oneDInit
        if ssLS==1                  
            (t0,f,_,lsFailed,nLsIterInit,nOEInit,nGEInit,nMM) = 
                lsWolfeNonOpt(funObj,funObjNoGrad,false,X,w_prev,c1,c2,f_prev,f_prev,g,gTd,D[:,1],
                verbose=verbose,maxLsIter=maxLsIter)
            nGradEvals = nGradEvals + nGEInit
        else                
            (t0,f,_,lsFailed,nLsIterInit,nOEInit,nGEInit,nMM) = 
                lsArmijoNonOpt(funObjNoGrad,X,w_prev,1.0,c1,f_prev,f_prev,g,gTd,D[:,1],
                verbose=verbose,maxLsIter=maxLsIter) 
        end
        if !lsFailed
            t[1] = t0
        end
        nLsIter = nLsIter + nLsIterInit
        nObjEvals = nObjEvals + nOEInit
        nGradEvals = nGradEvals + nGEInit
        nMatMult = nMatMult + nMM
    end
    
    # function value and gradient wrt step sizes used for standard minFunc
    funObj(t,X) = funObjForT(t,D,w_prev,X)
 
    if verbose
        (f_prev,g_prev) = funObj(t,X)
        @printf("lsDDir at %dth iter. initial f_prev=%f, norm(w_prev,2)=%f, norm(g_prev,2)=%f\n",
            outerIter,f_prev,norm(w_prev,2),norm(g_prev,2))
    end
    
    if relativeStopping
        progTol = progTol*gradNorm
        optTol = optTol*gradNorm
    end

    (t,f,nObjEvals,nGradEvals,nIter,nLsIterMain,nMM,_,_) = minFuncNonOpt(funObj,t,X,
        method=ssMethod,maxIter=max(maxLsIter-nLsIter,1),lsInit=0,
        lsType=ssLS,c1=c1,verbose=verbose,progTol=progTol,optTol=optTol)

    nIter = nIter + nLsIter + nLsIterMain
    w = w_prev + D*t
    Xw = X*w
    nMatMult += nMM
    
    if verbose
        @printf("  after calling minFunc. f=%f, norm(Xw,2)=%f\n t: %s\n",f,norm(Xw,2),
            string(t))
    end
    
    return (t,f,w,Xw,lsFailed,nIter,nObjEvals,nGradEvals,nMatMult)
end

#  same as lsDDir except solves this with minFuncLinear
function lsDDirLinear(objFunc,gradFunc,XD,D,X,Xw_prev,w_prev,f_prev,g,outerIter,gradNorm,gTd,c1,c2;
 konst=nothing,ssMethod=0,ssLS=0,verbose=false,maxLsIter=25,progTol=1e-6,optTol=1e-6,
 relativeStopping=true,oneDInit=true)
    lsFailed = false
    nLsIter = 1
    nObjEvals = 0
    nGradEvals = 0
    (m,n) = size(X)
    (_,k) = size(XD)
    
    if verbose
        f_prev = objFunc(Xw_prev,konst=konst)
        normKonst = -1.0
        if konst != nothing
            normKonst = norm(konst,2)
        end
        @printf("lsDDirLinear at %dth outer iter. initial f_prev=%f. norm(w_prev,2)=%f, 
            k=%d, norm(konst,2)=%f\n",outerIter,f_prev,norm(w_prev,2),k,normKonst)
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
    
    t = zeros(k,1) 
    if oneDInit
        if ssLS==1            
            (t0,f,_,_,lsFailed,nLsIterInit,nOEInit,nGEInit) = 
                lsWolfe(objFunc,gradFunc,false,X,Xw_prev,XD[:,1],w_prev,c1,c2,f_prev,f_prev,g,gTd,D[:,1],
                konst=konst,verbose=verbose,maxLsIter=maxLsIter)
            nGradEvals = nGradEvals + nGEInit
        else
            (t0,f,_,_,lsFailed,nLsIterInit,nOEInit) = 
                lsArmijo(objFunc,X,Xw_prev,XD[:,1],w_prev,1.0,c1,f_prev,f_prev,f_prev,g,
                gTd,D[:,1],konst=konst,verbose=verbose,maxLsIter=maxLsIter)
        end
        if !lsFailed
            t[1] = t0
        end
        nLsIter = nLsIter + nLsIterInit 
        nObjEvals = nObjEvals + nOEInit
         
    end
    (t,f,nObjEvals,nGradEvals,nIter,nLsIterMain,nMatMult,_,_) = 
        minFuncSO(objFunc,gradFunc,t,XD,konst=konst2,method=ssMethod,maxIter=max(maxLsIter-nLsIter,1),
        lsInit=0,lsType=ssLS,c1=c1,verbose=verbose,progTol=progTol,optTol=optTol)
    
    nIter = nIter + nLsIter + nLsIterMain
    w = w_prev + D*t 
    Xw = X*w  
    nMatMult += (k+1)
    
    if verbose
        @printf("  after calling minFunc. f=%f, norm(Xw,2)=%f\n t: %s\n",f,norm(Xw,2),string(t))
    end
    
    return (t,f,w,Xw,lsFailed,nIter,nObjEvals,nGradEvals,nMatMult)
end