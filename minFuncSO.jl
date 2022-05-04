using Printf, LinearAlgebra
include("misc.jl")
include("linesearch.jl")
include("descentDir.jl")

function minFuncSO(objFunc,fPrimeFunc,gradFunc,w0,X;konst=nothing,method=1,maxIter=100,maxLsIter=50,maxTimeInSec=600,optTol=1e-5,
 progTol=1e-9,lsInit=0,nFref=1,lsType=0,c1=1e-4,c2=0.9,lsInterpType=0,lBfgsSize=30,momentumDirections=0,
 ssMethod=0,ssLS=0,ssRelStop=true,ssOneDInit=true,derivativeCheck=false,numDiff=false,nonOpt=false,verbose=false,
 funObjForT=nothing,funObj=nothing,funObjNoGrad=nothing,XXT=nothing,fPrimePrimeFunc=nothing,hessFunc=nothing,
 maxFailHessInv=2,lbfgsCautious=false,eigDecomp=false)
    (m,n)=size(X)
	nObjEvals = 0
	nGradEvals = 0
	nIter = 1
	totalLsIter = 0
	nMatMult = 0
    gradNorm = NaN
    normD = NaN
    lambdaHMin = NaN
    lambdaHMax = NaN

    time_start = time_ns() # Start timer
    time_end = time_start
    fValues = zeros(maxIter,1) # fValues by iterations
    tValues = zeros(maxIter,1) # total time taken in seconds

    # solution, model, objective value, gradient
    # gradients get calculated once at the start and then once per iter outside LS loop	
    w = w0
    Xw = fill(0.0, m)
    H = zeros((n,n))
    if nonOpt
        if numDiff
            f,_,nmm = funObjNoGrad(w,X); nMatMult += nmm
            g,nmm = numGrad(funObjNoGrad,w,X); nMatMult += nmm
            nObjEvals += 2*n+1            
        else
            f,g,nmm = funObj(w,X); nMatMult += nmm
            nObjEvals += 1
            nGradEvals += 1
        end
    else
        Xw = X*w; nMatMult += 1
        f,nmm = objFunc(Xw,w,konst=konst); nMatMult += nmm
        nObjEvals += 1	
        if numDiff
            g,nmm = numGrad(objFunc,w,X,Xw,k=konst); nMatMult += nmm
            nObjEvals += 2*n
        else
            g,nmm = gradFunc(Xw,w,X,konst=konst); nMatMult += nmm
            nGradEvals += 1
        end
        if method == 5 && fPrimePrimeFunc!=nothing && hessFunc!=nothing
            H,nMM = hessFunc(Xw,w,X,konst=konst)
        end
    end
    fValues[1,1] = f
    time_end = time_ns()
    tValues[1,1] = (time_end-time_start)/1.0e9
	
	if derivativeCheck
    	if numDiff
        	@printf("Not doing derivative check with nummDiff on\n")
    	else
        	if nonOpt
            	g2,nmm = numGrad(funObjNoGrad,w,X); nMatMult += nmm
        	else
        		g2,nmm = numGrad(objFunc,w,X,Xw,k=konst); nMatMult += nmm
        	end
    		if maximum(abs.(g-g2)) > optTol
    			@printf("User and numerical derivatives differ%s\n",string([g g2]))
    			sleep(1)
    		else
    			@printf("User and numerical derivatives agree\n")
    		end
    	end
	end
	
	# check initial point optimality
	gradNorm = norm(g,Inf)
	if gradNorm < optTol
    	return (w0,f,nObjEvals,nGradEvals,nIter,totalLsIter,nMatMult,fValues,tValues,gradNorm,NaN,
            normD,lambdaHMin,lambdaHMax,1.0)
    end
    
    if verbose
        @printf("minFuncSO called with initial f=%f, initial norm(g,Inf)=%f\n",f,gradNorm)
    end
    
    # step size, descent direction, directional derivative
    t = 1.0
    ts = []
    minT = t
    d = -g
    gTd = dot(g,d)
    
    dTN = zeros(n,1) # truncated Newton direction for lsType 5

    # old function values for nonmonotonic Armijo
	oldFvals = fill(-Inf,nFref)	
    oldFvals[1]=f

    # stored values
    DiffIterates = [zeros(n,1) for _ in 1:lBfgsSize]
    DiffGrads = [zeros(n,1) for _ in 1:lBfgsSize] 
    XDiffIterates = [zeros(m,1) for _ in 1:lBfgsSize] # assumes lbfgsSize >= momentumDirections
    XDiffGrads = [zeros(m,1) for _ in 1:lBfgsSize]

	lsDone = false 	
	alpha = 1.0 # for BB
	g_prev = g
	f_prev = f
	w_prev = w
	d_prev = d
	Xw_prev = Xw
    H_prev = H
    nFailedInv = 0 # for Newton
    t_prev = t # for Lipschitz linesearch

	while nIter < maxIter
        # compute descent direction
        d_prev = d
        cgIter = 0
        if method == 0 # steepest descent
            d = -g
    	elseif method == 1 # BB
        	d,alpha = bb(g,g_prev,w,w_prev,nIter,alpha)
        elseif method == 2 # CG
            d = cg(g,g_prev,nIter,d_prev,optTol)
        elseif method == 3 # l-BFGS
            d,H = lbfgs(g,nIter,DiffIterates,DiffGrads,cautious=lbfgsCautious)
            #d,H = lbfgs(g,nIter,DiffIterates,DiffGrads,skip=true) #HERE
        elseif method == 4 # Newton-CG
            if nonOpt
                d,cgIter,nOE,nGE,nMM = newtonCG(g,(maxIter-nIter-1),funObjNoGrad,funObj,numDiff,
                    w,X,k=konst,nonOpt=nonOpt)
            else
                d,cgIter,nOE,nGE,nMM = newtonCG(g,(maxIter-nIter-1),objFunc,gradFunc,numDiff,
                    w,X,k=konst,nonOpt=nonOpt)
            end
            nObjEvals += nOE
            nGradEvals += nGE
            nMatMult += nMM
        elseif method == 5 # Newton
            d,failedInv = newton(g,H)
            if failedInv
                nFailedInv += 1
                d=-g
            end
    	end
    	
    	# calculate truncated Newton as a second direction. counts towards maxIters
    	if lsType==5
        	if nonOpt
            	dTN,nIterTN,nOE,nGE,nMM = newtonCG(g,(maxIter-nIter-1),funObjNoGrad,funObj,numDiff,
                    w,X,k=konst,nonOpt=nonOpt)
        	else
            	dTN,nIterTN,nOE,nGE,nMM = newtonCG(g,(maxIter-nIter-1),objFunc,gradFunc,numDiff,
                	w,X,k=konst,nonOpt=nonOpt)
            end
            nObjEvals += nOE
            nGradEvals += nGE
            nMatMult += nMM
            cgIter = cgIter + nIterTN
    	end

        # calculate (diagonal) Adagrad as a second direction.
        if lsType==7
            dAdagrad = -sqrt(Diagonal(g .* g'))*g # optimized for Diagonal and no mat-vec mult
        end
    	
    	# if doing subspace search, normalize descent direction
    	if lsTypeIsSO(lsType)
        	colNorm = norm(d,2)
        	if colNorm > 1e-4
                d = d./colNorm
            end
    	end
    	
    	# update directional derivative
    	gTd_prev = gTd
    	gTd = dot(g,d)
        if verbose
            normKonst = -1.0
            if konst != nothing
                normKonst = norm(konst,2)
            end
        	@printf("minFuncSO(%d): norm(w_prev,2)=%f, norm(w,2)=%f, gTd_prev= %.0f, gTd=%.0f,
            	 f_prev=%.4f, f=%.4f, gTg=%.0f, dTd=%.0f, norm(konst,2)=%f\n",
                nIter,norm(w_prev,2),norm(w,2),gTd_prev,gTd,f_prev,f,dot(g',g),dot(d',d),normKonst) 
        end
    	
        t_prev = t
    	t = getLSInitStep(lsInit,nIter,f,f_prev,gTd,verbose=verbose) # get initial step size
    	
    	Xd = fill(0.0, m)
    	if !nonOpt
        	Xd = X*d; nMatMult += 1
        end
    	w_prev = w
    	Xw_prev = Xw
	    f_prev = f
	    
    	w = w_prev + t*d # take step
    	Xw = Xw_prev + t*Xd
    	if nonOpt
        	f,_,nmm = funObjNoGrad(w,X)
    	else
            f,nmm = objFunc(Xw,w,konst=konst)
        end
        nMatMult += nmm
        nObjEvals += 1
        
        # update f values in memory and get reference f value (fref). fref could be largest in 
        #  recent history (getAndSetLSFref) or ever encountered (getAndSetLSFref2)
    	fref,frefInd = getAndSetLSFref2(nFref,oldFvals,nIter,f_prev)  	    	
    	    	 	    	    	
    	# set step sizes via line search / subspace search
    	lsFailed = false
    	if nIter==1 || lsType==0 # Armijo
        	if nonOpt
    	       (t,f,w,Xw,lsFailed,nLsIter,nOE,nGE,nMM) = 
                    lsArmijo(funObjNoGrad,funObj,X,Xw_prev,Xd,w_prev,t,c1,f_prev,fref,g,gTd,d,
                    verbose=verbose,maxLsIter=maxLsIter,nonOpt=true)
        	else
            	(t,f,w,Xw,lsFailed,nLsIter,nOE,nGE,nMM) = 
                    lsArmijo(objFunc,fPrimeFunc,X,Xw_prev,Xd,w_prev,t,c1,f_prev,fref,g,gTd,d,
                    konst=konst,verbose=verbose,maxLsIter=maxLsIter)
            end
        elseif lsType==1 # lsWolfe
            if nonOpt
                (t,f,w,Xw,lsFailed,nLsIter,nOE,nGE,nMM) = 
                    lsWolfe(funObjNoGrad,funObj,numDiff,X,Xw_prev,Xd,w_prev,c1,c2,f_prev,g,gTd,d,
                    verbose=verbose,maxLsIter=maxLsIter,nonOpt=true)
            else            
                (t,f,w,Xw,lsFailed,nLsIter,nOE,nGE,nMM) = 
                    lsWolfe(objFunc,fPrimeFunc,numDiff,X,Xw_prev,Xd,w_prev,c1,c2,f_prev,g,gTd,d,
                    konst=konst,verbose=verbose,maxLsIter=maxLsIter)
            end
        elseif lsType==6 # Lipschitz
            (t,f,w,Xw,lsFailed,nLsIter,nOE,nGE,nMM) = 
                lsLipschitz(objFunc,fPrimeFunc,X,Xw_prev,Xd,w_prev,t_prev,f_prev,fref,g,gTd,d,
                konst=konst,verbose=verbose,maxLsIter=maxLsIter)
        elseif lsTypeIsSO(lsType)
            # set up the matrix of directions in the case of subspace searches
            if lsType==8 || lsType==10
                k = 3
            elseif lsType==2 || lsType==3 || lsType==4
                k = min(momentumDirections,nIter-1) +1
            elseif lsType==5 || lsType==7 || lsType==9
                k = 2
            end
            D = zeros(n,k)
            XD = zeros(m,k)
            if lsType == 5
                D[:,1] = d
                XD[:,1] = Xd
                D[:,2] = dTN
                XD[:,2] = X*dTN; nMatMult += 1
            elseif lsType == 7
                D[:,1] = d
                XD[:,1] = Xd
                D[:,2] = dAdagrad
                XD[:,2] = X*dAdagrad; nMatMult += 1
            elseif lsType==8 # 3-term SO on extrapolated point
                D[:,1] = d
                XD[:,1] = Xd
                D[:,2] = DiffIterates[mod(nIter-2,lBfgsSize)+1]
                XD[:,2] = XDiffIterates[mod(nIter-2,lBfgsSize)+1]
                D[:,3] = DiffGrads[mod(nIter-2,lBfgsSize)+1]
                XD[:,3] = XDiffGrads[mod(nIter-2,lBfgsSize)+1]
            elseif lsType==9 || lsType==10
                D[:,1] = d
                XD[:,1] = Xd
                D[:,2] = -g
                XD[:,2] = -X*g; nMatMult += 1
                if lsType==10
                    D[:,3] = DiffIterates[mod(nIter-2,lBfgsSize)+1]
                    XD[:,3] = XDiffIterates[mod(nIter-2,lBfgsSize)+1]
                end
            else # lsType==2 || lsType==3 || lsType==4 (momentum as secondary directions)
                D[:,1] = d
                XD[:,1] = Xd
            
                for a in 2:k                     
                    D[:,a] = DiffIterates[mod(nIter-a,lBfgsSize)+1]
                    XD[:,a] = XDiffIterates[mod(nIter-a,lBfgsSize)+1]
                end
            end
        
            # solve subproblem
            if lsType==2 # call mdWolfe for subproblem
                if lsType == 2
                    fPrime,_,nMM = fPrimeFunc(Xw,w); nMatMult += nMM; nGradEvals += 1
                    gTdVec = transpose(fPrime'*XD)
                end

                if nonOpt
                    (ts,f,w,Xw,lsFailed,nLsIter,nOE,nGE,nMM) = mdWolfe(funObjNoGrad,funObj,numDiff,X,Xw_prev,
                        XD,w_prev,c1,c2,f,f_prev,g,gTdVec,D,verbose=verbose,maxLsIter=maxLsIter,nonOpt=nonOpt)
                else
                    (ts,f,w,Xw,lsFailed,nLsIter,nOE,nGE,nMM) = mdWolfe(objFunc,fPrimeFunc,numDiff,X,Xw_prev,
                        XD,w_prev,c1,c2,f,f_prev,g,gTdVec,D,konst=konst,verbose=verbose,maxLsIter=maxLsIter,
                        nonOpt=nonOpt)
                end
            elseif (3<=lsType==3 && lsType<=5) || (7<=lsType  && lsType<=10) # call minFunc for subproblem         
                if lsType==4
                    nonOpt = true
                    of = funObjNoGrad
                    ff = funObj
                    gf = funObj
                else
                    nonOpt = false
                    of = objFunc
                    ff = fPrimeFunc
                    gf = gradFunc
                end
                
                if verbose
                    @printf("  before calling lsDDir (nonOpt=%d): norm(w,2)=%f, norm(w_prev,2)=%f\n",
                        nonOpt,norm(w,2),norm(w_prev,2))
                end          
                (ts,f,w,Xw,lsFailed,nLsIter,nOE,nGE,nMM) =
                    lsDDir(of,ff,gf,XD,D,X,Xw_prev,w_prev,f_prev,g,nIter,gradNorm,gTd,c1,c2,
                    ssMethod=ssMethod,ssLS=ssLS,verbose=verbose,maxLsIter=maxLsIter,relativeStopping=ssRelStop,
                    oneDInit=ssOneDInit,nonOpt=nonOpt,funObjForT=funObjForT,fPrimePrimeFunc=fPrimePrimeFunc,
                    hessFunc=hessFunc)               
                if verbose
                    @printf("  after calling lsDDir: norm(w,2)=%f, norm(w_prev,2)=%f\n",
                        norm(w,2),norm(w_prev,2))
                    if lsFailed
                        @printf("method=%d, lsType=%d: lsFailed at nIter=%d\n",method,lsType,nIter)
                    end
                end
            end
        else
            if verbose
                @printf("Unknown line search type %d. Using Armijo backtrack.\n",lsType) 
            end
            if nonOpt
    	       (t,f,w,Xw,lsFailed,nLsIter,nOE,nGE,nMM) = 
                    lsArmijo(funObjNoGrad,funObj,X,Xw_prev,Xd,w_prev,t,c1,f_prev,fref,g,gTd,d,
                    verbose=verbose,maxLsIter=maxLsIter,nonOpt=true)
        	else
            	(t,f,w,Xw,lsFailed,nLsIter,nOE,nGE,nMM) = 
                    lsArmijo(objFunc,fPrimeFunc,X,Xw_prev,Xd,w_prev,t,c1,f_prev,fref,g,gTd,d,konst=konst,
                    verbose=verbose,maxLsIter=maxLsIter)
            end
    	end
    	totalLsIter += nLsIter
    	nObjEvals += nOE
    	nGradEvals += nGE
        nMatMult += nMM
        
        time_end = time_ns()
    	currTs = (time_end-time_start)/1.0e9
    	# CG iterations in truncated Newton counts towards total interations
    	if method==4 || lsType==5 || ssMethod==4
        	idx = nIter
        	for j in 1:cgIter
            	if idx < maxIter-1
                    idx += 1
                	fValues[idx,1] = f_prev
                    tValues[idx,1] = currTs
            	end	                         
            end            
            nIter = idx
    	end
    	fValues[nIter+1,1] = f
    	tValues[nIter+1,1] = currTs 
    	    	
    	# continue when line search fails
    	if lsFailed
        	if verbose
                @printf("Line search failed at %dth iter %dth lsIter, t=%f, method=%d, lsType=%d, f=%f.\n",
                    nIter,nLsIter,t,method,lsType,f)
                if lsTypeIsSO(lsType)
                    @printf("  lsType=%d, ts: %s\n",lsType,string(ts))
                end
            end
    	end
   	    
   	    # check for invalid f - revert back to last valid values and return
        if isfinitereal(fref) && !isfinitereal(f)
            if verbose
                @printf("method=%d, lsType=%d: invalid value at %dth iter. f_prev=%f, f=%f\n",
                    method,lsType,nIter,f_prev,f)
                if lsTypeIsSO(lsType)
                    @printf("  ts: %s\n",string(ts))
                end 
            end
            f = f_prev
            w = w_prev 
            if lsTypeIsSO(lsType)
                ts = ts .* 0.0
            end
            break
        end
   	    	
    	# gradient is only updated once after the line search
        g_prev = g 
        if nonOpt       
            if numDiff
                g,nmm = numGrad(funObjNoGrad,w,X)
                nObjEvals += 2*n            
            else
                _,g,nmm = funObj(w,X)
                nObjEvals += 1
                nGradEvals += 1
            end
        else
            if numDiff
                g,nmm = numGrad(objFunc,w,X,Xw,k=konst)
                nObjEvals += 2*n
            else
                g,nmm =gradFunc(Xw,w,X,konst=konst)
            	nGradEvals += 1
            end
        end
        nMatMult += nMM        

        H_prev = H
        # no nonOpt for norm2r_new
        if method==5 && fPrimePrimeFunc!=nothing && hessFunc!=nothing
            H,nMM = hessFunc(Xw,w,X,konst=konst)
        end
        nMatMult += nMM

        # update memory
        if verbose
            @printf("+++++ before updateDiffs, 1-norms: w_prev=%f, w=%f.\n",norm(w_prev,1),
                norm(w,1))
        end
        
        if lsTypeIsSO(lsType)
            if lsType==8
                nmm = updateDiffs(nIter,lBfgsSize,g_prev,g,w_prev,w,X,DiffIterates,DiffGrads,XDiffIterates,
                	XDiffGrads=XDiffGrads,normalizeColumns=true,calcXDiffIterates=true, calcXDiffGrads=true)
            else
            	nmm = updateDiffs(nIter,lBfgsSize,g_prev,g,w_prev,w,X,DiffIterates,DiffGrads,XDiffIterates,
                	normalizeColumns=true,calcXDiffIterates=true)
            end
        else
            nmm = updateDiffs(nIter,lBfgsSize,g_prev,g,w_prev,w,X,DiffIterates,DiffGrads,XDiffIterates)
        end
        
    	if verbose
        	j = mod(nIter - 1,lBfgsSize)+1
        	@printf("++++++ after updateDiffs: 1-norm of DiffIerates[%d]=%f.\n",j,
            	norm(DiffIterates[j],1))
    	end
        nMatMult += nMM

    	# check optimality conditions   	
    	gradNorm = norm(g,Inf)
        normD = norm(d,Inf)
    	if gradNorm < optTol 
        	if verbose
                @printf("method=%d, lsType=%d: Optimality conditions met at %dth iter. 
                    gradNorm=%f, f=%f\n",method,lsType,nIter,gradNorm,f)
                if lsTypeIsSO(lsType)
                    @printf("  ts: %s\n",string(ts))
                end    
            end
            break
        end
        
        # check for lack of progress on objective value
        diffF = fref - f
        if diffF < progTol 
            if verbose
                @printf("method=%d, lsType=%d: Stopped making progress met at %dth iter. 
                    fref=%.12f, f=%.12f, diff=%.12f\n",method,lsType,nIter,fref,f,diffF)
                if lsTypeIsSO(lsType)
                    @printf("  ts: %s\n",string(ts))
                end 
            end
            break
        end

        # check that step sizes have not become too small
        minT = t
        if lsTypeIsSO(lsType) && length(ts)>0
            minT = min(minT,maximum(ts))
        end
        if abs(minT) < 1e-12
            if verbose
                @printf("method=%d, lsType=%d, nIter=%d, t is too small: f=%.5f, fref=%.5f, t=%f\n",
                    method,lsType,nIter,f,fref,minT)
            end
            break
        end
    	      
        if currTs >= maxTimeInSec
            if verbose
                @printf("method=%d, lsType=%d, nIter=%d, time limit reached: f=%.5f, fref=%.5f, t=%f\n",
                    method,lsType,nIter,f,fref,minT)
            end
            break
        end
        
        if nFailedInv >= maxFailHessInv
            if verbose
                @printf("method=%d, lsType=%d, nIter=%d, max failed Hessian inversion reached: failed=%d, f=%.5f, fref=%.5f, t=%f\n",
                    method,lsType,nIter,nFailedInv,f,fref,minT)
            end
            break
        end
        nIter += 1  
	end
	
	if nIter>=maxIter
    	if verbose
            @printf("method=%d, lsType=%d, nIter=%d, max nIter reached: f=%.5f, fref=%.5f, t=%f\n",
                    method,lsType,nIter,f,fref,minT)
    	end         
	end
	
    diffIter = maxIter-nIter
    for j in 1:diffIter
        fValues[nIter+j,1] = NaN
        tValues[nIter+j,1] = NaN
    end  

    lambdaHMin = 0.0
    lambdaHMax = 0.0
    if eigDecomp && methodHasPrecond(method)
        try
            lambdasH = eigen(H).values
            try
                lambdaHMin = minimum(lambdasH)
                lambdaHMax = maximum(lambdasH)
                if verbose
                    @printf("Eigen decomp results in %d eigenvalues for method=%d, lsType=%d.\n",
                        length(lambdasH),method,lsType)
                end
            catch e
                if verbose
                    @printf("minimum/ maximum failed for method=%d, lsType=%d with error %s.\n",
                        method,lsType,e)
                end
            end
        catch e
            if verbose
                @printf("Eigen decomp failed for method=%d, lsType=%d with error %s.\n",
                    method,lsType,e)
            end
        end
    end

    if verbose && method==5
        @printf("n=%d: nFailedInv = %d out of %d\n",n,nFailedInv,nIter)
    end

	return (w,f,nObjEvals,nGradEvals,nIter,totalLsIter,nMatMult,fValues,tValues,gradNorm,gTd,
        normD,lambdaHMin,lambdaHMax,minT)
end