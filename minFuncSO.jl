using Printf, LinearAlgebra
include("misc.jl")
include("linesearch.jl")
include("descentDir.jl")

# optimized (for the number of matrix-vector multiplication) version of minFunc 
#  that also does subspace searches
function minFuncSO(objFunc,gradFunc,w0,X;konst=nothing,method=1,maxIter=100,maxLsIter=50,maxTimeInSec=600,optTol=1e-5,
 progTol=1e-9,lsInit=0,nFref=1,lsType=0,c1=1e-4,c2=0.9,lsInterpType=0,lBfgsSize=30,momentumDirections=0,
 ssMethod=0,ssLS=0,ssRelStop=true,ssOneDInit=true,derivativeCheck=false,numDiff=false,nonOpt=false,verbose=false,
 funObjForT=nothing)
    (m,n)=size(X)
	nObjEvals = 0
	nGradEvals = 0
	nIter = 1
	totalLsIter = 0
	nMatMult = 0

    time_start = time_ns() # Start timer
    fValues = zeros(maxIter,1) # fValues by iterations
    tValues = zeros(maxIter,1) # total time taken in seconds

    # solution, model, objective value, gradient
    # gradients get calculated once at the start and then once per iter outside LS loop	
    w = w0
    Xw = fill(0.0, m)
    Xw = X*w
    f = objFunc(Xw,konst=konst)
    fValues[1,1] = f
    tValues[1,1] = (time_ns()-time_start)/1.0e9
    
    if numDiff
        g = numGrad(objFunc,w,X,Xw,k=konst)
        nObjEvals += 2*n+1
        nMatMult += 1
    else
        g = X'*gradFunc(Xw,konst=konst)
        nGradEvals += 1
        nObjEvals += 1	
        nMatMult += 2
    end
	
	if derivativeCheck
    	if numDiff
        	@printf("Not doing derivative check with nummDiff on\n")
    	else
    		g2 = numGrad(objFunc,w,X,Xw,k=konst)
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
    	return (w0,f,nObjEvals,nGradEvals,nIter,totalLsIter,nMatMult,fValues,tValues)
    end
    
    if verbose
        @printf("minFuncLinear called with initial f=%f, initial norm(g,Inf)=%f\n",f,gradNorm)
    end
    
    # step size, descent direction, directional derivative
    t = 1.0
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

	lsDone = false 	
	alpha = 1.0 # for BB
	g_prev = g
	f_prev = f
	w_prev = w
	d_prev = d
	Xw_prev = Xw

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
            d = lbfgs(g,nIter,DiffIterates,DiffGrads)
        elseif method == 4 # Newton-CG
            d,cgIter = newtonCG(g,(maxIter-nIter-1),objFunc,gradFunc,numDiff,w,X,k=konst)
            if numDiff
                nObjEvals += 2*cgIter*n
                nMatMult += cgIter
            else
                nGradEvals += cgIter
                nMatMult += 2*cgIter
            end
    	end
    	
    	# calculate truncated Newton as a second direction. counts towards maxIters
    	if lsType==5
        	dTN,nIterTN = newtonCG(g,(maxIter-nIter-1),objFunc,gradFunc,numDiff,w,X,k=konst)
            if numDiff
                nObjEvals += 2*nIterTN*n
                nMatMult += nIterTN
            else
                nGradEvals += nIterTN
                nMatMult += 2*nIterTN
            end
            cgIter = cgIter + nIterTN
    	end
    	
    	# if doing subspace search, normalize descent direction
    	if lsType==3 || lsType==4 || lsType==5
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
        	@printf("minFuncLinear(%d): norm(w_prev,2)=%f, norm(w,2)=%f, gTd_prev= %.0f, gTd=%.0f,
            	 f_prev=%.4f, f=%.4f, gTg=%.0f, dTd=%.0f, norm(konst,2)=%f\n",
                nIter,norm(w_prev,2),norm(w,2),gTd_prev,gTd,f_prev,f,dot(g',g),dot(d',d),normKonst) 
        end
    	
    	t = getLSInitStep(lsInit,nIter,f,f_prev,gTd,verbose=verbose) # get initial step size
    	
    	Xd = X*d
    	nMatMult += 1
    	w_prev = w
    	Xw_prev = Xw
	    f_prev = f
    	
    	w = w_prev + t*d # take step
    	Xw = Xw_prev + t*Xd
        f = objFunc(Xw,konst=konst)
        nObjEvals += 1
        # update the list of most recent fVals and get back the largest f value stored in history
        #  (and its index into oldFVals)
    	fref,frefInd = getAndSetLSFref(nFref,oldFvals,nIter,f_prev)  	    	
    	    	 	    	    	
    	# line search / subspace search
    	lsFailed = false
	    nGE=0
    	if lsType==0
        	(t,f,w,Xw,lsFailed,nLsIter,nOE) = 
                lsArmijo(objFunc,X,Xw_prev,Xd,w_prev,t,c1,f,f_prev,fref,g,gTd,d,konst=konst,
                verbose=verbose,maxLsIter=maxLsIter)
        elseif lsType==1
            (t,f,w,Xw,lsFailed,nLsIter,nOE,nGE) = 
                lsWolfe(objFunc,gradFunc,numDiff,X,Xw_prev,Xd,w_prev,c1,c2,f,f_prev,g,gTd,d,
                konst=konst,verbose=verbose,maxLsIter=maxLsIter)
        elseif lsType==3 || lsType==4 || lsType==5
            # set up the matrix of directions in the case of subspace searches
            k = 2
            if lsType==3 || lsType==4
                k = min(momentumDirections,nIter-1) +1
            end
            D = zeros(n,k)
            XD = zeros(m,k)
            D[:,1] = d
            XD[:,1] = Xd
            for a in 2:k
                if lsType==3 || lsType==4
                    D[:,a] = DiffIterates[mod(nIter-2,lBfgsSize)+1]
                    XD[:,a] = XDiffIterates[mod(nIter-2,lBfgsSize)+1]
                elseif lsType==5
                    D[:,a] = dTN
                    XD[:,a] = X*dTN
                    nMatMult += 1
                end
            end
            
            if lsType==3 || lsType==5 # call subproblem solver that is optimized for linear structure
                if verbose
                    @printf("  before calling lsDDirLinear: norm(w,2)=%f, norm(w_prev,2)=%f\n",
                        norm(w,2),norm(w_prev,2))
                end          
                (ts,f,w,Xw,lsFailed,nLsIter,nOE,nGE,nMM) =
                    lsDDirLinear(objFunc,gradFunc,XD,D,X,Xw_prev,w_prev,f_prev,g,nIter,gradNorm,gTd,c1,c2,
                    ssMethod=ssMethod,ssLS=ssLS,verbose=verbose,maxLsIter=maxLsIter,relativeStopping=ssRelStop,
                    oneDInit=ssOneDInit)               
                if verbose
                    @printf("  after calling lsDDirLinear: norm(w,2)=%f, norm(w_prev,2)=%f\n",
                        norm(w,2),norm(w_prev,2))
                end
            elseif lsType==4 # call subproblem solver that is not optimized for linear structure
                if verbose
                    @printf("  before calling lsDDir: norm(w,2)=%f, norm(w_prev,2)=%f\n",norm(w,2),
                        norm(w_prev,2))
                end
                (ts,f,w,Xw,lsFailed,nLsIter,nOE,nGE,nMM) =
                    lsDDir(funObjForT,XD,D,X,Xw_prev,w_prev,f_prev,g,nIter,gradNorm,gTd,c1,c2,
                    ssMethod=ssMethod,ssLS=ssLS,verbose=verbose,maxLsIter=maxLsIter,relativeStopping=ssRelStop,
                    oneDInit=ssOneDInit)
                if verbose
                    @printf("  after calling lsDDir: norm(w,2)=%f, norm(w_prev,2)=%f\n",norm(w,2),
                        norm(w_prev,2))
                end            
            end
            nMatMult += nMM
        else
           @printf("Unknown line search type %d. Using Armijo backtrack.\n",lsType) 
           (t,f,w,Xw,lsFailed,nLsIter,nOE) = 
                lsArmijo(objFunc,X,Xw_prev,Xd,w_prev,t,c1,f,f_prev,fref,g,gTd,d,verbose=verbose,maxLsIter=maxLsIter)
    	end
    	totalLsIter += nLsIter
    	nObjEvals += nOE
    	nGradEvals += nGE
    	
    	currTs = (time_ns()-time_start)/1.0e9
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
                if lsType==3 || lsType==4 || lsType==5
                    @printf("  lsType=%d, ts: %s\n",lsType,string(ts))
                end
            end
    	end
   	    
   	    # check for invalid f - revert back to last valid values and return
        if isfinitereal(fref) && !isfinitereal(f)
            if verbose
                @printf("method=%d, lsType=%d: invalid value at %dth iter. f_prev=%f, f=%f\n",
                    method,lsType,nIter,f_prev,f)
                if lsType==3 || lsType==4 || lsType==5
                    @printf("  ts: %s\n",string(ts))
                end 
            end
            f = f_prev
            w = w_prev 
            if lsType==3 || lsType==4 || lsType==5
                ts = ts .* 0.0
            end
            break
        end
   	    	
    	# gradient is only updated once after the line search
        g_prev = g 
        if numDiff
            g = numGrad(objFunc,w,X,Xw,k=konst)
            nObjEvals += 2*n
        else
            g = X'*gradFunc(Xw,konst=konst)
        	nMatMult += 1
        	nGradEvals += 1
        end
        
        # update memory
        if verbose
            @printf("+++++ before updateDiffs, 1-norms: w_prev=%f, w=%f.\n",norm(w_prev,1),
                norm(w,1))
        end
        if lsType==3
        	updateDiffs(nIter,lBfgsSize,g_prev,g,w_prev,w,X,DiffIterates,DiffGrads,XDiffIterates,
            	normalizeColumns=true,calcXDiffIterates=true)
        	nMatMult += 1 
        else
            updateDiffs(nIter,lBfgsSize,g_prev,g,w_prev,w,X,DiffIterates,DiffGrads,XDiffIterates)
        end
    	if verbose
        	j = mod(nIter - 1,lBfgsSize)+1
        	@printf("++++++ after updateDiffs: 1-norm of DiffIerates[%d]=%f.\n",j,
            	norm(DiffIterates[j],1))
    	end

    	# check optimality conditions   	
    	gradNorm = norm(g,Inf)
    	if gradNorm < optTol 
        	if verbose
                @printf("method=%d, lsType=%d: Optimality conditions met at %dth iter. 
                    gradNorm=%f, f=%f\n",method,lsType,nIter,gradNorm,f)
                if lsType==3 || lsType==4 || lsType==5
                    @printf("  ts: %s\n",string(ts))
                end    
            end
            break
        end
        
        # check for lack of progress on objective value
        diffF = abs(fref-f)
        #if f > progTol
        #    diffF/= f
        #else
        #    diffF = 0
        #end
        if diffF < progTol
            if verbose
                @printf("method=%d, lsType=%d: Stopped making progress met at %dth iter. 
                    fref=%.12f, f=%.12f, diff=%.12f\n",method,lsType,nIter,fref,f,diffF)
                if lsType==3 || lsType==4 || lsType==5
                    @printf("  ts: %s\n",string(ts))
                end 
            end
            break
        end
    	
    	nIter += 1        
        if currTs >= maxTimeInSec
            break
        end        
	end
	
	if nIter>maxIter
    	if verbose
        	@printf("Maximum iterations (%d) reached\n",nIter)
    	end         
	end
    diffIter = maxIter-nIter
    for j in 1:diffIter-1
        fValues[nIter+1+j,1] = f
        tValues[nIter+1+j,1] = (time_ns()-time_start)/1.0e9
    end  

	return (w,f,nObjEvals,nGradEvals,nIter,totalLsIter,nMatMult,fValues,tValues)
end

# non-optimized version of minFunc (and no subspace search)
function minFuncNonOpt(funObj,funObjNoGrad,w0,X;method=1,maxIter=100,maxLsIter=50,maxTimeInSec=600,
 optTol=1e-5,progTol=1e-9,lsInit=0,nFref=1,lsType=0,c1=1e-4,c2=0.9,lsInterpType=0,
 lBfgsSize=30,derivativeCheck=false,numDiff=false,verbose=true)
    (m,n)=size(X)
    # metrics and counters
	nObjEvals = 0
	nGradEvals = 0
	nIter = 1
	totalLsIter = 0
	nMatMult = 0
	
	time_start = time_ns() # Start timer
    fValues = zeros(maxIter,1) # fValues by iterations
    tValues = zeros(maxIter,1) # total time taken in ms
    
    # solution, objective value, gradient
    w = w0
    f,g = funObj(w,X)
    fValues[1,1] = f
    tValues[1,1] = (time_ns()-time_start)/1.0e9
    
    if numDiff
        g = numGrad(funObj,w,X)
        nMatMult += 2*n+1
        nObjEvals += 2*n+1
    else
        nMatMult += 2
        nObjEvals += 1
    	nGradEvals += 1
    end
		
	if derivativeCheck
    	if numDiff
        	@printf("Not doing derivative check with nummDiff on\n")
    	else
    		g2 = numGrad(funObj,w0,X)
    		if maximum(abs.(g-g2)) > 1e-4
        		@show([g g2])
        		@printf("User and numerical derivatives differ\n")
    			sleep(1)
    		else
    			@printf("User and numerical derivatives agree\n")
    		end
    	end
	end
	
	# check initial point optimality
	gradNorm = norm(g,Inf)
	if gradNorm < optTol
    	return (w0,f,nObjEvals,nGradEvals,nIter,totalLsIter,nMatMult)
    end
    
    # step size, descent direction, directional derivative
    t = 1.0
    d = -g
    gTd = dot(g,d)

    # old function values for nonmonotonic Armijo
	oldFvals = fill(-Inf,nFref)	
    oldFvals[1]=f
    
    # lbfgs
    DiffIterates = [zeros(n,1) for _ in 1:lBfgsSize] # s_k's in NW
    DiffGrads = [zeros(n,1) for _ in 1:lBfgsSize] # y_k's in NW

	lsDone = false 	
	alpha = 1.0 # for BB
	g_prev = g
	f_prev = f
	w_prev = w
	d_prev = d
	t_prev = t

 	while nIter < maxIter   	
        # compute descent direction
        d_prev = d
        if method == 0 # steepest descent
            d = -g
    	elseif method == 1 # BB
        	d,alpha = bb(g,g_prev,w,w_prev,nIter,alpha)
        elseif method == 2 # CG
            d = cg(g,g_prev,nIter,d,optTol)
        elseif method == 3 # l-BFGS
            d = lbfgs(g,nIter,DiffIterates,DiffGrads)
        elseif method == 4 # Newton-CG (but uses linear structure gradFunc)
            d,cgIter = newtonCG(g,(maxIter-nIter),funObj,w,X,numDiff)
            nIter += cgIter
            if numDiff
                nObjEvals += 2*cgIter
                nMatMult += 2*cgIter
            else
                nGradEvals += cgIter
                nMatMult += 2*cgIter
            end
    	end
    	
    	gTd_prev = gTd
    	gTd = dot(g,d)
        if verbose
        	@printf("minFuncNonOpt(%d): gTd_prev= %f. gTd=%f\n",nIter,gTd_prev,gTd) 
        end
    	
    	t_prev = t
    	t = getLSInitStep(lsInit,nIter,f,f_prev,gTd) # get initial step size
    	
    	w_prev = w
    	w = w_prev + t*d  # take step
        f_prev = f
        g_prev = g
        
        f,g = funObj(w,X) # these are f_new and g_new
        if numDiff
            g = numGrad(funObj,w,X)
            nObjEvals += 2*n+1
            nMatMult += 2*n+1
        else
            nObjEvals += 1
            nGradEvals += 1
            nMatMult += 2
        end
        	 
    	# update the list of most recent fVals and get back the largest f value stored in history
        #  (and its index into oldFVals)
    	fref,frefInd = getAndSetLSFref(nFref,oldFvals,nIter,f_prev)  	    	    	    	
    	    	 	    	    	
    	# line search  	   	
	    lsFailed = false
    	if lsType==0
        	(t,f,w,lsFailed,nLsIter,nOE,nGE,nMM) = 
                lsArmijoNonOpt(funObjNoGrad,X,w_prev,t,c1,f,f_prev,fref,g,gTd_prev,d,
                verbose=verbose,maxLsIter=maxLsIter)
        elseif lsType==1
            (t,f,w,lsFailed,nLsIter,nOE,nGE,nMM) = 
                lsWolfeNonOpt(funObj,funObjNoGrad,numDiff,X,w_prev,c1,c2,f,f_prev,g,gTd_prev,d,
                verbose=verbose,maxLsIter=maxLsIter)
        else
            if verbose
                @printf("Unknown line search type %d. Using Armijo backtrack.\n",lsType) 
            end
            (t,f,w,lsFailed,nLsIter,nOE,nGE,nMM) = 
                lsArmijoNonOpt(funObjNoGrad,X,w_prev,t,c1,f,f_prev,fref,g,gTd_prev,d,
                verbose=verbose,maxLsIter=maxLsIter)
    	end
    	totalLsIter += nLsIter
    	nObjEvals += nOE
    	nGradEvals += nGE
    	nMatMult += nMM

        currTs = (time_ns()-time_start)/1.0e9
    	# CG iterations in truncated Newton counts towards total interations
    	if method==4
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

        #@show(size(g))
        #@show(size(g_prev))
        #@show(w)
        #@show(w_prev)
        #@show(size(w))
        #@show(size(w_prev))
    	updateDiffs(nIter,lBfgsSize,g_prev,g,w_prev,w,DiffIterates,DiffGrads)
    	
    	# line search failed 
    	if lsFailed
    	    if verbose
                @printf("Line search failed at %dth iter %dth lsIter, t=%f.\n",nIter,nLsIter,t)
            end
        	break
    	end

    	# check optimality conditions
    	gradNorm = norm(g,Inf)
    	if gradNorm < optTol
        	if verbose
                @printf("Optimality conditions met at %dth iter. gradNorm=%f\n",nIter,gradNorm)
            end
            break
        end
        
        # check for lack of progress
        diffF = abs(fref-f)
        #if f > progTol
        #    diffF/= f
        #else
        #    diffF = 0.0
        #end
        if diffF < progTol
            if verbose
                @printf("method=%d, lsType=%d: Stopped making progress met at %dth iter. fref=%f, f=%f\n",
                    method,lsType,nIter,fref,f)
            end
            break
        end
        
        nIter += 1  
        if currTs >= maxTimeInSec
            break
        end   
	end
	
	if nIter==maxIter
    	if verbose
        	@printf("Maximum iterations (%d) reached\n",nIter)
    	end
	end
	
	diffIter = maxIter-nIter
    for j in 1:diffIter-1
        fValues[nIter+1+j,1] = f
        tValues[nIter+1+j,1] = (time_ns()-time_start)/1.0e9
    end  
	
	return (w,f,nObjEvals,nGradEvals,nIter,totalLsIter,nMatMult,fValues,tValues)
end