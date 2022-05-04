# Load X and y variable
using JLD2, Printf, LinearAlgebra, Plots, MAT, DelimitedFiles,
    Plots.PlotMeasures, DataFrames, Tables, CSV, PyCall, ArgParse
include("minFuncSO.jl")
include("misc.jl")
#plotlyjs()
gr()

# global settings
progTol = 1e-9 # function value progress tolerance
optTol = 1e-9 # first order optimality tolerance
c1= 1e-4 # parameter for Armijo/ sufficient decrease condition
c2=0.9 # parameter for Wolfe/ curvature condition
verbose = false # print debug
savePlotData = true
getFStar = true
fStarMethods = [3, 0]
fStarLsTypes = [1, 3]
fStarSsMethods = [1, 1] 
dimNewtonCutoff = 500
datasetNamesFromCmd = []
dsType = ""
objLabel = ""
lbfgsCautious = false # if true, then skips correction pairs with negative inner products
eigDecomp = false

y=nothing

# function we're going to minimize optimized for linear composition problem
objFunc(z,w;konst=nothing) = objLinear(z,w,y,k=konst)
fPrimeFunc(z,w;konst=nothing) = fPrimeLinear(z,w,y,k=konst)
gradFunc(z,w,X;konst=nothing) = gradLinear(z,w,X,y,k=konst)
fPrimePrimeFunc(z,w,X;konst=nothing) = doublePrimeLinear(z,w,X,y,k=konst)
hessFunc(z,w,X;konst=nothing) = hessianLinear(z,w,X,y,k=konst)

# function and gradient wrt step size for line search that calls minFunc
funObjForT(t,D,w,X) = objAndGrad(t,D,w,X,y)

# function and gradient for minFuncNonOpt
funObjForNonOpt(w,X) = objAndGrad(w,X,y)

# function and no gradient evaluation for minFuncNonOpt/ lsArmijoNonOpt
funObjNoGradForNonOpt(w,X) = objAndGrad(w,X,y)

function parseCmdLine()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--ds"
            help = "a comma-separated list of datasets to run"
            arg_type = String
            default = ""
        "--linReg"
            help = "set objective to linear regression"
            action = :store_true
        "--linRegLHalf"
            help = "set objective to linear regression using a p=1/2 norm"
            action = :store_true
        "--logReg"
            help = "set objective to logistic regression"
            action = :store_true
        "--logRegL2"
            help = "set objective to logistic regression with L2 regularization"
            action = :store_true
        "--hh"
            help = "set objective to huberized hinge"
            action = :store_true
        #"arg1"
        #    help = "a positional argument"
        #    required = true
    end

    return parse_args(s)
end

# adds data from a run to stuff to plot
function addToPlot(methodLabel,lsLabel,fValuesPlot,fValuesPlotLabels,fValues)
    if sum(abs.(fValuesPlot[methodLabel]))==0.0
        fValuesPlot[methodLabel].=fValues
        fValuesPlotLabels[methodLabel].=[lsLabel]
    else
        fValuesPlot[methodLabel]=[fValuesPlot[methodLabel] fValues]
        fValuesPlotLabels[methodLabel]=[fValuesPlotLabels[methodLabel] lsLabel]
    end
end

parsed_args = parseCmdLine()
for (arg,val) in parsed_args
    if arg=="ds"
        global datasetNamesFromCmd = split(val,",")
    elseif arg=="linReg" && val
        include("linReg.jl")
        global dsType = "Regression"
        global objLabel = "linReg"
    elseif arg=="linRegLHalf" && val
        include("linRegLHalf.jl")
        global dsType = "Regression"
        global objLabel = "linRegLHalf"
    elseif arg=="logReg" && val
        include("logReg.jl")
        global dsType = "Classification"
        global objLabel = "logReg"
    elseif arg=="logRegL2" && val
        include("logRegL2.jl")
        global dsType = "Classification"
        global objLabel = "logRegL2"
    elseif arg=="hh" && val
        include("huberHinge.jl")
        global dsType = "Classification"
        global objLabel = "hh"
    end
end

# List of datasets
dsdl = pyimport("dsdl")
if length(datasetNamesFromCmd[1])>0
    datasetNames = datasetNamesFromCmd
else
    datasetNames = dsdl.available_datasets()
    # excluding some very big datasets or ones with format that code does not handle
    dsToExclude = ["E2006-log1p","E2006-E2006-tfidf","boston-housing","power-plant",
        "news20.binary","rcv1.binary","real-sim","digits"]
    for ds in dsToExclude
        deleteat!(datasetNames, findall(x->x==ds,datasetNames))
    end
end

# TODO: sparse matrices needed to get this working for larger ds
nDatasets=size(datasetNames,1)
datasets = [] # holds configuration for each dataset (such as max number of iterations to fit)

# read in settings for calls to minFuncSO 
configDF = CSV.read("./config/fRef5.csv",DataFrame)
settingNames = DataFrames.names(configDF)
nSettings = DataFrames.nrow(configDF)

# holds summary of all datasets in one place
allDScsvFileName = string("./output/compareDS_dsdl.",objLabel,".csv")
if isfile(allDScsvFileName)
    allDScsvIO = open(allDScsvFileName,"a")
else
    allDScsvIO = open(allDScsvFileName,"w")
    @printf(allDScsvIO,"%s,%s,%s,%s,%s,%s,%s,%s\n",
    "Dataset","nDataPoints","nFeatures","Method","ObjVal","nIter","nLSIter","time(s)")
end

# filter datasets
for a=1:nDatasets
    ds = dsdl.load(datasetNames[a]) # load ds before filtering because we need to load before knowing characteristics
    if ds.task != dsType
        if verbose
            @printf("%s not the right ds task - skipping\n",datasetNames[a])
        end
        continue
    end
    currX, currY = ds.get_train()
    #currX, currY = ds.get_test()
    #if currX==nothing
    #    currX, currY = ds.get_train()
    #end

    if currX isa Array
        local X = currX
    else
        local X = currX.todense()
    end
    local y = currY[:,1]
    
    (m,n) = size(X)

    # this is arbitrary but couldn't think of another way to get something similar
    maxPasses=100
    maxTimeInSec=10
    if m > 5e4 || n > 5e4
        maxPasses = 500
        maxTimeInSec=60
    elseif m > 3e4 || n > 3e4
        maxPasses = 300
        maxTimeInSec=30
    elseif m > 1e4 || n > 1e4
        maxPasses = 100
        maxTimeInSec = 10
    else
        maxPasses = 50
        maxTimeInSec = 5
    end

    push!(datasets,(label=datasetNames[a],maxPasses=maxPasses,maxLsIter=25,maxTimeInSec=maxTimeInSec,
        X=X,y=y,m=m,n=n))
end
nDatasets = size(datasets,1)

# Run experiments
for a=1:nDatasets
    @printf("Dataset %d: %s. m=%d, n=%d, maxPasses=%d, maxTimeInSec=%d\n",
        a,datasets[a].label,datasets[a].m,datasets[a].n,datasets[a].maxPasses,datasets[a].maxTimeInSec)

    global X = datasets[a].X
    global y = datasets[a].y

    w0 = zeros(datasets[a].n,1) # initial solution
    fInit,_ = objFunc(X*w0,w0) # used for plotting

    # holds data for plots
    fValuesIterPlot = Dict(methodToLabel(b) => zeros(datasets[a].maxPasses,1) for b = 0:6)
    fValuesIterPlotLabels = Dict(methodToLabel(b) => fill("",1,1) for b = 0:6)
    fValuesClockPlot = Dict(methodToLabel(b) => zeros(datasets[a].maxPasses,1) for b = 0:6)
    fValuesClockPlotLabels = Dict(methodToLabel(b) => fill("",1,1) for b = 0:6)

    XXT = nothing
    #if 6 in configDF.lsType # need to calculate XX'
    #    XXT = X*X'
    #end

    # if getFStar is set to true, then run settings specified in fStarMethods and use the lowest objective value
    #  attained as `fStar'
    minFStars = 1e15 
    if getFStar
        fStars = []
        for b in 1:length(fStarMethods)
            if verbose
                @printf("getFStar(%d): running method=%d, lsType=%d, ssMethod=%d\n",
                    b,fStarMethods[b],fStarLsTypes[b],fStarSsMethods[b])
            end
            getFStarMult = 5
            (w,f,nObjEvals,nGradEvals,nIter,nLsIter,nMatMult,fValues,tValues,gradNorm,gTd,normD,minLambdaH,maxLambdaH,minT) = 
                minFuncSO(objFunc,fPrimeFunc,gradFunc,
                w0,X,method=fStarMethods[b],maxIter=datasets[a].maxPasses*getFStarMult,maxLsIter=datasets[a].maxLsIter,
                maxTimeInSec=datasets[a].maxTimeInSec*getFStarMult,optTol=optTol,progTol=progTol,lsInit=1,
                nFref=1,lsType=fStarLsTypes[b],c1=c1,c2=c2,lsInterpType=1,lBfgsSize=20,
                momentumDirections=1,ssMethod=fStarSsMethods[b],ssLS=0,ssRelStop=true,ssOneDInit=true,
                derivativeCheck=false,numDiff=false,verbose=false,funObjForT=funObjForT,
                funObj=funObjForNonOpt,funObjNoGrad=funObjNoGradForNonOpt,nonOpt=false,XXT=XXT,
                fPrimePrimeFunc=fPrimePrimeFunc,hessFunc=hessFunc,lbfgsCautious=lbfgsCautious,eigDecomp=false)
            push!(fStars,f)
        end
        minFStars = minimum(fStars)
    end

    # Run optimizer 
    runResults = []
    for b=1:nSettings    
        if verbose
            @printf("Running setting %d: method=%d, lsType=%d, ssMethod=%d, nonOpt=%d\n",
                b,configDF[b,:].method,configDF[b,:].lsType,configDF[b,:].ssMethod,configDF[b,:].nonOpt)
        end

        if configDF[b,:].method==5 && datasets[a].n>=dimNewtonCutoff
            push!(runResults,(fStar=9999,nObjEvals=0,nGradEvals=0,nIter=0,nLSIter=0,nMatMult=0,nSeconds=0.0,gradNorm=NaN,
                gTd=NaN,normD=NaN,minLambdaH=NaN,maxLambdaH=NaN,stepSize=minT))
            continue
        end
        
        time_start = time_ns() # Start timer

        (w,f,nObjEvals,nGradEvals,nIter,nLsIter,nMatMult,fValues,tValues,gradNorm,gTd,normD,minLambdaH,maxLambdaH,minT) = 
            minFuncSO(objFunc,fPrimeFunc,gradFunc,
            w0,X,method=configDF[b,:].method,maxIter=datasets[a].maxPasses,maxLsIter=datasets[a].maxLsIter,
            maxTimeInSec=datasets[a].maxTimeInSec,optTol=optTol,progTol=progTol,lsInit=configDF[b,:].lsInit,
            nFref=configDF[b,:].nFref,lsType=configDF[b,:].lsType,
            c1=c1,c2=c2,lsInterpType=configDF[b,:].lsInterp,lBfgsSize=configDF[b,:].lbfgsSize,
            momentumDirections=configDF[b,:].momentumDirs,ssMethod=configDF[b,:].ssMethod,
            ssLS=configDF[b,:].ssLS,ssRelStop=convert(Bool,configDF[b,:].ssRelStop),
            ssOneDInit=convert(Bool,configDF[b,:].ssOneDInit),derivativeCheck=convert(Bool,configDF[b,:].derivCheck),
            numDiff=convert(Bool,configDF[b,:].numDiff),verbose=verbose,funObjForT=funObjForT,
            funObj=funObjForNonOpt,funObjNoGrad=funObjNoGradForNonOpt,nonOpt=convert(Bool,configDF[b,:].nonOpt),XXT=XXT,
            fPrimePrimeFunc=fPrimePrimeFunc,hessFunc=hessFunc,lbfgsCautious=lbfgsCautious,eigDecomp=eigDecomp)
        nSeconds = (time_ns()-time_start)/1.0e9 # End timer
        
        if verbose
            @printf("f=%f, %f seconds, %d calls to f, %d calls to g, %d iters, %d LS iters, %d matrix-vector mults\n**********\n", 
                f,nSeconds,nObjEvals,nGradEvals,nIter,nLsIter,nMatMult) 
        end

        diverge = false
        if getFStar && abs(f)/1e2 > minFStars
            diverge = true
        end
        
        push!(runResults,(fStar=f,nObjEvals=nObjEvals,nGradEvals=nGradEvals,nIter=nIter,nLSIter=nLsIter,
            nMatMult=nMatMult,nSeconds=nSeconds,diverge=diverge,gradNorm=gradNorm,gTd=gTd,normD=normD,
            minLambdaH=minLambdaH,maxLambdaH=maxLambdaH,stepSize=minT))
        
        methodLabel=methodToLabel(configDF[b,:].method)
        lsLabel=lsTypeToLabel(configDF[b,:].lsType,configDF[b,:].lsInterp,configDF[b,:].momentumDirs,
            configDF[b,:].ssMethod,configDF[b,:].ssLS,configDF[b,:].nonOpt)
        if !diverge
            addToPlot(methodLabel,lsLabel,fValuesIterPlot,fValuesIterPlotLabels,fValues)
            addToPlot(methodLabel,lsLabel,fValuesClockPlot,fValuesClockPlotLabels,tValues)
        end
        
        @printf(allDScsvIO,"%s,%d,%d,%s-%s,%f,%d,%d,%f\n",
            datasets[a].label,datasets[a].m,datasets[a].n,methodLabel,lsLabel,f,nIter,nLsIter,nSeconds)
    end
    bestMethodF = minimum(runResults[b].fStar for b=1:nSettings)
    worstMethodF = maximum(runResults[b].fStar for b=1:nSettings)

    ##### print and save results #####
    csvFileName = string("./output/summary_",datasets[a].label,".",objLabel,".csv")
    txtFileName = string("./output/summary_",datasets[a].label,".",objLabel,".txt")
    open(csvFileName,"w") do csvIo; open(txtFileName,"w") do txtIo
        @printf("%-5s%-10s%-15s%-5s%-20s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-5s%-10s%-10s%-10s%-10s%-10s%-10s\n",
            "","Method","LStype","nMom","objVal","nIter","nLSIter","nMatVec","nObjEval","nGradEval",
            "time(s)","LSinit","LSinterp","Fref","normG","gTd","normD","minEigH","maxEigH","stepSize")
        @printf(csvIo,"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
            "Method","LStype","nMom","objVal","nIter","nLSIter","nMatMult","nObjEval","nGradEval",
            "time(s)","LSinit","LSinterp","Fref","normG","gTd","normD","minEigH","maxEigH","stepSize")
        @printf(txtIo,"%-10s%-15s%-5s%-20s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-5s%-10s%-10s%-10s%-10s%-10s%-10s\n",
            "Method","LStype","nMom","objVal","nIter","nLSIter","nMatMult","nObjEval","nGradEval",
            "time(s)","LSinit","LSinterp","Fref","normG","gTd","normD","minEigH","maxEigH","stepSize")
        @printf("%-5s%-10s%-15s%-5s%-20s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-5s%-10s%-10s%-10s%-10s%-10s%-10s\n",
            "","------","------","----","------","-----","-------","-------","--------","---------",
            "-------","------","--------","----","-----","---","-----","-------","-------","--------")
        @printf(txtIo,"%-10s%-15s%-5s%-20s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-5s%-10s%-10s%-10s%-10s%-10s%-10s\n",
            "------","------","----","------","-----","-------","--------","--------","---------",
            "-------","------","--------","----","-----","---","-----","-------","-------","--------")
        
        for b in 1:nSettings
            if configDF[b,:].method==5 && datasets[a].n>=dimNewtonCutoff
                continue
            end

            methodLabel=methodToLabel(configDF[b,:].method)
            lsLabel=lsTypeToLabel(configDF[b,:].lsType,configDF[b,:].lsInterp,configDF[b,:].momentumDirs,
                configDF[b,:].ssMethod,configDF[b,:].ssLS,configDF[b,:].nonOpt)

            if runResults[b].diverge
                @printf("%-5d%-10s%-15s%-5d%-20s%-10d%-10d%-10d%-10d%-10d%-10.2f%-10s%-10s%-5s%-10s%-10s%-10s%-10s%-10s%-10s\n",
                    a,methodLabel,lsLabel,configDF[b,:].momentumDirs,"Inf",runResults[b].nIter,
                    runResults[b].nLSIter,runResults[b].nMatMult,runResults[b].nObjEvals,runResults[b].nGradEvals,
                    runResults[b].nSeconds,configDF[b,:].lsInit,configDF[b,:].lsInterp,configDF[b,:].nFref,
                    "","","","","","")
                @printf(csvIo,"%s,%s,%d,%s,%d,%d,%d,%d,%d,%f,%s,%s,%s,%f,%f,%f,%f,%f,%f\n",
                    methodLabel,lsLabel,configDF[b,:].momentumDirs,"Inf",runResults[b].nIter,
                    runResults[b].nLSIter,runResults[b].nMatMult,runResults[b].nObjEvals,runResults[b].nGradEvals,
                    runResults[b].nSeconds,configDF[b,:].lsInit,configDF[b,:].lsInterp,configDF[b,:].nFref,
                    runResults[b].gradNorm,runResults[b].gTd,runResults[b].normD,runResults[b].minLambdaH,
                    runResults[b].maxLambdaH,runResults[b].stepSize)
                @printf(txtIo,"%-10s%-15s%-5d%-20s%-10d%-10d%-10d%-10d%-10d%-10.2f%-10s%-10s%-5s%-10s%-10s%-10s%-10s%-10s%-10s\n",
                    methodLabel,lsLabel,configDF[b,:].momentumDirs,"Inf",runResults[b].nIter,
                    runResults[b].nLSIter,runResults[b].nMatMult,runResults[b].nObjEvals,runResults[b].nGradEvals,
                    runResults[b].nSeconds,configDF[b,:].lsInit,configDF[b,:].lsInterp,configDF[b,:].nFref,
                    "","","","","","")      
            else
                @printf("%-5d%-10s%-15s%-5d%-20.5f%-10d%-10d%-10d%-10d%-10d%-10.2f%-10s%-10s%-5s%-10.2e%-10.2e%-10.2e%-10.2e%-10.2e%-10.2e\n",
                    a,methodLabel,lsLabel,configDF[b,:].momentumDirs,runResults[b].fStar,runResults[b].nIter,
                    runResults[b].nLSIter,runResults[b].nMatMult,runResults[b].nObjEvals,runResults[b].nGradEvals,
                    runResults[b].nSeconds,configDF[b,:].lsInit,configDF[b,:].lsInterp,configDF[b,:].nFref,
                    runResults[b].gradNorm,runResults[b].gTd,runResults[b].normD,runResults[b].minLambdaH,
                    runResults[b].maxLambdaH,runResults[b].stepSize)
                @printf(csvIo,"%s,%s,%d,%f,%d,%d,%d,%d,%d,%f,%s,%s,%s,%f,%f,%f,%f,%f,%f\n",
                    methodLabel,lsLabel,configDF[b,:].momentumDirs,runResults[b].fStar,runResults[b].nIter,
                    runResults[b].nLSIter,runResults[b].nMatMult,runResults[b].nObjEvals,runResults[b].nGradEvals,
                    runResults[b].nSeconds,configDF[b,:].lsInit,configDF[b,:].lsInterp,configDF[b,:].nFref,
                    runResults[b].gradNorm,runResults[b].gTd,runResults[b].normD,runResults[b].minLambdaH,
                    runResults[b].maxLambdaH,runResults[b].stepSize)
                @printf(txtIo,"%-10s%-15s%-5d%-20.5f%-10d%-10d%-10d%-10d%-10d%-10.2f%-10s%-10s%-5s%-10.2e%-10.2e-%-10.2e%-10.2e%-10.2e%-10.2e\n",
                    methodLabel,lsLabel,configDF[b,:].momentumDirs,runResults[b].fStar,runResults[b].nIter,
                    runResults[b].nLSIter,runResults[b].nMatMult,runResults[b].nObjEvals,runResults[b].nGradEvals,
                    runResults[b].nSeconds,configDF[b,:].lsInit,configDF[b,:].lsInterp,configDF[b,:].nFref,
                    runResults[b].gradNorm,runResults[b].gTd,runResults[b].normD,runResults[b].minLambdaH,
                    runResults[b].maxLambdaH,runResults[b].stepSize)      
            end  
            if b<nSettings && configDF[b,:].method != configDF[b+1,:].method
                @printf("------------------------------------------------------------------------------------------------------------------------\n")
                @printf(txtIo,"------------------------------------------------------------------------------------------------------------------------\n")            
            end
        end
    end; end
    @printf("Csv summary file saved as %s\n",csvFileName)
    @printf("Text summary file saved as %s\n",txtFileName)

    ##### plot results #####
    x = 1:datasets[a].maxPasses   
    #methodsToPlot = [0 2 3 4 ] # skip BB because same search direction as GD
    methodsToPlot = [0 3 5 ] # GD vs quasi-Newton vs Newton
    subplotByIter = Dict(methodToLabel(b) => [] for b in methodsToPlot)
    subplotByTime = Dict(methodToLabel(b) => [] for b in methodsToPlot)
    if getFStar
        fmin = minFStars
    else
        fmin = bestMethodF
    end
    
    if savePlotData
        detailsFileName = string("./output/details_",datasets[a].label,".",objLabel,".csv")
        detailsDf = DataFrame(xLabel=x)
    end
    
    #=
    upperLim = 1.
    lowerLim = 0.
    if isfinitereal(fInit)
        if isfinitereal(fmin)
            upperLim = log10(abs(fInit-fmin))
            upperLim = floor(Int,upperLim)
        end
        if isfinitereal(bestMethodF)
            lowerLim = log10(abs(bestMethodF-fmin))
            lowerLim = floor(Int,lowerLim)
            lowerLim = maximum([0,lowerLim])
        end
    end 
    upperLim = 10. ^ upperLim
    lowerLim = 10. ^ lowerLim
    currYLims = (lowerLim,upperLim)
    =#
    

    upperYLim = floor(Int,fInit - bestMethodF)
    currYLims = (0,upperYLim) 
    @show(currYLims)
    @printf("fInit=%f,minFStars=%f,bestMethodF=%f,worstMethodF=%f,fmin=%f\n",
        fInit,minFStars,bestMethodF,worstMethodF,fmin)

    for b in methodsToPlot
        methodLabel=methodToLabel(b)
        fValuesIterPlot[methodLabel]=fValuesIterPlot[methodLabel].-fmin.+1e-9
        if savePlotData
            nLS=size(fValuesIterPlotLabels[methodLabel],2)
            for c in 1:nLS
                fValuesColLabel = string("fValue_",methodLabel,"_",fValuesIterPlotLabels[methodLabel][:,c])
                clockColLabel = string("clockValue_",methodLabel,"_",fValuesIterPlotLabels[methodLabel][:,c])
                insertcols!(detailsDf, fValuesColLabel=>fValuesIterPlot[methodLabel][:,c],
                clockColLabel=>fValuesClockPlot[methodLabel][:,c])
            end
        end
        #=
        push!(subplotByIter[methodLabel],plot(x,fValuesIterPlot[methodLabel],label=fValuesIterPlotLabels[methodLabel],
            title=methodLabel,titlefont=font(8),xlabel="iterations",ylabel="optimality gap",
            legend=:outertopright,legendtitle="   Search Type",legendtitlefonthalign=:left,legendtitlefontsize=5,
            legendfontsize=5,guidefontsize=5,tickfontsize=5,yaxis=:log,ylims=currYLims,margin=5mm)) 
        push!(subplotByTime[methodLabel],plot(fValuesClockPlot[methodLabel],fValuesIterPlot[methodLabel],
            label=fValuesClockPlotLabels[methodLabel],title=methodLabel,titlefont=font(8),xlabel="total time (sec)",
            ylabel="optimality gap",legend=:outertopright,legendtitle="   Search Type",legendtitlefonthalign=:left,
            legendtitlefontsize=5,legendfontsize=5,guidefontsize=5,tickfontsize=5,yaxis=:log,ylims=currYLims,margin=5mm))
        =#
        
        push!(subplotByIter[methodLabel],plot(x,fValuesIterPlot[methodLabel],label=fValuesIterPlotLabels[methodLabel],
            title=methodLabel,titlefont=font(8),xlabel="iterations",ylabel="optimality gap",
            legend=:outertopright,legendtitle="   Search Type",legendtitlefonthalign=:left,legendtitlefontsize=5,
            legendfontsize=5,guidefontsize=5,tickfontsize=5,ylims=currYLims,margin=5mm)) 
        push!(subplotByTime[methodLabel],plot(fValuesClockPlot[methodLabel],fValuesIterPlot[methodLabel],
            label=fValuesClockPlotLabels[methodLabel],title=methodLabel,titlefont=font(8),xlabel="total time (sec)",
            ylabel="optimality gap",legend=:outertopright,legendtitle="   Search Type",legendtitlefonthalign=:left,
            legendtitlefontsize=5,legendfontsize=5,guidefontsize=5,tickfontsize=5,ylims=currYLims,margin=5mm))
    end
    
    if savePlotData
        CSV.write(detailsFileName,detailsDf)
        @printf("Details file saved as %s\n",detailsFileName)
    end
    
    # not sure how to do this nicely. This is hardcoded the same way as methodsToPlot list above.
    #fValPlot = plot(subplotByIter[methodToLabel(0)][1],subplotByTime[methodToLabel(0)][1],subplotByIter[methodToLabel(2)][1],
    #    subplotByTime[methodToLabel(2)][1],subplotByIter[methodToLabel(3)][1],subplotByTime[methodToLabel(3)][1],
    #    subplotByIter[methodToLabel(5)][1],subplotByTime[methodToLabel(5)][1],size=(1200,1000),layout=(4,2));

    fValPlot = plot(subplotByIter[methodToLabel(0)][1],subplotByTime[methodToLabel(0)][1],subplotByIter[methodToLabel(3)][1],
        subplotByTime[methodToLabel(3)][1],subplotByIter[methodToLabel(5)][1],subplotByTime[methodToLabel(5)][1],
        size=(1200,1000),layout=(3,2));
    
    plotName=string("./output/plot_",datasets[a].label,".",objLabel,".png")
    savefig(fValPlot,plotName)
    @printf("Plot saved as %s\n",plotName)
end  

close(allDScsvIO)