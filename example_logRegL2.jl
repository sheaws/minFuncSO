# Load X and y variable
using JLD2, Printf, LinearAlgebra, Plots, MAT, DelimitedFiles,
    Plots.PlotMeasures, DataFrames, Tables, CSV, PyCall
include("minFuncSO.jl")
include("logRegL2.jl")
#plotlyjs()
gr()

# global settings
progTol = 1e-9 # function value progress tolerance
optTol = 1e-9 # first order optimality tolerance
c1= 1e-4 # parameter for Armijo/ sufficient decrease condition
c2=0.9 # parameter for Wolfe/ curvature condition
verbose = false # print more output
savePlotData = true
y=nothing

# function we're going to minimize optimized for linear composition problem
objFunc(z,w;konst=nothing) = logisticL2ObjLinear(z,w,y,k=konst)
fPrimeFunc(z,w;konst=nothing) = logisticL2FPrimeLinear(z,w,y,k=konst)
gradFunc(z,w,X;konst=nothing) = logisticL2GradLinear(z,w,X,y,k=konst)

# function and gradient wrt step size for line search that calls minFunc
funObjForT(t,D,w,X) = logisticL2ObjAndGrad(t,D,w,X,y)

# function and gradient for minFuncNonOpt
funObjForNonOpt(w,X) = logisticL2ObjAndGrad(w,X,y)

# function and no gradient evaluation for minFuncNonOpt/ lsArmijoNonOpt
funObjNoGradForNonOpt(w,X) = logisticL2ObjAndNoGrad(w,X,y)

# returns a string description of iterative method
function methodToLabel(method)
    methodLabel=""
    if method==0
        methodLabel="GD"
    elseif method==1
        methodLabel="BB"
    elseif method==2
        methodLabel="CG"
    elseif method==3
        methodLabel="lBFGS"
    elseif method==4
        methodLabel="TN"
    end
    return methodLabel
end

function lsTypeToLabel(lsType)
    lsLabel=""
    if lsType==0
        lsLabel = "Arm"
    elseif lsType==1
        lsLabel = "Wol"
    end
    return lsLabel
end

# returns a string description of line/ subspace search 
function lsTypeToLabel(lsType,nMomDirs,ssMethod,ssLS,nonOpt)
    lsLabel=""
    if lsType==0
        lsLabel = "Armijo"
    elseif lsType==1
        lsLabel = "Wolfe"
    elseif lsType==3 || lsType==4
        lsLabel="2"
        if nMomDirs!=0
            lsLabel=string(nMomDirs+1)
        end
        lsLabel=string(lsLabel,"d-Mm")
    elseif lsType==5
        lsLabel = "2d-TN"
    end
    
    if 3<=lsType && lsType <= 5
        lsLabel=string(lsLabel,"-",methodToLabel(ssMethod),"-",
        lsTypeToLabel(ssLS))
    end
    
    if lsType==4 || nonOpt==1
        lsLabel=string(lsLabel,"-")
    end
    
    return lsLabel
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

# List of datasets
dsdl = pyimport("dsdl")
datasetNames = dsdl.available_datasets()
# excluding some very big datasets or ones with format that code does not handle
dsToExclude = ["news20.binary","rcv1.binary","real-sim","boston-housing","energy",
    "power-plant","digits"]
for ds in dsToExclude
    deleteat!(datasetNames, findall(x->x==ds,datasetNames))
end

nDatasets=size(datasetNames,1)
datasets = [] # holds configuration for each dataset (such as max number of iterations to fit)

# read in settings for calls to minFuncSO 
configDF = CSV.read("./config/minFuncSO_smallTest.csv",DataFrame)
settingNames = DataFrames.names(configDF)
nSettings = DataFrames.nrow(configDF)

# holds summary of all datasets in one place
allDScsvFileName = "./output/compareDS_dsdl.csv"
allDScsvIO = open(allDScsvFileName,"w")
@printf(allDScsvIO,"%s,%s,%s,%s,%s,%s,%s,%s\n",
    "Dataset","nDataPoints","nFeatures","Method","ObjVal","nIter","nLSIter","time(s)")

# Run experiments
for a=1:nDatasets
    @printf("Dataset %d: %s\n",a,datasetNames[a])
    
    push!(datasets,(label=datasetNames[a],maxPasses=500,maxLsIter=25,maxTimeInSec=60))
    
    ds = dsdl.load(datasetNames[a])
    if ds.task != "Classification"
        @printf("   not classification - skipping\n")
        continue
    end
    currX, currY = ds.get_test()
    if currX==nothing
        currX, currY = ds.get_train()
    end
    global X = currX.toarray()
    global y = currY[:,1]
    
    (m,n) = size(X)
    w0 = zeros(n,1) # initial solution
    fInit,_ = objFunc(X*w0,w0) # used for plotting

    # holds data for plots
    fValuesIterPlot = Dict(methodToLabel(b) => zeros(datasets[a].maxPasses,1) for b = 0:4)
    fValuesIterPlotLabels = Dict(methodToLabel(b) => fill("",1,1) for b = 0:4)
    fValuesClockPlot = Dict(methodToLabel(b) => zeros(datasets[a].maxPasses,1) for b = 0:4)
    fValuesClockPlotLabels = Dict(methodToLabel(b) => fill("",1,1) for b = 0:4)

    # Run optimizer 
    runResults = []
    for b=1:nSettings    
        if verbose
            @printf("Running setting %d: method=%d, lsType=%d, nonOpt=%d\n",
                b,configDF[b,:].method,configDF[b,:].lsType,configDF[b,:].nonOpt)
        end
        
        time_start = time_ns() # Start timer
        (w,f,nObjEvals,nGradEvals,nIter,nLsIter,nMatMult,fValues,tValues) = minFuncSO(objFunc,fPrimeFunc,gradFunc,
            w0,X,method=configDF[b,:].method,maxIter=datasets[a].maxPasses,maxLsIter=datasets[a].maxLsIter,
            maxTimeInSec=datasets[a].maxTimeInSec,optTol=optTol,progTol=progTol,lsInit=configDF[b,:].lsInit,
            nFref=configDF[b,:].nFref,lsType=configDF[b,:].lsType,
            c1=c1,c2=c2,lsInterpType=configDF[b,:].lsInterp,lBfgsSize=configDF[b,:].lbfgsSize,
            momentumDirections=configDF[b,:].momentumDirs,ssMethod=configDF[b,:].ssMethod,
            ssLS=configDF[b,:].ssLS,ssRelStop=convert(Bool,configDF[b,:].ssRelStop),
            ssOneDInit=convert(Bool,configDF[b,:].ssOneDInit),derivativeCheck=convert(Bool,configDF[b,:].derivCheck),
            numDiff=convert(Bool,configDF[b,:].numDiff),verbose=verbose,funObjForT=funObjForT,
            funObj=funObjForNonOpt,funObjNoGrad=funObjNoGradForNonOpt,nonOpt=convert(Bool,configDF[b,:].nonOpt))
        nSeconds = (time_ns()-time_start)/1.0e9 # End timer
        
        if verbose
            @printf("f=%f, %f seconds, %d calls to f, %d calls to g, %d iters, %d LS iters, %d matrix-vector
             mults\n**********\n", 
                f,nSeconds,nObjEvals,nGradEvals,nIter,nLsIter,nMatMult) 
        end
        
        push!(runResults,(fStar=f,nObjEvals=nObjEvals,nGradEvals=nGradEvals,nIter=nIter,nLSIter=nLsIter,
            nMatMult=nMatMult,nSeconds=nSeconds))
        
        methodLabel=methodToLabel(configDF[b,:].method)
        lsLabel=lsTypeToLabel(configDF[b,:].lsType,configDF[b,:].momentumDirs,configDF[b,:].ssMethod,
            configDF[b,:].ssLS,configDF[b,:].nonOpt)
        addToPlot(methodLabel,lsLabel,fValuesIterPlot,fValuesIterPlotLabels,fValues)
        addToPlot(methodLabel,lsLabel,fValuesClockPlot,fValuesClockPlotLabels,tValues)
        
        @printf(allDScsvIO,"%s,%d,%d,%s-%s,%f,%d,%d,%f\n",
            datasetNames[a],m,n,methodLabel,lsLabel,f,nIter,nLsIter,nSeconds)
    end

    ##### print and save results #####
    csvFileName = string("./output/summary_",datasets[a].label,".csv")
    txtFileName = string("./output/summary_",datasets[a].label,".txt")
    open(csvFileName,"w") do csvIo; open(txtFileName,"w") do txtIo
        @printf("%-5s%-10s%-15s%-5s%-20s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-5s\n",
            "","Method","LStype","nMom","objVal","nIter","nLSIter",
            "nMatMult","nObjEval","nGradEval","time(s)","LSinit","LSinterp","Fref")
        @printf(csvIo,"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
            "Method","LStype","nMom","objVal","nIter","nLSIter",
            "nMatMult","nObjEval","nGradEval","time(s)","LSinit","LSinterp","Fref")
        @printf(txtIo,"%-10s%-15s%-5s%-20s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-5s\n",
            "Method","LStype","nMom","objVal","nIter","nLSIter",
            "nMatMult","nObjEval","nGradEval","time(s)","LSinit","LSinterp","Fref")
        @printf("%-5s%-10s%-15s%-5s%-20s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-5s\n",
            "","------","------","----","------","-----","-------",
            "--------","--------","---------","-------","------","--------","----")
        @printf(txtIo,"%-10s%-15s%-5s%-20s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-5s\n",
            "------","------","----","------","-----","-------",
            "--------","--------","---------","-------","------","--------","----")
        
        for b in 1:nSettings
            methodLabel=methodToLabel(configDF[b,:].method)
            lsLabel=lsTypeToLabel(configDF[b,:].lsType,configDF[b,:].momentumDirs,
                configDF[b,:].ssMethod,configDF[b,:].ssLS,configDF[b,:].nonOpt)
            @printf("%-5d%-10s%-15s%-5d%-20.9f%-10d%-10d%-10d%-10d%-10d%-10.2f%-10s%-10s%-5s\n",
                a,methodLabel,lsLabel,configDF[b,:].momentumDirs,runResults[b].fStar,runResults[b].nIter,
                runResults[b].nLSIter,runResults[b].nMatMult,runResults[b].nObjEvals,runResults[b].nGradEvals,
                runResults[b].nSeconds,configDF[b,:].lsInit,configDF[b,:].lsInterp,configDF[b,:].nFref)
            @printf(csvIo,"%s,%s,%d,%f,%d,%d,%d,%d,%d,%f,%s,%s,%s\n",
                methodLabel,lsLabel,configDF[b,:].momentumDirs,runResults[b].fStar,runResults[b].nIter,
                runResults[b].nLSIter,runResults[b].nMatMult,runResults[b].nObjEvals,runResults[b].nGradEvals,
                runResults[b].nSeconds,configDF[b,:].lsInit,configDF[b,:].lsInterp,configDF[b,:].nFref)
            @printf(txtIo,"%-10s%-15s%-5d%-20.9f%-10d%-10d%-10d%-10d%-10d%-10.2f%-10s%-10s%-5s\n",
                methodLabel,lsLabel,configDF[b,:].momentumDirs,runResults[b].fStar,runResults[b].nIter,
                runResults[b].nLSIter,runResults[b].nMatMult,runResults[b].nObjEvals,runResults[b].nGradEvals,
                runResults[b].nSeconds,configDF[b,:].lsInit,configDF[b,:].lsInterp,configDF[b,:].nFref)        
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
    methodsToPlot = [0 2 3 4] # skip BB because same search direction as GD
    subplotByIter = Dict(methodToLabel(b) => [] for b in methodsToPlot)
    subplotByTime = Dict(methodToLabel(b) => [] for b in methodsToPlot)
    fmin = minimum(runResults[b].fStar for b=1:nSettings)
    
    if savePlotData
        detailsFileName = string("./output/details_",datasets[a].label,".csv")
        detailsDf = DataFrame(xLabel=x)
    end
    
    yscalePower = log10(fInit-fmin)
    if isfinitereal(yscalePower)
        yscalePower = floor(Int,yscalePower)
        if yscalePower < 1e-2
            yscalePower = 1
        end
    else
        yscalePower = 1
    end
    upperLim = 10. ^ yscalePower
    currYLims = (progTol,upperLim)
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
        push!(subplotByIter[methodLabel],plot(x,fValuesIterPlot[methodLabel],label=fValuesIterPlotLabels[methodLabel],
            title=methodLabel,titlefont=font(8),xlabel="iterations",ylabel="optimality gap",
            legend=:outertopright,legendtitle="   Search Type",legendtitlefonthalign=:left,legendtitlefontsize=5,
            legendfontsize=5,guidefontsize=5,tickfontsize=5,yaxis=:log,ylims=currYLims,margin=5mm)) 
        push!(subplotByTime[methodLabel],plot(fValuesClockPlot[methodLabel],fValuesIterPlot[methodLabel],
            label=fValuesClockPlotLabels[methodLabel],title=methodLabel,titlefont=font(8),xlabel="total time (sec)",
            ylabel="optimality gap",legend=:outertopright,legendtitle="   Search Type",legendtitlefonthalign=:left,
            legendtitlefontsize=5,legendfontsize=5,guidefontsize=5,tickfontsize=5,yaxis=:log,ylims=currYLims,margin=5mm))
    end
    
    if savePlotData
        CSV.write(detailsFileName,detailsDf)
        @printf("Details file saved as %s\n",detailsFileName)
    end
    
    # not sure how to do this nicely
    fValPlot = plot(subplotByIter[methodToLabel(0)][1],subplotByTime[methodToLabel(0)][1],subplotByIter[methodToLabel(2)][1],
        subplotByTime[methodToLabel(2)][1],subplotByIter[methodToLabel(3)][1],subplotByTime[methodToLabel(3)][1],
        subplotByIter[methodToLabel(4)][1],subplotByTime[methodToLabel(4)][1],size=(1200,1000),layout=(4,2));
    
    plotName=string("./output/plot_",datasets[a].label,".png")
    savefig(fValPlot,plotName)
    @printf("Plot saved as %s\n",plotName)
end  

close(allDScsvIO)