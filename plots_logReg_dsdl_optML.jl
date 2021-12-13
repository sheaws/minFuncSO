# Load X and y variable
using JLD2, Printf, LinearAlgebra, Plots, MAT, DelimitedFiles,
    Plots.PlotMeasures, DataFrames, Tables, CSV, PyCall
include("minFuncSO.jl")
include("logReg.jl")
#plotlyjs()
gr()

# global settings
progTol = 1e-9 # function value progress tolerance
optTol = 1e-9 # first order optimality tolerance
c1= 1e-4 # parameter for Armijo/ suffient decrease condition
c2=0.9 # parameter for Wolfe/ curvature condition
verbose = false # print more output
savePlotData = true
y=nothing

# function we're going to minimize 
objFunc(z;konst=nothing) = logisticObjLinear(z,y,k=konst)
gradFunc(z;konst=nothing) = logisticGradLinear(z,y,k=konst)

# function and gradient wrt step size for line search that calls minFuncNonOpt
funObjForT(t,D,w,X) = logisticObjAndGrad(t,D,w,X,y)

# function and gradient for minFuncNonOpt
funObjForNonOpt(w,X) = logisticObjAndGrad(w,X,y)

# function and no gradient evaluation for minFuncNonOpt/ lsArmijoNonOpt
funObjNoGradForNonOpt(w,X) = logisticObjAndNoGrad(w,X,y)

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
function lsTypeToLabel(lsType,nMomDirs,ssMethod,ssLS)
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
    
    if lsType==4
        lsLabel=string(lsLabel,"*")
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

# List of datasets - for additional experiments in the appendix
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
configDF = CSV.read("./config/minFuncSO_optML.csv",DataFrame)
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
    push!(datasets,(label=datasetNames[a],maxPasses=1000,maxLsIter=25,maxTimeInSec=120))
    w0 = zeros(n,1) # initial solution
    fInit = objFunc(X*w0) # used for plotting

    # holds data for plots
    fValuesIterPlot = Dict(datasetNames[a] => zeros(datasets[a].maxPasses,1))
    fValuesIterPlotLabels = Dict(datasetNames[a] => fill("",1,1))
    fValuesClockPlot = Dict(datasetNames[a] => zeros(datasets[a].maxPasses,1))
    fValuesClockPlotLabels = Dict(datasetNames[a] => fill("",1,1))

    # Run optimizer 
    runResults = []
    for b=1:nSettings    
        if verbose
            @printf("Running setting %d: method=%d, lsType=%d\n",b,configDF[b,:].method,configDF[b,:].lsType)
        end
        
        time_start = time_ns() # Start timer
        (w,f,nObjEvals,nGradEvals,nIter,nLsIter,nMatMult,fValues,tValues) = minFuncSO(objFunc,gradFunc,w0,X,
            method=configDF[b,:].method,maxIter=datasets[a].maxPasses,maxLsIter=datasets[a].maxLsIter,
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
        
        if isnan(f) # for plotting
            f = Inf
        end
        push!(runResults,(fStar=f,nObjEvals=nObjEvals,nGradEvals=nGradEvals,nIter=nIter,nLSIter=nLsIter,
            nMatMult=nMatMult,nSeconds=nSeconds))
        
        # these are the labels for the methods for the workshop paper
        methodLabel=datasetNames[a]
        if b==1
            lsLabel="GD-SO"
        elseif b==2
            lsLabel="lBFGS-W-Def"
        elseif b==3
            lsLabel="lBFGS-W-Opt"
        elseif b==4
            lsLabel="lBFGS-SO-Opt"
        else
            lsLabel="lBFGS-SO-Opt2"
        end
         
        addToPlot(methodLabel,lsLabel,fValuesIterPlot,fValuesIterPlotLabels,fValues)
        addToPlot(methodLabel,lsLabel,fValuesClockPlot,fValuesClockPlotLabels,tValues)
        
        @printf(allDScsvIO,"%s,%d,%d,%s,%f,%d,%d,%f\n",
            datasetNames[a],m,n,lsLabel,f,nIter,nLsIter,nSeconds)
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
                configDF[b,:].ssMethod,configDF[b,:].ssLS)
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
    if savePlotData
        detailsFileName = string("./output/details_",datasets[a].label,".csv")
        detailsDf = DataFrame(xLabel=x)
    end
    subplotByIter = Dict(datasetNames[a] => [])
    subplotByTime = Dict(datasetNames[a] => [])
    fmin = minimum(runResults[b].fStar for b=1:nSettings)
    
    upperLim = 10^floor(Int,log10(fInit-fmin))
    currYLims = (optTol,upperLim)
    methodLabel=datasetNames[a]
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
        legend=:topright,legendfontsize=5,guidefontsize=5,tickfontsize=5,yaxis=:log,ylims=currYLims,margin=5mm)) 
    push!(subplotByTime[methodLabel],plot(fValuesClockPlot[methodLabel],fValuesIterPlot[methodLabel],
        label=fValuesClockPlotLabels[methodLabel],title=methodLabel,titlefont=font(8),xlabel="total time (sec)",
        ylabel="optimality gap",legend=:topright,legendfontsize=5,guidefontsize=5,tickfontsize=5,yaxis=:log,
        ylims=currYLims,margin=5mm))
    
    if savePlotData
        CSV.write(detailsFileName,detailsDf)
        @printf("Details file saved as %s\n",detailsFileName)
    end
    
    # not sure how to do this nicely
    fValPlot = plot(subplotByIter[methodLabel][1],subplotByTime[methodLabel][1],size=(1200,250),layout=(1,2));
    
    plotName=string("./output/plot_",datasets[a].label,".png")
    savefig(fValPlot,plotName)
    @printf("Plot saved as %s\n",plotName)
end  

close(allDScsvIO)