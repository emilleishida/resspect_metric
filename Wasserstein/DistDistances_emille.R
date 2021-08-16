#
# Distances between distributions for RESSPECT
#
# To calculate all the distances and write the output file containing the 
# distances between all pairs of distributions, use resspectGenerateDistanceMatrix()
# setting the parameters as needed.
# 
# COIN, June, 2021
#

require(vroom)
require(pbapply)
require(transport)
require(modeest)
require(parallel)
require(tictoc)

# This is the main function used to produce the distance matrix
resspectGenerateDistanceMatrix <- function(distpath="/media/RESSPECT/data/PLAsTiCC/for_metrics/ddf/posteriors/samples_emille/", 
                                           outfile="wassersteinDistances_ddf.dat", nptsPerDrawn=1000, nDrawns=100, nThreads=20) {
  
  cat(paste("\nCOIN-LSST/DESC 2019-2021 :: RESSPECT\n\n"))
  cat(paste("Estimating Wasserstein distances for multidimensional distributions.\n"))
  
  # Get filenames
  filenames <- dir(distpath, full.names = TRUE)
  
  cat(paste("Each distribution is in a different file. Number of distributions      :",length(filenames)," \n"))
  
  # Organize the triangular matrix positions with the filenames
  pMat <- c()
  
  for(i in 1:length(filenames)) {
    for(j in i:length(filenames)) {
      pMat <- rbind(pMat, data.frame(i=filenames[i], j=filenames[j]))
    }
  }
  
  # Initialize the additional threads
  myCl <- makeForkCluster(nThreads)
  cat(paste("                                          Number of drawns             :",nDrawns," \n"))
  cat(paste("                                          Number of points per drawn   :",nptsPerDrawn," \n"))
  cat(paste("                                          Number of distance estimates :",nrow(pMat)," \n"))
  cat(paste("                                          Number of threads            :",nThreads," \n\n"))
  
  # Run distance estimation between all pairs of files
  dList <- pbapply(pMat, 1,
                   FUN = function(x) {
                     cbind(basename(x[1]), basename(x[2]),
                           resspectComputeDistResampling(npts=nptsPerDrawn, nSamples = nDrawns, fileA=x[1], fileB=x[2]))
                   },
                   cl = myCl
  )
  
  # Stop the additional threads
  stopCluster(myCl)
  
  # Convert back into a dataframe... it is not elegant,
  # but it works even for non atomic vectors
  finalDf <- c()
  for(i in 1:length(dList)) {
    finalDf <- rbind(finalDf, dList[[i]])
  }
  namesCol <- names(finalDf)
  namesCol[1] <- "FileA"
  namesCol[2] <- "FileB"
  colnames(finalDf) <- namesCol
  
  # Write output file
  write.csv(file = outfile, x = finalDf)
  
  # That's all folks!
  cat(paste("\n\nDone. Results were written in the file: ",outfile,"\n\n"))
  
}

# Computes the Wasserstein distance for a pair pair of files
# The files must contain points sampled from a distribution
# Each point must be contained in a different row, and there can be
# any number of columns (dimensions). If the dataset is to large, you can
# select the number of points to be randomly taken in the distance
# calculation using the npts parameter (if npts=0 all points will be used).
resspectComputeDist <- function(npts=100,
                                fileA="../../samples_emille/chains_72SNIa28SNII_lowz_withbias.csv.gz",
                                fileB="../../samples_emille/chains_75SNIa25SNIax_lowz_withbias.csv.gz") {
  
  # A small work around to supress vroom messages
  Sys.setenv("VROOM_SHOW_PROGRESS"="false")
  
  print(fileB)
  
  # Load the required data
  sampleA <- suppressMessages(vroom(fileA))
  sampleB <- suppressMessages(vroom(fileB))
  
  # Resample if necessary
  if(npts > 0) {
    if(npts <= nrow(sampleA)) {
      sampleA <- sampleA[sample(1:nrow(sampleA),npts),]
    } else {
      stop("You cannot request more samples than available in the file from the distribution A!")
    }
    if(npts <= nrow(sampleB)) {
      sampleB <- sampleB[sample(1:nrow(sampleB),npts),]
    } else {
      stop("You cannot request more samples than available in the file from the distribution B!")
    }
  }
  
  # Transform into a point pattern object for the optimal transport package
  ppA <- pp(sampleA)
  ppB <- pp(sampleB)
  
  # Compute and return the Wasserstein distance, using an euclidean transportation cost
  return(wasserstein(ppA, ppB, p=1, prob=TRUE))
}

# Computes the mean, st.dev. and Shapiro-Wilk normality tests for Wasserstein distances
# calculated between several resamplings of datasets read from fileA and fileB
resspectComputeDistResampling <- function(npts=100, nSamples=100,
                                          fileA="../../samples_emille/chains_72SNIa28SNII_lowz_withbias.csv.gz",
                                          fileB="../../samples_emille/chains_75SNIa25SNIax_lowz_withbias.csv.gz",
                                          verbose=FALSE) {
  
  cat(paste("\nEstimating distance and uncertainty...\n"))
  cat(paste("\    fileA =",fileA,"\n"))
  cat(paste("\    fileB =",fileB," \n"))
  dVec <- pbapply(t(1:nSamples), 2, FUN = function(x) {resspectComputeDist(npts=npts, fileA=fileA, fileB=fileB)} )
  
  if(verbose) {
    cat(paste("Mean   = ", mean(dVec) ,"\nSt.Dev = ", sd(dVec), "\n"))
  }
  
  # Check for normality
  shapiroTest <- shapiro.test(dVec)
  if(verbose) {
    print(shapiroTest)
  }
  
  return(data.frame(WassersteinDistanceMean=mean(dVec), WassersteinDistanceStDev=sd(dVec), 
                    WassersteinDistanceMedian=median(dVec), WassersteinDistanceMode=venter(dVec),
                    ShapiroWilkStatistic=shapiroTest$statistic, ShapiroWilkPvalue=shapiroTest$p.value,
                    NPointsPerDrawn=npts, NDrawns=nSamples))
  
}


#############################################################################
# Additional functions to diagnose the scaling with respect to the number of
# points in the input distributions
#############################################################################

testScalingW <- function(nPtsTest = c(100, 250, 500, 1000, 1500, 2000), overSamplingFactor=2) {
  yy <- c()
  tic.clearlog()
  
  # Create the sample os number of points to be tested
  nPtsTest <- c(nPtsTest, runif(n=(overSamplingFactor*length(nPtsTest)), min = min(nPtsTest), max=max(nPtsTest)))
  nPtsTest <- nPtsTest[order(nPtsTest)]
  
  # Compute the times and distances
  for(i in 1:length(nPtsTest)){
    tic()
    retVal <- resspectLoadDist(nPtsTest[i])
    toc(log = TRUE, quiet = TRUE)
    log.lst <- tic.log(format = FALSE)
    timings <- unlist(lapply(log.lst, function(x) x$toc - x$tic))
    print(timings)
    yy <- cbind(yy, retVal)
    
    # Show diagnostics plots
    plotDiagnostics(nPtsTest[1:i], yy, timings, addLines=TRUE)
  }
  
  # Estimate the time that it would take to run the complete dataset
  lmMod <- lm(time ~ npoints, data=data.frame(npoints=nPtsTest[1:i], time=log10(timings)))
  cat(paste("Estimated time for 10k points: ", (10^predict(lmMod, data.frame(npoints=10000)))/60/60/24, " days \n"))
  
  plotDiagnostics(nPtsTest, yy, timings, addLines=FALSE)
  lines(1:3000, predict(lmMod, data.frame(npoints=c(1:3000))), col="red")
  
  # Estimate for full dataset
}

plotDiagnostics <- function(nPtsTest, yy, timings, addLines=TRUE) {
  par(mfrow=c(2,1), mar=c(4,4,2,1))
  # Distances
  plot(nPtsTest, yy, pch=19, cex=0.5, xlab="Number of points", ylab="2D Wasserstein distance")
  grid()
  if(addLines) {
    lines(nPtsTest, yy, col="red")
  }
  points(nPtsTest, yy, pch=19, cex=0.5)
  # Number of points
  plot(nPtsTest, log10(timings), pch=19, cex=0.5, xlab="Number of points", ylab="Log10(Time) [s]")
  grid()
  if(addLines) {
    lines(nPtsTest, log10(timings), col="red")
  }
  points(nPtsTest, log10(timings), pch=19, cex=0.5)
}
