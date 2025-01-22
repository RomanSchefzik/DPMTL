
################################################################################
#' @title Sensitivity analysis for MTL with logistic loss and joint feature selection and low-rank structure, respectively
#' @description Sensitivity analysis for MTL with logistic loss and joint feature selection and low-rank structure, respectively: For each of \eqn{T} tasks, compute the component-wise distance between an original gradient vector and a gradient vector obtained when removing one data point from the input data, perform this procedure for in total \code{nSimulations} data point removals and calculate the maximum of each gradient vector component over all \code{nSimulations} repetitions  
#' @param rData A list consisting of \eqn{T} elements, each consisting of a list comprising two elements named grad_w (a \eqn{(P \times 1)} matrix including a gradient vector corresponding to \eqn{P} predictors) and funcVal (a function value), respectively
#' @param C The hyper-parameter associated with L2 term in the respective MTL model
#' @param W A \eqn{(P \times T)} coefficient matrix
#' @param X A set of \eqn{T} \eqn{(N_t \times P)} feature matrices, \eqn{t \in \{1,\ldots,T\}}
#' @param Y A set of \eqn{T} binary response vectors \eqn{\in\{1, -1\}^{N_t}}, \eqn{t \in \{1,\ldots,T\}}
#' @param nSimulations number of runs in the sensitivity analysis involved in the differential privacy mechanism
#' 
#' @return A \eqn{(P \times T)} matrix containing the maximum component-wise distances between the original gradient vector and a gradient vector obtained when removing one data point from the input data, across \code{nSimulations} runs for each of the \eqn{T} tasks
#' 
#' @examples
#' set.seed(24)
#' X0<-list(matrix(rnorm(100000),nrow=200,ncol=500),matrix(rnorm(150000),nrow=300,ncol=500))
#' set.seed(24)
#' Y0<-list(sample(c(1,-1),200,replace=TRUE),sample(c(1,-1),300,replace=TRUE))
#' set.seed(24)
#' W0<-matrix(rnorm(1000),nrow=500,ncol=2)
#' nTasks<-2
#' rData0<-lapply(1:nTasks, function(k){
#' x=X0[[k]]; y=Y0[[k]];w=W0[,k]
#' weight <- 1/length(y)
#' l <- -y*(x %*% w) 
#' lp <- l
#' lp[lp<0] <- 0
#' funcVal <- sum(weight * ( log( exp(-lp) +  exp(l-lp) ) + lp ))
#' b <- (-weight*y)*(1 - 1/ (1+exp(l)))
#' grad_w <- (t(x) %*% b)
#' return(list(grad_w=grad_w, funcVal=funcVal))
#' })
#' C0<-1
#' n<-100
#' simdiff<-simulateDifferences_LR(rData=rData0, C=C0, W=W0, X=X0, Y=Y0, nSimulations=n)
#' 
#' @export 
simulateDifferences_LR <- function(rData, C, W, X, Y, nSimulations) {
  # Calculate original values with all data
  originalGradWs <- sapply(rData, function(x) x$grad_w) + 2 * C * W
  originalFuncVals <- sum(sapply(rData, function(x) x$funcVal)) + C * norm(W, 'f')^2
  
  gradWsDifferences <- list()
  
  for(i in 1:nSimulations) {
    # Perform the sensitivity analysis for each task
    modifiedRData <- lapply(1:length(X), function(k) {
      x <- X[[k]]
      y <- Y[[k]]
      w <- W[,k]
      
      nRemove <- 1
      rowsToRemove <- sample(1:nrow(x), nRemove)
      xModified <- x[-rowsToRemove, ]
      yModified <- y[-rowsToRemove]
      
      # Recalculate function value and gradient without the removed rows
      weight <- 1 / length(yModified)
      l <- -yModified * (xModified %*% w)
      lp <- l
      lp[lp < 0] <- 0
      funcVal <- sum(weight * (log(exp(-lp) + exp(l - lp)) + lp))
      b <- (-weight * yModified) * (1 - 1 / (1 + exp(l)))
      grad_w <- (t(xModified) %*% b)
      return(list(grad_w = grad_w, funcVal = funcVal))
    })
    
    # Recompute grad_Ws and funcVals
    modifiedGradWs <- sapply(modifiedRData, function(x) x$grad_w) + 2 * C * W
    
    # Store the differences
    gradWsDifferences[[i]] <- abs(originalGradWs - modifiedGradWs)
  }
  
  # Find the maximum differences
  maxGradWsDifference <- do.call(pmax, c(gradWsDifferences, na.rm = TRUE))
  
  return(maxGradWsDifference)
}




################################################################################
#' @title Sensitivity analysis for MTL with logistic loss and network incorporation
#' @description Sensitivity analysis for MTL with logistic loss and network incorporation: For each of \eqn{T} tasks, compute the component-wise distance between an original gradient vector and a gradient vector obtained when removing one data point from the input data, perform this procedure for in total \code{nSimulations} data point removals and calculate the maximum of each gradient vector component over all \code{nSimulations} repetitions  
#' @param rData A list consisting of \eqn{T} elements, each consisting of a list comprising two elements named grad_w (a \eqn{(P \times 1)} matrix including a gradient vector corresponding to \eqn{P} predictors) and funcVal (a function value), respectively
#' @param C The hyper-parameter associated with L2 term in the respective MTL model
#' @param W A \eqn{(P \times T)} coefficient matrix
#' @param X A set of \eqn{T} \eqn{(N_t \times P)} feature matrices, \eqn{t \in \{1,\ldots,T\}}
#' @param Y A set of \eqn{T} binary response vectors \eqn{\in\{1, -1\}^{N_t}}, \eqn{t \in \{1,\ldots,T\}}
#' @param G Matrix describing the task relatedness
#' @param nSimulations number of runs in the sensitivity analysis involved in the differential privacy mechanism
#' 
#' @return A \eqn{(P \times T)} matrix containing the maximum component-wise distances between the original gradient vector and a gradient vector obtained when removing one data point from the input data, across \code{nSimulations} runs for each of the \eqn{T} tasks
#' 
#' @examples
#' set.seed(24)
#' X0<-list(matrix(rnorm(100000),nrow=200,ncol=500),matrix(rnorm(150000),nrow=300,ncol=500))
#' set.seed(24)
#' Y0<-list(sample(c(1,-1),200,replace=TRUE),sample(c(1,-1),300,replace=TRUE))
#' set.seed(24)
#' W0<-matrix(rnorm(1000),nrow=500,ncol=2)
#' nTasks<-2
#' rData0<-lapply(1:nTasks, function(k){
#' x=X0[[k]]; y=Y0[[k]];w=W0[,k]
#' weight <- 1/length(y)
#' l <- -y*(x %*% w) 
#' lp <- l
#' lp[lp<0] <- 0
#' funcVal <- sum(weight * ( log( exp(-lp) +  exp(l-lp) ) + lp ))
#' b <- (-weight*y)*(1 - 1/ (1+exp(l)))
#' grad_w <- (t(x) %*% b)
#' return(list(grad_w=grad_w, funcVal=funcVal))
#' })
#' G0<-diag(nTasks)-(1/nTasks)
#' C0<-1
#' n<-100
#' simdiff<-simulateDifferences_LR_Net(rData=rData0, C=C0, W=W0, X=X0, Y=Y0,G=G0, nSimulations=n)
#' 
#' @export 
simulateDifferences_LR_Net <- function(rData, C, W, X, Y,G, nSimulations) {
  # Calculate original values with all data
  GGt <- G %*% t(G)
  originalGradWs <- sapply(rData, function(x) x$grad_w) + 2 * C * W %*% GGt
  originalFuncVals <- sum(sapply(rData, function(x) x$funcVal)) + C * norm(W %*% G, 'f')^2
  
  gradWsDifferences <- list()
  
  for(i in 1:nSimulations) {
    # Perform the sensitivity analysis for each task
    modifiedRData <- lapply(1:length(X), function(k) {
      x <- X[[k]]
      y <- Y[[k]]
      w <- W[,k]
      
      nRemove <- 1
      rowsToRemove <- sample(1:nrow(x), nRemove)
      xModified <- x[-rowsToRemove, ]
      yModified <- y[-rowsToRemove]
      
      # Recalculate function value and gradient without the removed rows
      weight <- 1 / length(yModified)
      l <- -yModified * (xModified %*% w)
      lp <- l
      lp[lp < 0] <- 0
      funcVal <- sum(weight * (log(exp(-lp) + exp(l - lp)) + lp))
      b <- (-weight * yModified) * (1 - 1 / (1 + exp(l)))
      grad_w <- (t(xModified) %*% b)
      return(list(grad_w = grad_w, funcVal = funcVal))
    })
    
    # Recompute grad_Ws and funcVals
    modifiedGradWs <- sapply(modifiedRData, function(x) x$grad_w) + 2 * C * W %*% GGt
    
    # Store the differences
    gradWsDifferences[[i]] <- abs(originalGradWs - modifiedGradWs)
  }
  
  # Find the maximum differences
  maxGradWsDifference <- do.call(pmax, c(gradWsDifferences, na.rm = TRUE))
  
  return(maxGradWsDifference)
}




################################################################################
#' @title Sensitivity analysis for MTL with least-square loss and joint feature selection and low-rank structure, respectively
#' @description Sensitivity analysis for MTL with least-square loss and joint feature selection and low-rank structure, respectively: For each of \eqn{T} tasks, compute the component-wise distance between an original gradient vector and a gradient vector obtained when removing one data point from the input data, perform this procedure for in total \code{nSimulations} data point removals and calculate the maximum of each gradient vector component over all \code{nSimulations} repetitions  
#' @param rData A list consisting of \eqn{T} elements, each consisting of a list comprising two elements named grad_w (a \eqn{(P \times 1)} matrix including a gradient vector corresponding to \eqn{P} predictors) and funcVal (a function value), respectively
#' @param C The hyper-parameter associated with L2 term in the respective MTL model
#' @param W A \eqn{(P \times T)} coefficient matrix
#' @param X A set of \eqn{T} \eqn{(N_t \times P)} feature matrices, \eqn{t \in \{1,\ldots,T\}}
#' @param Y A set of \eqn{T} response vectors \eqn{\in \mathbb{R}^{N_t}}, \eqn{t \in \{1,\ldots,T\}}
#' @param nSimulations number of runs in the sensitivity analysis involved in the differential privacy mechanism
#' 
#' @return A \eqn{(P \times T)} matrix containing the maximum component-wise distances between the original gradient vector and a gradient vector obtained when removing one data point from the input data, across \code{nSimulations} runs for each of the \eqn{T} tasks
#' 
#' @examples
#' set.seed(24)
#' X0<-list(matrix(rnorm(100000),nrow=200,ncol=500),matrix(rnorm(150000),nrow=300,ncol=500))
#' set.seed(24)
#' Y0<-list(rnorm(200),rnorm(300))
#' set.seed(24)
#' W0<-matrix(rnorm(1000),nrow=500,ncol=2)
#' nTasks<-2
#' rData0<-lapply(1:nTasks, function(k){
#' x=X0[[k]]; y=Y0[[k]];w=W0[,k]
#' grad_w <-  t(x) %*% (x %*% w - y) / nrow(x)
#' funcVal <- 0.5 * mean((y - x %*% w)^2)
#' return(list(grad_w=grad_w, funcVal=funcVal))
#' })
#' C0<-1
#' n<-100
#' simdiff<-simulateDifferences_LS(rData=rData0, C=C0, W=W0, X=X0, Y=Y0, nSimulations=n)
#' 
#' @export 
simulateDifferences_LS <- function(rData, C, W, X, Y, nSimulations) {
  # Calculate original values with all data
  originalGradWs <- sapply(rData, function(x) x$grad_w) + 2 * C * W
  originalFuncVals <- sum(sapply(rData, function(x) x$funcVal)) + C * norm(W, 'f')^2
  
  gradWsDifferences <- list()
  
  for(i in 1:nSimulations) {
    # Perform the sensitivity analysis for each task
    modifiedRData <- lapply(1:length(X), function(k) {
      x <- X[[k]]
      y <- Y[[k]]
      w <- W[,k]
      
      nRemove <- 1
      rowsToRemove <- sample(1:nrow(x), nRemove)
      xModified <- x[-rowsToRemove, ]
      yModified <- y[-rowsToRemove]
      
      # Recalculate function value and gradient without the removed rows
      grad_w <-  t(xModified) %*% (xModified %*% w - yModified) / nrow(xModified)
      funcVal <- 0.5 * mean((yModified - xModified %*% w)^2)
      return(list(grad_w = grad_w, funcVal = funcVal))
    })
    
    # Recompute grad_Ws and funcVals
    modifiedGradWs <- sapply(modifiedRData, function(x) x$grad_w) + 2 * C * W
    
    # Store the differences
    gradWsDifferences[[i]] <- abs(originalGradWs - modifiedGradWs)
  }
  
  # Find the maximum differences
  maxGradWsDifference <- do.call(pmax, c(gradWsDifferences, na.rm = TRUE))
  
  return(maxGradWsDifference)
}





################################################################################
#' @title Sensitivity analysis for MTL with least-square loss and network incorporation
#' @description Sensitivity analysis for MTL with least-square loss and network incorporation: For each of \eqn{T} tasks, compute the component-wise distance between an original gradient vector and a gradient vector obtained when removing one data point from the input data, perform this procedure for in total \code{nSimulations} data point removals and calculate the maximum of each gradient vector component over all \code{nSimulations} repetitions  
#' @param rData A list consisting of \eqn{T} elements, each consisting of a list comprising two elements named grad_w (a \eqn{(P \times 1)} matrix including a gradient vector corresponding to \eqn{P} predictors) and funcVal (a function value), respectively
#' @param C The hyper-parameter associated with L2 term in the respective MTL model
#' @param W A \eqn{(P \times T)} coefficient matrix
#' @param X A set of \eqn{T} \eqn{(N_t \times P)} feature matrices, \eqn{t \in \{1,\ldots,T\}}
#' @param Y A set of \eqn{T} response vectors \eqn{\in \mathbb{R}^{N_t}}, \eqn{t \in \{1,\ldots,T\}}
#' @param G Matrix describing the task relatedness
#' @param nSimulations number of runs in the sensitivity analysis involved in the differential privacy mechanism
#' 
#' @return A \eqn{(P \times T)} matrix containing the maximum component-wise distances between the original gradient vector and a gradient vector obtained when removing one data point from the input data, across \code{nSimulations} runs for each of the \eqn{T} tasks
#' 
#' @examples
#' set.seed(24)
#' X0<-list(matrix(rnorm(100000),nrow=200,ncol=500),matrix(rnorm(150000),nrow=300,ncol=500))
#' set.seed(24)
#' Y0<-list(rnorm(200),rnorm(300))
#' set.seed(24)
#' W0<-matrix(rnorm(1000),nrow=500,ncol=2)
#' nTasks<-2
#' rData0<-lapply(1:nTasks, function(k){
#' x=X0[[k]]; y=Y0[[k]];w=W0[,k]
#' grad_w <-  t(x) %*% (x %*% w - y) / nrow(x)
#' funcVal <- 0.5 * mean((y - x %*% w)^2)
#' return(list(grad_w=grad_w, funcVal=funcVal))
#' })
#' G0<-diag(nTasks)-(1/nTasks)
#' C0<-1
#' n<-100
#' simdiff<-simulateDifferences_LS_Net(rData=rData0, C=C0, W=W0, X=X0, Y=Y0,G=G0, nSimulations=n)
#' 
#' @export 
simulateDifferences_LS_Net <- function(rData, C, W, X, Y,G, nSimulations) {
  # Calculate original values with all data
  GGt <- G %*% t(G)
  originalGradWs <- sapply(rData, function(x) x$grad_w) + 2 * C * W %*% GGt
  originalFuncVals <- sum(sapply(rData, function(x) x$funcVal)) + C * norm(W %*% G, 'f')^2
  
  gradWsDifferences <- list()
  
  for(i in 1:nSimulations) {
    # Perform the sensitivity analysis for each task
    modifiedRData <- lapply(1:length(X), function(k) {
      x <- X[[k]]
      y <- Y[[k]]
      w <- W[,k]
      
      nRemove <- 1
      rowsToRemove <- sample(1:nrow(x), nRemove)
      xModified <- x[-rowsToRemove, ]
      yModified <- y[-rowsToRemove]
      
      # Recalculate function value and gradient without the removed rows
      grad_w <-  t(xModified) %*% (xModified %*% w - yModified) / nrow(xModified)
      funcVal <- 0.5 * mean((yModified - xModified %*% w)^2)
      return(list(grad_w = grad_w, funcVal = funcVal))
    })
    
    # Recompute grad_Ws and funcVals
    modifiedGradWs <- sapply(modifiedRData, function(x) x$grad_w) +  2 * C * W %*% GGt
    
    # Store the differences
    gradWsDifferences[[i]] <- abs(originalGradWs - modifiedGradWs)
  }
  
  # Find the maximum differences
  maxGradWsDifference <- do.call(pmax, c(gradWsDifferences, na.rm = TRUE))
  
  return(maxGradWsDifference)
}




######################################################################
######################################################################
######################################################################




################################################################################
#' @title Sensitivity analysis for MTL with logistic loss and joint feature selection and low-rank structure, respectively
#' @description Sensitivity analysis for MTL with logistic loss and joint feature selection and low-rank structure, respectively: Compute the difference between an original objective function value and a modified objective function value obtained when removing one data point from the input data, perform this procedure for in total \code{nSimulations} data point removals and calculate the maximum of the differences over all \code{nSimulations} repetitions  
#' @param X A set of \eqn{T} \eqn{(N_t \times P)} feature matrices, \eqn{t \in \{1,\ldots,T\}}
#' @param Y A set of \eqn{T} binary response vectors \eqn{\in\{1, -1\}^{N_t}}, \eqn{t \in \{1,\ldots,T\}}
#' @param W A \eqn{(P \times T)} coefficient matrix
#' @param C The hyper-parameter associated with L2 term in the respective MTL model
#' @param funcVal The original function value
#' @param nSimulations number of runs in the sensitivity analysis involved in the differential privacy mechanism
#' 
#' @return A list of two elements named maxDifference (maximum difference between an original objective function value and a modified objective function value obtained when removing one data point from the input data, across \code{nSimulations} runs) and originalValue (original objective function value), respectively
#' 
#' @examples
#' set.seed(24)
#' X0<-list(matrix(rnorm(100000),nrow=200,ncol=500),matrix(rnorm(150000),nrow=300,ncol=500))
#' set.seed(24)
#' Y0<-list(sample(c(1,-1),200,replace=TRUE),sample(c(1,-1),300,replace=TRUE))
#' set.seed(24)
#' W0<-matrix(rnorm(1000),nrow=500,ncol=2)
#' C0<-1
#' funcVal0<-0.375
#' n<-100
#' maxdiff<-computeMaxDifference_LR(X=X0, Y=Y0, W=W0, C=C0,funcVal=funcVal0, nSimulations=n)
#' 
#' @export 
computeMaxDifference_LR <- function(X, Y, W, C, funcVal, nSimulations) {
  originalValue <- funcVal # Assume funcVal has been computed with the full dataset
  
  differences <- numeric(nSimulations) # Vector to store differences
  
  for(i in 1:nSimulations) {
    nRemove <- 1
    
    # Perform the sensitivity analysis for each task
    modifiedValues <- sapply(1:length(X), function(k) {
      x <- X[[k]]
      y <- Y[[k]]
      w <- W[,k]
      
      # Randomly select rows to remove
      rowsToRemove <- sample(1:nrow(x), nRemove)
      xModified <- x[-rowsToRemove, ]
      yModified <- y[-rowsToRemove]
      
      # Recompute the weighted sum without the removed rows
      weight <- 1 / length(yModified)
      l <- - yModified * (xModified %*% w)
      lp <- l
      lp[lp < 0] <- 0
      sum(weight * (log(exp(-lp) + exp(l - lp)) + lp))
    })
    
    # Sum the values across tasks and add the regularization term
    modifiedValue <- sum(modifiedValues) + C * norm(W, 'f')^2
    
    # Store the difference
    differences[i] <- abs(originalValue - modifiedValue)
  }
  
  # Find the maximum difference
  maxDifference <- max(differences)
  return(list(
    maxDifference = maxDifference,
    originalValue = originalValue
  ))
}




################################################################################
#' @title Sensitivity analysis for MTL with logistic loss and network incorporation
#' @description Sensitivity analysis for MTL with logistic loss and network incorporation: Compute the difference between an original objective function value and a modified objective function value obtained when removing one data point from the input data, perform this procedure for in total \code{nSimulations} data point removals and calculate the maximum of the differences over all \code{nSimulations} repetitions  
#' @param X A set of \eqn{T} \eqn{(N_t \times P)} feature matrices, \eqn{t \in \{1,\ldots,T\}}
#' @param Y A set of \eqn{T} binary response vectors \eqn{\in\{1, -1\}^{N_t}}, \eqn{t \in \{1,\ldots,T\}}
#' @param W A \eqn{(P \times T)} coefficient matrix
#' @param C The hyper-parameter associated with L2 term in the respective MTL model
#' @param funcVal The original function value
#' @param G Matrix describing the task relatedness
#' @param nSimulations number of runs in the sensitivity analysis involved in the differential privacy mechanism
#' 
#' @return A list of two elements named maxDifference (maximum difference between an original objective function value and a modified objective function value obtained when removing one data point from the input data, across \code{nSimulations} runs) and originalValue (original objective function value), respectively
#' 
#' @examples
#' set.seed(24)
#' X0<-list(matrix(rnorm(100000),nrow=200,ncol=500),matrix(rnorm(150000),nrow=300,ncol=500))
#' set.seed(24)
#' Y0<-list(sample(c(1,-1),200,replace=TRUE),sample(c(1,-1),300,replace=TRUE))
#' set.seed(24)
#' W0<-matrix(rnorm(1000),nrow=500,ncol=2)
#' C0<-1
#' funcVal0<-0.375
#' G0<-diag(2)-(1/2)
#' n<-100
#' maxdiff<-computeMaxDifference_LR_Net(X=X0, Y=Y0, W=W0, C=C0,funcVal=funcVal0,G=G0, nSimulations=n)
#' 
#' @export
computeMaxDifference_LR_Net <- function(X, Y, W, C, funcVal,G, nSimulations) {
  originalValue <- funcVal # Assume funcVal has been computed with the full dataset
  
  differences <- numeric(nSimulations) # Vector to store differences
  
  for(i in 1:nSimulations) {
    nRemove <- 1
    
    # Perform the sensitivity analysis for each task
    modifiedValues <- sapply(1:length(X), function(k) {
      x <- X[[k]]
      y <- Y[[k]]
      w <- W[,k]
      
      # Randomly select rows to remove
      rowsToRemove <- sample(1:nrow(x), nRemove)
      xModified <- x[-rowsToRemove, ]
      yModified <- y[-rowsToRemove]
      
      # Recompute the weighted sum without the removed rows
      weight <- 1 / length(yModified)
      l <- - yModified * (xModified %*% w)
      lp <- l
      lp[lp < 0] <- 0
      sum(weight * (log(exp(-lp) + exp(l - lp)) + lp))
    })
    
    # Sum the values across tasks and add the regularization term
    modifiedValue <- sum(modifiedValues) + C * norm(W %*% G, 'f')^2
    
    # Store the difference
    differences[i] <- abs(originalValue - modifiedValue)
  }
  
  # Find the maximum difference
  maxDifference <- max(differences)
  return(list(
    maxDifference = maxDifference,
    originalValue = originalValue
  ))
}



################################################################################
#' @title Sensitivity analysis for MTL with least-square loss and joint feature selection and low-rank structure, respectively
#' @description Sensitivity analysis for MTL with least-square loss and joint feature selection and low-rank structure, respectively: Compute the difference between an original objective function value and a modified objective function value obtained when removing one data point from the input data, perform this procedure for in total \code{nSimulations} data point removals and calculate the maximum of the differences over all \code{nSimulations} repetitions  
#' @param X A set of \eqn{T} \eqn{(N_t \times P)} feature matrices, \eqn{t \in \{1,\ldots,T\}}
#' @param Y A set of \eqn{T} response vectors \eqn{\in \mathbb{R}^{N_t}}, \eqn{t \in \{1,\ldots,T\}}
#' @param W A \eqn{(P \times T)} coefficient matrix
#' @param C The hyper-parameter associated with L2 term in the respective MTL model
#' @param funcVal The original function value
#' @param nSimulations number of runs in the sensitivity analysis involved in the differential privacy mechanism
#' 
#' @return A list of two elements named maxDifference (maximum difference between an original objective function value and a modified objective function value obtained when removing one data point from the input data, across \code{nSimulations} runs) and originalValue (original objective function value), respectively
#' 
#' @examples
#' set.seed(24)
#' X0<-list(matrix(rnorm(100000),nrow=200,ncol=500),matrix(rnorm(150000),nrow=300,ncol=500))
#' set.seed(24)
#' Y0<-list(rnorm(200),rnorm(300))
#' set.seed(24)
#' W0<-matrix(rnorm(1000),nrow=500,ncol=2)
#' C0<-1
#' funcVal0<-0.375
#' n<-100
#' maxdiff<-computeMaxDifference_LS(X=X0, Y=Y0, W=W0, C=C0,funcVal=funcVal0, nSimulations=n)
#' 
#' @export 
computeMaxDifference_LS <- function(X, Y, W, C, funcVal, nSimulations) {
  originalValue <- funcVal # Assume funcVal has been computed with the full dataset
  
  differences <- numeric(nSimulations) # Vector to store differences
  
  for(i in 1:nSimulations) {
    nRemove <- 1
    
    # Perform the sensitivity analysis for each task
    modifiedValues <- sapply(1:length(X), function(k) {
      x <- X[[k]]
      y <- Y[[k]]
      w <- W[,k]
      
      # Randomly select rows to remove
      rowsToRemove <- sample(1:nrow(x), nRemove)
      xModified <- x[-rowsToRemove, ]
      yModified <- y[-rowsToRemove]
      
      # Recompute the weighted sum without the removed rows
      
      #weight <- 1 / length(yModified)
      #l <- - yModified * (xModified %*% w)
      #lp <- l
      #lp[lp < 0] <- 0
      #sum(weight * (log(exp(-lp) + exp(l - lp)) + lp))
      
      #funcVal <- sum(sapply(1:nTasks, function(k){
      0.5*mean((yModified - xModified %*% w)^2)
      
      #})) + C * norm(W, 'f')^2
      
      
    })
    
    # Sum the values across tasks and add the regularization term
    modifiedValue <- sum(modifiedValues) + C * norm(W, 'f')^2
    
    # Store the difference
    differences[i] <- abs(originalValue - modifiedValue)
  }
  
  # Find the maximum difference
  maxDifference <- max(differences)
  return(list(
    maxDifference = maxDifference,
    originalValue = originalValue
  ))
}








################################################################################
#' @title Sensitivity analysis for MTL with least-square loss and  network incorporation
#' @description Sensitivity analysis for MTL with least-square loss and  network incorporation: Compute the difference between an original objective function value and a modified objective function value obtained when removing one data point from the input data, perform this procedure for in total \code{nSimulations} data point removals and calculate the maximum of the differences over all \code{nSimulations} repetitions  
#' @param X A set of \eqn{T} \eqn{(N_t \times P)} feature matrices, \eqn{t \in \{1,\ldots,T\}}
#' @param Y A set of \eqn{T} response vectors \eqn{\in \mathbb{R}^{N_t}}, \eqn{t \in \{1,\ldots,T\}}
#' @param W A \eqn{(P \times T)} coefficient matrix
#' @param C The hyper-parameter associated with L2 term in the respective MTL model
#' @param funcVal The original function value
#' @param G Matrix describing the task relatedness
#' @param nSimulations number of runs in the sensitivity analysis involved in the differential privacy mechanism
#' 
#' @return A list of two elements named maxDifference (maximum difference between an original objective function value and a modified objective function value obtained when removing one data point from the input data, across \code{nSimulations} runs) and originalValue (original objective function value), respectively
#' 
#' @examples
#' set.seed(24)
#' X0<-list(matrix(rnorm(100000),nrow=200,ncol=500),matrix(rnorm(150000),nrow=300,ncol=500))
#' set.seed(24)
#' Y0<-list(rnorm(200),rnorm(300))
#' set.seed(24)
#' W0<-matrix(rnorm(1000),nrow=500,ncol=2)
#' C0<-1
#' funcVal0<-0.375
#' G0<-diag(2)-(1/2)
#' n<-100
#' maxdiff<-computeMaxDifference_LS_Net(X=X0, Y=Y0, W=W0, C=C0,funcVal=funcVal0,G=G0, nSimulations=n)
#' 
#' @export
computeMaxDifference_LS_Net <- function(X, Y, W, C, funcVal,G, nSimulations) {
  originalValue <- funcVal # Assume funcVal has been computed with the full dataset
  
  differences <- numeric(nSimulations) # Vector to store differences
  
  for(i in 1:nSimulations) {
    nRemove <- 1
    
    # Perform the sensitivity analysis for each task
    modifiedValues <- sapply(1:length(X), function(k) {
      x <- X[[k]]
      y <- Y[[k]]
      w <- W[,k]
      
      # Randomly select rows to remove
      rowsToRemove <- sample(1:nrow(x), nRemove)
      xModified <- x[-rowsToRemove, ]
      yModified <- y[-rowsToRemove]
      
      # Recompute the weighted sum without the removed rows
      
      #weight <- 1 / length(yModified)
      #l <- - yModified * (xModified %*% w)
      #lp <- l
      #lp[lp < 0] <- 0
      #sum(weight * (log(exp(-lp) + exp(l - lp)) + lp))
      
      #funcVal <- sum(sapply(1:nTasks, function(k){
      0.5*mean((yModified - xModified %*% w)^2)
      
      #})) + C * norm(W, 'f')^2
      
      
    })
    
    # Sum the values across tasks and add the regularization term
    modifiedValue <- sum(modifiedValues) + C * norm(W %*% G, 'f')^2
    
    # Store the difference
    differences[i] <- abs(originalValue - modifiedValue)
  }
  
  # Find the maximum difference
  maxDifference <- max(differences)
  return(list(
    maxDifference = maxDifference,
    originalValue = originalValue
  ))
}

