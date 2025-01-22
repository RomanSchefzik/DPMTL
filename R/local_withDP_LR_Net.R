################################################################################
#' @title Solver of MTL with logistic loss and network incorporation
#' @description Solver of MTL with logistic loss and network incorporation
#' @param X A set of feature matrices
#' @param Y A set of binary responses \eqn{\in\{1, -1\}}
#' @param lam The hyper-parameter controlling the sparsity   
#' @param C   The hyper-parameter associated with L2 term
#' @param G   Matrix describing the task relatedness
#' @param opts A list of options controlling the optimization procedure and specifying the differential privacy component. See Details.     
#' 
#' @return The converged result of optimization
#' @details Solver of MTL with logistic loss and network incorporation. \cr A list of options controlling the optimization procedure and specifying the differential privacy component based on a privacy parameter epsilon>0 can be provided via the \code{opts} argument:
#' \itemize{
#' \item{\code{init} \cr A value (0 or 1) specifying the starting point of the involved gradient descent algorithm. Specifically, two options are provided: A value of 0 (default) uses the 0 matrix as starting point, while a value of 1 uses a starting point specified by the user. If applicable (i.e., \code{init=1}), the user-defined starting point (matrix) has to be specified via the \code{w0} argument in \code{opts}.}
#' \item{\code{w0} \cr A user-defined starting point (matrix) for the involved gradient descent algorithm, in case \code{init=1} is specified (otherwise, for \code{init=0}, the value is set to NULL, the default).}
#' \item{\code{tol} \cr A value >0 specifying the tolerance of the acceptable precision of solution to terminate the algorithm. Default value is set to 0.01.}
#' \item{\code{maxIter} \cr A value >0 specifying the maximum number of iterations. Default value is set to 50.}
#' \item{\code{ter} \cr A value (1, 2 or 3) specifying one out of three termination rules to determine whether the optimization converges. The first rule (\code{ter=1}) checks whether the current objective value is close enough to 0. The second rule (\code{ter=2}) considers the last two objective values and checks whether the decrement is close enough to 0 (default). The third rule (\code{ter=3}) allows the optimization to be performed for a certain maximum number of iterations (\code{maxIter}).}
#' \item{\code{diffPrivEpsilon} \cr A value >0 serving as a privacy parameter to control the degree of differential privacy. Setting the value to NULL (default) means that no differential privacy is included.}
#' \item{\code{nRunsSensitAn} \cr A value >0 specifying the number of simulation runs for the respective sensitivity analyses in case a differential privacy component is included. Default value is set to 100 (only effective if a corresponding value for \code{diffPrivEpsilon} is provided; otherwise, no differential privacy is included).}
#' }
#' 
#' @examples
#' set.seed(24)
#' X0<-list(matrix(rnorm(100000),nrow=200,ncol=500),matrix(rnorm(150000),nrow=300,ncol=500))
#' set.seed(24)
#' Y0<-list(sample(c(1,-1),200,replace=TRUE),sample(c(1,-1),300,replace=TRUE))
#' lam0<-0.05
#' C0<-1
#' G0<-diag(2)-(1/2)
#' #non-private model
#' opts0<-list(init=0,w0=NULL,tol=0.01,maxIter=50,ter=2,diffPrivEpsilon=NULL,nRunsSensitAn=NULL)
#' model.nonprivate<-LR_MTL_Net(X=X0,Y=Y0,lam=lam0,C=C0,G=G0,opts=opts0)
#' #example for a private model
#' opts1<-list(init=0,w0=NULL,tol=0.01,maxIter=50,ter=2,diffPrivEpsilon=0.7,nRunsSensitAn=100)
#' set.seed(24)
#' model.private<-LR_MTL_Net(X=X0,Y=Y0,lam=lam0,C=C0,G=G0,opts=opts1)
#' 
#' @export 
LR_MTL_Net <- function (X, Y, lam, C, G, opts){
  
  #min_W sum_k{log(1+exp(-Y_k*X_k*W_k))} +lam||W||_21 + c||W||^2_2 
  #with starting points: 0 or W_0; Data: X and Y; Hyper-parameters: lam, C
  
  proximal_l1 <- function (W, lambda ){
    p <- abs(W) - lambda
    p=p*(p>0)
    Wp <- sign(W) * p
    return(Wp)
  }
  
  nonsmooth_eval <- function (W){
    return(lam*sum(abs(W)))
  }  
  
  LR_iter_update_tasks <- function(W){
    rData=lapply(1:nTasks, function(k){
      x=X[[k]]; y=Y[[k]];w=W[,k]
      weight <- 1/length(y)
      l <- -y*(x %*% w) 
      lp <- l
      lp[lp<0] <- 0
      funcVal <- sum(weight * ( log( exp(-lp) +  exp(l-lp) ) + lp ))
      b <- (-weight*y)*(1 - 1/ (1+exp(l)))
      grad_w <- (t(x) %*% b)
      return(list(grad_w=grad_w, funcVal=funcVal))
    })
    
    if(is.null(opts$diffPrivEpsilon)==FALSE){
    
    grad_Ws=sapply(rData, function(x)x$grad_w) + 2* C * W %*% GGt
    funcVals=sum(sapply(rData, function(x)x$funcVal)) + C * norm(W %*% G, 'f')^2
    
    maxGradWsDifference <- simulateDifferences_LR_Net(rData, C, W, X, Y,G,nSimulations=opts$nRunsSensitAn)
    
    grad_Ws <- differential_privacy(
      list(
        og_value = grad_Ws,
        l1_sens = maxGradWsDifference
      ),
      opts$diffPrivEpsilon
    )
    
    }else{
      grad_Ws=sapply(rData, function(x)x$grad_w) + 2* C * W %*% GGt
      funcVals=sum(sapply(rData, function(x)x$funcVal)) + C * norm(W %*% G, 'f')^2
    }
    
    return(list(grad_Ws=grad_Ws, funcVals=funcVals))
  }
  
  LR_funcVal_eval_tasks <- function (W){
    
    if(is.null(opts$diffPrivEpsilon)==FALSE){
      
    funcVal <- sum(sapply(1:nTasks, function(k){
      x=X[[k]]; y=Y[[k]];w=W[,k]
      weight <- 1/length(y)
      l <- - y*(x %*% w)
      lp <- l
      lp[lp<0] <- 0
      return(sum(weight * ( log( exp(-lp) +  exp(l-lp) ) + lp )))
    })) + C * norm(W%*%G, 'f')^2
    
    maxDifference <- computeMaxDifference_LR_Net(X, Y, W, C, funcVal,G,nSimulations=opts$nRunsSensitAn)
    
    return(list(
      og_value = maxDifference$originalValue,
      l1_sens = maxDifference$maxDifference
    ))
    
    }else{funcVal<-sum(sapply(1:nTasks, function(k){
      x=X[[k]]; y=Y[[k]];w=W[,k]
      weight <- 1/length(y)
      l <- - y*(x %*% w)
      lp <- l
      lp[lp<0] <- 0
      return(sum(weight * ( log( exp(-lp) +  exp(l-lp) ) + lp )))
    })) + C * norm(W%*%G, 'f')^2}
    return(funcVal)
    
  }
  
  #################################  
  # Main algorithm
  #################################  
  Obj <- vector(); 
  nFeats=ncol(X[[1]])
  nTasks=length(X)
  GGt <- G %*% t(G)
  log.niterCall=0; log.nfuncCall=0
  
  #initialize a starting point
  if(opts$init==0){
    w0=matrix(0,nrow=nFeats, ncol=length(X))
  }else if(opts$init==1){
    w0 <- opts$w0
  }    
  
  bFlag <- 0; 
  wz <- w0;
  wz_old <- w0;
  
  t <- 1;
  t_old <- 0;
  iter <- 0;
  gamma <- 1;
  gamma_inc <- 2;
  
  while (iter < opts$maxIter){
    alpha <- (t_old - 1) /t;
    ws <- (1 + alpha) * wz - alpha * wz_old;
    
    iter_update=LR_iter_update_tasks(ws)
    Gws <- iter_update$grad_Ws
    Fs <- iter_update$funcVals
    log.niterCall=log.niterCall+1
    
    # the Armijo Goldstein line search scheme
    while (TRUE){
      wzp <- proximal_l1(ws - Gws/gamma, lam / gamma)
      Fzp=LR_funcVal_eval_tasks(wzp)
      if(is.null(opts$diffPrivEpsilon)==FALSE){
        Fzp <- differential_privacy(Fzp, opts$diffPrivEpsilon)
      }else{Fzp<-Fzp}
      log.nfuncCall=log.nfuncCall+1
      
      delta_wzp <- wzp - ws;
      r_sum <- norm(delta_wzp, 'f')^2
      
      #second order approximation
      Fzp_gamma = Fs + sum(delta_wzp * Gws) + gamma * r_sum/2
      
      if (r_sum <=1e-20){
        bFlag=1; 
        break;
      }
      
      if (Fzp <= Fzp_gamma) break else {gamma = gamma * gamma_inc}
    }
    
    wz_old = wz; wz = wzp; Obj = c(Obj, Fzp + nonsmooth_eval(wz));
    
    #test stop condition.
    if (bFlag) break;
    if (iter>=2){
      if (opts$ter==1 & abs( Obj[length(Obj)] - Obj[length(Obj)-1] ) <= opts$tol){
        break
      } else if(opts$ter==2 & abs( Obj[length(Obj)] - Obj[length(Obj)-1] ) <= opts$tol*Obj[length(Obj)-1]){
        break
      } else if(opts$ter==3 & iter==opts$maxIter){
        break
      } 
    }
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
  }
  return(list(W=wzp, Obj=Obj, Logs=c(log.niterCall, log.nfuncCall), gamma=gamma))
}



