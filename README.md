# DPMTL
Differential privacy in multi-task learning approaches

# Description 
This package provides an implementation of differential privacy in several multi-task learning (MTL) approaches for classification and regression problems, respectively. Basically, it offers the non-federated versions of the approaches implemented in the R packages [dsMTLBase] (https://github.com/transbioZI/dsMTLBase) and [dsMTLClient] (https://github.com/transbioZI/dsMTLClient) for differential privacy in federated multi-task learning methods as introduced in Schefzik et al. (2025) as an add-on to the implemention presented in Cao et al. (2022).
 In particular, differential privacy implementations for the three different MTL approaches MTL_L21, MTL_Trace and MTL_Net as described in Schefzik et al. (2025) are provided, for regression and classification settings, respectively. 

# Contributors
Roman Schefzik aggregated, edited and finalized the code of the MTL and differential privacy functions, respectively. Han Cao provided the original implementation of the MTL functions. Xavier Escriba-Montagut and Juan R. Gonzalez contributed to the implemenation of the differential privacy mechanism.

# Installation
```R
install.packages("devtools")
library(devtools)
install_github("RomanSchefzik/DPMTL")
```
# Example of usage
Examples for a usage of the functions can be found in the respective function documentations.
Exemplarily, we here explicitly consider the usage of the MTL_L21 implementation in a classification setting.
```R
library(DPMTL)
#creation of synthetic predictor matrices and response vectors and setting of the model hyperparameters
set.seed(24)
X0<-list(matrix(rnorm(100000),nrow=200,ncol=500),matrix(rnorm(150000),nrow=300,ncol=500))
set.seed(24)
Y0<-list(sample(c(1,-1),200,replace=TRUE),sample(c(1,-1),300,replace=TRUE))
lam0<-0.05
C0<-1
#non-private model (i.e., model without using differential privacy)
opts0<-list(init=0,w0=NULL,tol=0.01,maxIter=50,ter=2,diffPrivEpsilon=NULL,nRunsSensitAn=NULL)
model.nonprivate<-LR_MTL_L21(X=X0,Y=Y0,lam=lam0,C=C0,opts=opts0)
#example for a private model based on a privacy parameter of 0.7 and 100 runs for the involved sensitivity analyses
opts1<-list(init=0,w0=NULL,tol=0.01,maxIter=50,ter=2,diffPrivEpsilon=0.7,nRunsSensitAn=100)
set.seed(24)
model.private<-LR_MTL_L21(X=X0,Y=Y0,lam=lam0,C=C0,opts=opts1)
``` 
# References
Cao, H., Zhang, Y., Baumbach, J., Burton, P. R., Dwyer, D., Koutsouleris, N., Matschinske, J., Marcon, Y., Rajan, S., Rieg, T., Ryser-Welch, P., Späth, J., The COMMITMENT Consortium, Herrmann, C., and Schwarz, E. (2022). dsMTL: a computational framework for privacy-preserving, distributed multi-task machine learning. Bioinformatics, 38(21), 4919-4926. DOI: 10.1093/bioinformatics/btac616

Schefzik, R., Cao, H., Rajan, S., Escriba-Montagut, X., Gonzalez, J. R., and Schwarz, E. (2025). Integrating differential privacy into federated multi-task learning algorithms in dsMTL.



