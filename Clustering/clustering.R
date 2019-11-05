library(ggpubr)
library(ggplot2)      ## using packge for plotting graphs
library(Rtsne)        ## package for t-sne
library(inaparc)
library(fossil)

data <- read.delim("iyer.txt", header= FALSE)   #reading file from computer
cols<- ncol(data)            #calculating number of columns
rows<-nrow(data)            #calculating number of rows
data<-na.omit (data)          #omitting invalid values
gt<- data[2]                #ground truth column
data <- data[3:cols]
data=as.matrix(data) # Turn into a matrix


### K-means algorithm
centers<- forgy(data, 10)$v  #initialize centers with forgy

euclidean <- function(point1, point2) {
    disMat <- matrix(NA, nrow=dim(point1)[1], ncol=dim(point2)[1])
    for(i in 1:nrow(point2)) {
        disMat[,i] <- sqrt(rowSums(t(t(point1)-point2[i,])^2))
    }
    disMat
}

K_means <- function(x, centers, distance, iterations) {
    for(i in 1:iterations) {
        distsToCenters <- distance(x, centers)
        clusters <- apply(distsToCenters, 1, which.min)
        centers <- apply(x, 2, tapply, clusters, mean)
    }
    clusters
}

kmn <- K_means(data, centers, euclidean, 10)



### hierarchial agglomorative algorithm with single link
disMatrix<- function(x)  #distance matrix function
{
    x<- as.matrix(x)
    dis<- apply(x*x,1,sum) %*% matrix(1.0,1,nrow(x))
    sqrt(abs(dis + t(dis) - 2 * x %*% t(x)))
}

sor<- function(m)
{
    N<- nrow(m) + 1
    sor<- rep(0,N)
    sor[1]<- m[N-1,1]
    sor[2]<- m[N-1,2]
    loc<- 2
    for(i in seq(N-2,1))
    {
        for(j in seq(1,loc))
        {
            if(sor[j] == i)
            {
                sor[j]<- m[i,1]
                if(j==loc)
                {
                    loc<- loc + 1
                    sor[loc]<- m[i,2]
                } else
                {
                    loc<- loc + 1
                    for(k in seq(loc, j+2)) sor[k]<- sor[k-1]
                    sor[j+1]<- m[i,2]
                }
            }
        }
    }
    -sor
}

# HAC algorithm
hc<- function(d)
{
    if(!is.matrix(d)) d = as.matrix(d)
    N = nrow(d)
    diag(d)=Inf
    n = -(1:N)
    m = matrix(0,nrow=N-1, ncol=2)
    h = rep(0,N-1)
    for(j in seq(1,N-1))
    {
        # minimum
        h[j] = min(d)
        i = which(d - h[j] == 0, arr.ind=TRUE)
        i = i[1,,drop=FALSE]
        p = n[i]
        p = p[order(p)]
        m[j,] = p
        grp = c(i, which(n %in% n[i[1,n[i]>0]]))
        n[grp] = j
        r = apply(d[i,],2,min)
        d[min(i),] = d[,min(i)] = r
        d[min(i),min(i)]        = Inf
        d[max(i),] = d[,max(i)] = Inf
    }
    structure(list(merge = m, height = h, order = sor(m),
                   labels = rownames(d), dist.method = "euclidean"),
              class = "hclust")
}
h1 = hc(disMatrix(data))
hac <- cutree(h1, 10)



### spectral clustering algorithm
s <- function(x1, x2, alpha=1) {
    exp(- alpha * norm(as.matrix(x1-x2), type="F"))
}

similarity <- function(data, similarity) {
    N <- nrow(data)
    S <- matrix(rep(NA,N^2), ncol=N)
    for(i in 1:N) {
        for(j in 1:N) {
            S[i,j] <- similarity(data[i,], data[j,])
        }
    }
    S
}

S <- similarity(data, s)

affinity <- function(S, neighboors=2) {
    N <- length(S[,1])

    if (neighboors >= N) {
        A <- S
    } else {
        A <- matrix(rep(0,N^2), ncol=N)
        for(i in 1:N) {
            similarities <- sort(S[i,], decreasing=TRUE)[1:neighboors]
            for (s in similarities) {
                j <- which(S[i,] == s)
                A[i,j] <- S[i,j]
                A[j,i] <- S[i,j]
            }
        }
    }
    A
}

A <- affinity(S, 3)
D <- diag(apply(A, 1, sum))
U <- D - A
k   <- 10
evL <- eigen(U, symmetric=TRUE)
Z   <- evL$vectors[,(ncol(evL$vectors)-k+1):ncol(evL$vectors)]
spc <- kmeans(Z, centers=k)$cluster


val<-rand.index(unlist(gt),kmn)
val1<-rand.index(unlist(gt),hac)
val2<-rand.index(unlist(gt),spc)
print(val)
print(val1)
print(val2)


data<-matrix(as.numeric(unlist(data)),nrow=nrow(data)) #converting df to matrix

xbar<-colMeans(data)        #calculating mean of columns
for (r in 1:nrow(data)){
    for(c in 1:ncol(data)){
        data[r,c]<-data[r,c]-xbar[c]       #calulating X-Xbar
    }
}

covM<-(t(data) %*% (data))/(nrow(data)-1)    #calculating covariance matrix
ev<-eigen(covM)$vectors[,1:2]            #calculating eigenvectors and choosing two
newMat<- (data) %*% (ev)                 #mapping matrix on new principal components
newMat<-data.frame(newMat)
km<- cbind(newMat,kmn)         #appending cluster
colnames(km)<-c("first_PC","second_PC","cluster")
p1<-ggplot(km, aes(x = first_PC, y = second_PC, colour = kmn)) + geom_point()
p1<- p1 + labs(title = "PCA plot for k-means") + theme(plot.title = element_text(hjust = 0.5))
print(p1)

hier<- cbind(newMat,hac)         #appending cluster
colnames(hier)<-c("first_PC","second_PC","cluster")
p2<-ggplot(hier, aes(x = first_PC, y = second_PC, colour = hac)) + geom_point()
p2<- p2 + labs(title = "PCA plot for hierarchial agglomorative single") + theme(plot.title = element_text(hjust = 0.5))
print(p2)

spec<- cbind(newMat,spc)         #appending cluster
colnames(spec)<-c("first_PC","second_PC","cluster")
p3<-ggplot(spec, aes(x = first_PC, y = second_PC, colour = spc)) + geom_point()
p3<- p3 + labs(title = "PCA plot for spectral clustering") + theme(plot.title = element_text(hjust = 0.5))
print(p3)
