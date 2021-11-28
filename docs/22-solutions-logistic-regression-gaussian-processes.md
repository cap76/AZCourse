# Solutions to Chapter 4 - Linear regression and logistic regression {#solutions-logistic-regression}

Solutions to exercises of chapter \@ref(logistic-regression).

Exercise 1.1. Before we begin, we first need to visualise the data as a whole. Heatmaps are one way of looking at large datasets. Since we're looking for differences I will make a heatmap of the difference between control and infected at each time point and subcluster by pattern:


```r
library(pheatmap)
DeltaVals <- t(D[25:48,3:164] - D[1:24,3:164])
pheatmap(DeltaVals, cluster_cols = FALSE, cluster_rows = TRUE)
```
we can see a number of rows in which there appears to be large scale changes as the time series progresses. Pick one where this is particularly strong.

Exercise 1.1. We can systematically fit a model with increasing degree and evaluate/plot the RMSE on the held out data.


```r
library(pheatmap)
RMSE <- rep(NULL, 10)
lrfit1 <- train(y~poly(x,degree=1), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[1] <- lrfit1$results$RMSE
lrfit2 <- train(y~poly(x,degree=2), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[2] <- lrfit2$results$RMSE
lrfit3 <- train(y~poly(x,degree=3), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[3] <- lrfit3$results$RMSE
lrfit4 <- train(y~poly(x,degree=4), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[4] <- lrfit4$results$RMSE
lrfit5 <- train(y~poly(x,degree=5), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[5] <- lrfit5$results$RMSE
lrfit6 <- train(y~poly(x,degree=6), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[6] <- lrfit6$results$RMSE
lrfit7 <- train(y~poly(x,degree=7), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[7] <- lrfit7$results$RMSE
lrfit8 <- train(y~poly(x,degree=8), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[8] <- lrfit8$results$RMSE
lrfit9 <- train(y~poly(x,degree=9), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[9] <- lrfit9$results$RMSE
lrfit10 <- train(y~poly(x,degree=10), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[10] <- lrfit10$results$RMSE

plot(RMSE)
plot(RMSE[1:5])
#barplot(c(lrfit2$results$RMSE,lrfit3$results$RMSE,lrfit4$results$RMSE))
```
From these plots it looks like the best model is one with degree $d=2$ or $d=4$, suggesting there is a lot more complexity to this gene.


# Solutions to Chapter 5 - Neural Networks

Excersie 2.1: We increase the training size and tweak network structure in various ways.


```r
tdims <- 5000 #Number of samples to generate
x <-  runif(tdims, min=0, max=100) #Generate random x in range 0 to 100
y <- sqrt(x) #Calculate square root of x

trainingX  <- array(0, dim=c(tdims,1)) #Store data as an array (required by Keras)
trainingX[1:tdims,1] <- x
trainingY  <- array(0, dim=c(tdims,1))
trainingY[1:tdims,1] <- y

#Now do the same but for a independently generated test set
x <-  runif(tdims, min=0, max=100)
y <- sqrt(x)

testingX  <- array(0, dim=c(tdims,1)) #Store as arrays
testingX[1:tdims,1] <- x
testingY  <- array(0, dim=c(tdims,1))
testingY[1:tdims,1] <- y

mod <- Sequential()
mod$add(Dense(10, input_shape = c(1)))
mod$add(Activation("relu"))
mod$add(Dense(20))
mod$add(Activation("relu"))
mod$add(Dense(1))
mod$add(Activation("linear"))

keras_compile(mod,  loss = 'mean_squared_error', metrics = c('mean_squared_error'), optimizer = RMSprop())

set.seed(12345)
keras_fit(mod, trainingX, trainingY, validation_data = list(testingX, testingY), batch_size = 1000, epochs = 450, verbose = 1)

newX <- as.matrix(seq(from = 0, to = 200, by = 5))
predY <- keras_predict(mod, x = newX)
plot(newX,predY)
lines(newX,sqrt(newX))
```

For comparison we can also use linear regression to compare our predictions:


```r
colnames(trainingX) <- "x"
colnames(trainingY) <- "y"
lrfit <- lm(y~x)
newd <- data.frame(x=newX)
predictedValues<-predict.lm(lrfit, newdata = newd)
#RMSE = sqrt( mean((testingY - predictedValues)^2) )
lines(newX,predictedValues, col="red")
```

Excercsie 2.1: The network architecture should be fine for this task. However a noisy version of the input data will have to be generated (e.g., by setting a random set of pixels to zero) to be passed in to the AE. A clean version of the data should be retained and passed to the AE as the output. 


