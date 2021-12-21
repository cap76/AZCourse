# Solutions to Chapter 4 - Linear regression and logistic regression {#solutions-logistic-regression}

Solutions to exercises of chapter \@ref(logistic-regression).

Exercise 1.1. Before we begin, we first need to visualise the data as a whole. Heatmaps are one way of looking at large datasets. Since we're looking for differences I will make a heatmap of the difference between control and infected at each time point and subcluster by pattern:


```r
#install.packages("pheatmap")
library(pheatmap)
DeltaVals <- t(D[25:48,3:164] - D[1:24,3:164])
pheatmap(DeltaVals, cluster_cols = FALSE, cluster_rows = TRUE)
```

we can see a number of rows in which there appears to be large scale changes as the time series progresses. Going forward we can pick one where this is particularly strong.

Exercise 1.1. We can systematically fit a model with increasing degree and evaluate/plot the RMSE on the held out data.


```r
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
```
From these plots it looks like the best model is one with degree $d=2$ or $d=4$, suggesting there is a lot more complexity to this gene. You can clean the code up to make it run in a loop. Hint: you can not directly pass a variable over to poly (y~poly(x,i) will not work) and will have to convert to a function:

```r
setdegree <- 5
f <- bquote( y~poly(x,degree=.(setdegree) ) )
lrfit11 <- train( as.formula(f) , data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
```

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


model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(1)) %>% 
  layer_dense(units = 10, activation = "relu") %>% 
  layer_dense(units = 20, activation = "relu") %>% 
  layer_dense(1, activation = "linear")

model %>% compile(loss = "mse", optimizer = "adam", metrics = "mse")

cp_callback <- callback_model_checkpoint(filepath = 'data/RickandMorty/data/models/densemodel.h5', save_weights_only = FALSE, mode = "auto",  monitor = "val_mse", verbose = 0)


tensorflow::set_random_seed(42)
model %>% fit(x = trainingX, y = trainingY, validation_data = list(testingX, testingY), epochs = 100, verbose = 2,  callbacks = list(cp_callback))

model = load_model_hdf5('data/RickandMorty/data/models/densemodel.h5')


xstar <- seq(0,200,by=0.5)
forecastY <- model %>% predict(xstar)
plot(xstar,forecastY,'l')
lines(xstar,sqrt(xstar),col="red")
```

For comparison we can also use linear regression to compare our predictions:


```r
colnames(trainingX) <- "x"
colnames(trainingY) <- "y"
lrfit <- lm(y~x)
newd <- data.frame(x=newX)
predictedValues<-predict.lm(lrfit, newdata = newd)
lines(newX,predictedValues, col="red")
```

Excercsie 2.1: The network architecture should be fine for this task. However a noisy version of the input data will have to be generated (e.g., by setting a random set of pixels to zero) to be passed in to the AE. A clean version of the data should be retained and passed to the AE as the output. 


As a final model I have included a modification of a variational autoencoder on the Rick and Morty dataset based on the one [here](https://tensorflow.rstudio.com/guide/keras/examples/variational_autoencoder_deconv/)


```r
K <- keras::backend()

# input image dimensions and other parameters
img_rows <- 90L
img_cols <- 160L
img_chns <- 3L
filters <- 64L
num_conv <- 3L
latent_dim <- 2L
intermediate_dim <- 128L
epsilon_std <- 1.0
batch_size <- 100L
epochs <- 5L


original_img_size <- c(img_rows, img_cols, img_chns)
x <- layer_input(shape = c(original_img_size))
conv_1 <- layer_conv_2d(x, filters = img_chns, kernel_size = c(2L, 2L), strides = c(1L, 1L), padding = "same", activation = "relu")
conv_2 <- layer_conv_2d(conv_1, filters = filters, kernel_size = c(2L, 2L), strides = c(2L, 2L), padding = "same", activation = "relu")
conv_3 <- layer_conv_2d(conv_2, filters = filters, kernel_size = c(num_conv, num_conv), strides = c(1L, 1L), padding = "same", activation = "relu")
conv_4 <- layer_conv_2d(conv_3, filters = filters, kernel_size = c(num_conv, num_conv), strides = c(1L, 1L), padding = "same", activation = "relu")
flat <- layer_flatten(conv_4)
hidden <- layer_dense(flat, units = intermediate_dim, activation = "relu")
z_mean <- layer_dense(hidden, units = latent_dim)
z_log_var <- layer_dense(hidden, units = latent_dim)

sampling <- function(args) {
  z_mean <- args[, 1:(latent_dim)]
  z_log_var <- args[, (latent_dim + 1):(2 * latent_dim)]
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean = 0.,
    stddev = epsilon_std
  )
  z_mean + k_exp(z_log_var) * epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)

output_shape <- c(batch_size, 45L, 80L, filters)

decoder_hidden <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_upsample <- layer_dense(units = prod(output_shape[-1]), activation = "relu")

decoder_reshape <- layer_reshape(target_shape = output_shape[-1])
decoder_deconv_1 <- layer_conv_2d_transpose(filters = filters,kernel_size = c(num_conv, num_conv),strides = c(1L, 1L),padding = "same",activation = "relu")
decoder_deconv_2 <- layer_conv_2d_transpose(filters = filters,kernel_size = c(num_conv, num_conv),strides = c(1L, 1L),padding = "same",activation = "relu")
decoder_deconv_3_upsample <- layer_conv_2d_transpose(filters = filters,kernel_size = c(3L, 3L),strides = c(2L, 2L),padding = "valid",activation = "relu")
decoder_mean_squash <- layer_conv_2d(filters = img_chns,kernel_size = c(2L, 2L),strides = c(1L, 1L),padding = "valid",activation = "sigmoid")

hidden_decoded <- decoder_hidden(z)
up_decoded <- decoder_upsample(hidden_decoded)
reshape_decoded <- decoder_reshape(up_decoded)
deconv_1_decoded <- decoder_deconv_1(reshape_decoded)
deconv_2_decoded <- decoder_deconv_2(deconv_1_decoded)
x_decoded_relu <- decoder_deconv_3_upsample(deconv_2_decoded)
x_decoded_mean_squash <- decoder_mean_squash(x_decoded_relu)

# custom loss function
vae_loss <- function(x, x_decoded_mean_squash) {
  x <- k_flatten(x)
  x_decoded_mean_squash <- k_flatten(x_decoded_mean_squash)
  xent_loss <- 1.0 * img_rows * img_cols *
    loss_binary_crossentropy(x, x_decoded_mean_squash)
  kl_loss <- -0.5 * k_mean(1 + z_log_var - k_square(z_mean) -
                           k_exp(z_log_var), axis = -1L)
  k_mean(xent_loss + kl_loss)
}

## variational autoencoder
vae <- keras_model(x, x_decoded_mean_squash)
vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
summary(vae)

## encoder: model to project inputs on the latent space
encoder <- keras_model(x, z_mean)

## build a digit generator that can sample from the learned distribution
gen_decoder_input <- layer_input(shape = latent_dim)
gen_hidden_decoded <- decoder_hidden(gen_decoder_input)
gen_up_decoded <- decoder_upsample(gen_hidden_decoded)
gen_reshape_decoded <- decoder_reshape(gen_up_decoded)
gen_deconv_1_decoded <- decoder_deconv_1(gen_reshape_decoded)
gen_deconv_2_decoded <- decoder_deconv_2(gen_deconv_1_decoded)
gen_x_decoded_relu <- decoder_deconv_3_upsample(gen_deconv_2_decoded)
gen_x_decoded_mean_squash <- decoder_mean_squash(gen_x_decoded_relu)
generator <- keras_model(gen_decoder_input, gen_x_decoded_mean_squash)

vae %>% fit(trainX, trainX, shuffle = TRUE, epochs = epochs, batch_size = batch_size, validation_data = list(valX, valX))
```

Which we can visualise:


```r
## display a 2D plot of the digit classes in the latent space
x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)
x_test_encoded %>%
  as_data_frame() %>%
  mutate(class = as.factor(mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()

## display a 2D manifold of the digits
n <- 15  # figure with 15x15 digits
digit_size <- 28

# we will sample n points within [-4, 4] standard deviations
grid_x <- seq(-4, 4, length.out = n)
grid_y <- seq(-4, 4, length.out = n)

rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = digit_size))
  }
  rows <- cbind(rows, column)
}
rows %>% as.raster() %>% plot()
```
