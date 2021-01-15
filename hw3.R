library(DataExplorer)
library(ggcorrplot)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(moments)
library(caTools)
library(MASS)
library(Hmisc)
library(Metrics)
library(leaps)
library(sp)
library(class)
install.packages("caret")
library(caret)
library(tidyverse)
#install.packages("rsample")
library(rsample)
#install.packages("pls")
library(pls)



setwd("/Users/shifa/Downloads/tic")

#First explore the data set and doing some visualization to get idea about data set

train_data = read.table("ticdata2000.txt")
test_data = read.table("ticeval2000.txt")
head(train_data)
head(test_data)
dim(train_data)
dim(test_data)
View(train_data)
View(test_data)

sum(is.na(train_data)) # no missing values found
sum(is.na(test_data)) # no missing values found

train_data = rename(train_data, customer_main_type = V5)
train_data = rename(train_data, customer_age = V4)
train_data = rename(train_data, caravan_insurance = V86)
View(train_data)

#we would like to see how many 0's and 1's
count_0_1 = table(train_data$caravan_insurance)
count_0_1

colors=c("blue","red")
col=colors
pie(count_0_1, labels = count_0_1, main = "Customers of Caravan Insurance",col=colors)
box()

# first Would like to see how customer main_type doing on caravan insurance
customer_main_buying <- table(train_data$customer_main_type[train_data$caravan_insurance == '1'])
customer_main_buying

barplot <- barplot(customer_main_buying, main = "Customer main type Buying Caravan Insurance", col = "red", xlab = "customer main type" , ylim = c(0, 125), ylab = "count")                               
text(x = barplot,  y = customer_main_buying + 5, labels = customer_main_buying)


customer_main_not_buying <- table(train_data$customer_main_type[train_data$caravan_insurance == '0'])
customer_main_not_buying

barplot_not <- barplot(customer_main_not_buying, main = "Customer main type not Buying Caravan Insurance", col = "red", xlab = "customer main type" , ylim = c(0, 2000), ylab = "count")                               
text(x = barplot_not,  y = customer_main_not_buying + 60, labels = customer_main_not_buying)


# age on caravan insurance

age_notbuying <- table(train_data$customer_age[train_data$caravan_insurance == '0'])
age_notbuying

barplot1 <- barplot(age_notbuying, main = "Age not Buying Caravan Insurance", col = "blue", xlab = "age" , ylim = c(0, 3500), ylab = "count")                               
text(x = barplot1,  y = age_notbuying + 90, labels = age_notbuying)

age_buying <- table(train_data$customer_age[train_data$caravan_insurance == '1'])
age_buying

barplot2 <- barplot(age_buying, main = "Age Buying Caravan Insurance", col = "blue", xlab = "age" , ylim = c(0, 250), ylab = "count")                               
text(x = barplot2,  y = age_buying + 10, labels = age_buying)


#splitting the train dataset into training and testing
set.seed(1)
split = sample.split(train_data$caravan_insurance, SplitRatio = 0.8)
training_data = subset(train_data, split == TRUE)
test_data = subset(train_data, split == FALSE)

View(training_data)
View(test_data)
dim(train_data)
dim(test_data)

# Now will fit linear model and calculate MSE  on training data
length_1 = length(which(training_data$caravan_insurance ==1))
length_1

length_2 = length(which(test_data$caravan_insurance == 1))
length_2

train_model = lm(caravan_insurance ~., data= training_data)
summary(train_model)
predi_train = predict(train_model, training_data) #given the training data you predict
class1<-ifelse(predi_train > 0.5,1,0) 
class1
length(which(predi_train==training_data$caravan_insurance & predi_train ==1))

# MSE error for training data
mse_training = sum((training_data$caravan_insurance - predi_train)^2)
mse_training
rmse_train = sqrt(mse_training)
rmse_train

#now would like to see how it's performing on the test set 

predi_test = predict(train_model, test_data)
class_test <- ifelse(predi_test > 0.5, 1, 0)
class_test
length(which(predi_test == test_data$caravan_insurance & predi_test == 1))


mse_test = sum((test_data$caravan_insurance - predi_test))
mse_test

rmse_test = sqrt(mse_test)
rmse_test


# doing forward selection 
set.seed(123)
# Set up k-fold cross-validation
k_fold_train <- trainControl(method = "cv", number = 10)
# Train the model
forward_model <- train(caravan_insurance ~., data = training_data,
                    method = "leapForward", 
                    tuneGrid = data.frame(nvmax = 1:50),
                    trControl = k_fold_train
)
forward_model$results
forward_model$bestTune
sumss = summary(forward_model$finalModel)
sumss

par(mfrow=c(1,1))
plot(sumss$cp ,xlab="Number of Variables ",ylab="cp",type="l")
points(11,sumss$cp[11], col="red",cex=2,pch=20)

plot(sumss$bic ,xlab="Number of Variables ",ylab="bic",type="l")
points(6,sumss$bic[5], col="red",cex=2,pch=20)

plot(sumss$adjr2 ,xlab="Number of Variables ",ylab="adjustedR",type="l")
points(11,sumss$adjr2[11], col="red",cex=2,pch=20)


# Doing backward selection 
# Train backward model
backward_model <- train(caravan_insurance ~., data = training_data,
                       method = "leapBackward", 
                       tuneGrid = data.frame(nvmax = 1:50),
                       trControl = k_fold_train
)
backward_model$results
backward_model$bestTune
sum_backward = summary(backward_model$finalModel)
sum_backward

par(mfrow=c(1,1))
plot(sum_backward$cp ,xlab="Number of Variables ",ylab="cp",type="l")
points(6, sum_backward$cp[6], col="red",cex=2,pch=20)

plot(sum_backward$bic ,xlab="Number of Variables ",ylab="bic",type="l")
points(6,sum_backward$bic[6], col="red",cex=2,pch=20)

plot(sum_backward$adjr2 ,xlab="Number of Variables ",ylab="adjustedR",type="l")
points(6,sum_backward$adjr2[6], col="red",cex=2,pch=20)


# Ridge regression 

install.packages("glmnet")
library(glmnet)

x = training_data[, 1:85]
x = as.matrix(x)
View(x)
head(x)

y = training_data[, 86]
View(y)
head(y)

ridge.mod = cv.glmnet(x, y, alpha = 0)
#coef(ridge.mod, s = ridge.mod$lambda.min)
dim(coef(ridge.mod))
plot(ridge.mod)

opt_lambda <- ridge.mod$lambda.min
opt_lambda

w = test_data[, 1:85]
dim(w)
w = as.matrix(w)
head(w)
view(w)

v = test_data[, 86]
view(v)

pred_ridge = predict(ridge.mod, w, s=ridge.mod$lambda.min)
pred_ridge


mse_ridge = mean((v - pred_ridge )^2)
mse_ridge

class_ridge <- ifelse(pred_ridge > 0.5, 1, 0)
class_ridge
length(which(pred_ridge == v & pred_ridge == 1))



# Lasso Regression

lasso.mod = cv.glmnet(x, y, alpha = 1)
dim(coef(lasso.mod))
plot(lasso.mod)
lasso.mod$lambda.min

pred_lasso = predict(lasso.mod, w, s=lasso.mod$lambda.min)
pred_lasso

mse_lasso = mean((v - pred_lasso)^2)
mse_lasso

class_lasso <- ifelse(pred_lasso > 0.5, 1, 0)
#class_ridge
length(which(pred_lasso == v & pred_lasso == 1))



#normalize <- function(x) 
#{
#  return ((x - min(x)) / (max(x) - min(x)))
#}

#norm_train = as.data.frame(lapply(train_data[1:86], normalize))
#norm_test = as.data.frame(lapply(test_data[1:85], normalize))



# HOMEWORK PROBLEM #2

set.seed(10)
dataset = replicate(n = 20, expr = rnorm(n = 1000))
View(dataset)
dim(dataset)

#𝑌 = 𝑋𝛽 + 𝜀

beta = rnorm(20)
beta
beta[5] = 0
beta[9] = 0
beta[14] = 0
beta[18] = 0
beta[7] = 0
beta[3] = 0
beta

epsilon = rnorm(20)
epsilon

response = dataset%*%beta + epsilon
view(response)

frame_data = data.frame(dataset, response)
view(frame_data)
head(frame_data)
summary(frame_data)
sum(is.na(frame_data))
dim(frame_data)

plot_histogram(frame_data)

split_data = initial_split(frame_data, prop = .80)
extract_train = training(split_data)
extract_test = testing(split_data)
View(extract_train)
View(extract_test)



forward_subset = regsubsets(response~., data = extract_train, nvmax = 21, method = "forward")
summary(forward_subset)

plot(forward_subset, scale = "Cp")
plot(forward_subset, scale = "bic")

coef(forward_subset, 12)

# MSE error for training data
train_matrix = model.matrix(response~., data = extract_train, nvmax = 20)
head(train_matrix)
View(train_matrix)

mse_error = rep(NA, 20)

for (i in 1:20)
{
  coeff = coef(forward_subset, id = i)
  prediction = train_matrix[, names(coeff)] %*% coeff
  mse_error[i] = mean((extract_train$response - prediction)^2)
}

coeff
mse_error
plot(mse_error, xlab = "best model features", ylim = c(0, 22) ,ylab = " MSE error for training data", pch = 20, type = "p")
points(mse_error, cex = .3, col = "orange")

which.min(mse_error)
which.max(mse_error)
mse_error

# MSE ERROR FOR TESTING SET

test_matrix = model.matrix(response~., data = extract_test, nvmax = 20)
View(test_matrix)

mse_test_error = rep(NA, 20)

for (i in 1:20)
{
  coeff1 = coef(forward_subset, id = i)
  prediction_test = test_matrix[, names(coeff1)] %*% coeff1
  mse_test_error[i] = mean((extract_test$response - prediction_test)^2)
}

mse_test_error
coeff1
plot(mse_test_error, xlab = "best model features", ylim = c(0, 22), ylab = "MSE error for testing data", pch= 20, type = "p")
points(mse_test_error, cex = .3, col = "green")

which.min(mse_test_error)
which.max(mse_test_error)




############### Homework problem 3 ###############

################## 3(a) ##############
?iris
names(iris)
dim(iris)
View(iris)
sum(is.na(iris))
summary(iris)

is.factor(iris$Species)

# some visualization needs to do

count_species = table(iris$Species)
count_species

color_spe =c("red", "yellow", "green")
col=color_spe
pie(count_species, labels = count_species, main = "3 kind of species",col=color_spe)
box()

# lets see how many setosa for each sepal length
sepall <- table(iris$Sepal.Length[iris$Species == 'setosa'])
sepall

barplot <- barplot(sepall, main = "number of setosa on sepal length", col = "red", xlab = "sepal length" , ylab = "count")                               
text(x = barplot,  y = sepall + 0.5, labels = sepall)

petalw <- table(iris$Petal.Width[iris$Species == 'setosa'])
petalw

barplots <- barplot(petalw, main = "number of setosa on petal width", col = "red", xlab = "petal width" , ylim = c(0, 40), ylab = "count")                               
text(x = barplots,  y = petalw + 2.5, labels = petalw)


petallen <- table(iris$Petal.Length[iris$Species == 'versicolor'])
petallen

barplots1 <- barplot(petallen, main = "number of versicolor on petal length", col = "red", xlab = "petal length" , ylab = "count")                               
text(x = barplots1,  y = petallen + 0.5, labels = petallen)



#View(norm_data)
split_iris = initial_split(iris, prop = .80)
train_iris = training(split_iris)
test_iris = testing(split_iris)
View(train_iris)
dim(train_iris)
dim(test_iris)

#transformation of the data 
normalize <- function(x) 
{
  return ((x - min(x)) / (max(x) - min(x)))
}

norm_train = as.data.frame(lapply(train_iris[1:4], normalize))
norm_test = as.data.frame(lapply(test_iris[1:4], normalize))

View(norm_train)
head(norm_test)
View(norm_test)


#next run KNN algorithm 


accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

k_vals <- c(1:15)


list_error <- c()
list_accuracy = c()


for (i in 1:15)
{
  
  # select "k"
  kk <- k_vals[i]
  #get prediction on train data 
  train_preds <- knn(train = norm_train, test = norm_test, cl = train_iris$Species, k = kk)
  c_matrix1 <- table(train_preds, test_iris$Species)
  acc1 =accuracy(c_matrix1)
  list_accuracy[i] = acc1
  # error rate
  iris_err <- mean(test_iris$Species != train_preds)
  list_error[i] = iris_err     
}

plot(list_accuracy, xlab = "K value ", ylab = "accuracy Rate", pch= 25, type = "l")
points(list_accuracy, cex = .5, col = "blue")


plot(list_error, xlab = "K value ",  ylab = "Error Rate", pch= 25, type = "l")
points(list_error, cex = .5, col = "red")

matrix_iris = confusionMatrix(data = train_preds, reference = test_iris$Species)
matrix_iris
list_accuracy
list_error

install.packages('e1071', dependencies=TRUE)
library("e1071")


############################## problem 3(b) ##############################
############################ KNN on TWO PCA #############################


pca_iris = iris[c(-5)]
View(pca_iris)

#species_pca_classifier = train_iris[c(5)]
#View(species_pca_classifier)

#computing pca for iris data
compute.pca  <- prcomp(pca_iris, scale = TRUE)
compute.pca
summary(compute.pca)

names(compute.pca)

# getting first two pca for iris data
two_pca = compute.pca$x[,1:2]
two_pca

#turn it into a data frame
two_pca_frame = as.data.frame(two_pca)
View(two_pca_frame)

#the dividing two pca dataset into testing and training

split_iris = initial_split(two_pca_frame, prop = .80)
train_iris_pca = training(split_iris)
test_iris_pca = testing(split_iris)
View(train_iris_pca)
View(test_iris_pca)


# now performs knn algorithm on Two PCA dataset
# 
pca_error = c()
for (i in 1:15)
{
  # select "k"
  kk <- k_vals[i]
  #get prediction and error on test data
  preds_iris <- knn(train = train_iris_pca, test = test_iris_pca, cl = train_iris$Species, k = kk)
  pca_err <- mean(test_iris$Species != preds_iris)
  pca_error[i] = pca_err   
  
}

pca_error
plot(pca_error, xlab = "K value- KNN ", ylab = "error Rate in first two pca", pch= 25, type = "l")
points(pca_error, cex = .5, col = "red")

# reporting confusion matrix
matrix_iris_pca = confusionMatrix(data = preds_iris, reference = test_iris$Species)
matrix_iris_pca


# Plot the scores for the first two principal components and color the samples by class (Species).
compute_train_pca = prcomp(norm_train, scale = FALSE) #already normalized train and test
compute_train_pca
summary(compute_train_pca)

two_pca_compute = compute_train_pca$x[,1:2]
two_pca_compute


pca_predict = predict(compute_train_pca, newdata = norm_test)
pca_predict = as.data.frame(pca_predict)
View(pca_predict)
head(pca_predict)
pca_predict = pca_predict[c(1:2)]
View(pca_predict)


o =ggplot(as.data.frame(two_pca_compute))+ geom_point(aes(x = PC1,y = PC2), col = colors() [c(55,150,300)][as.factor(train_iris$Species)])                             
o+ggtitle("Prediction on pc1 and pc2 for training data")

p = ggplot(as.data.frame(pca_predict))+ geom_point(aes(x = PC1,y = PC2), col = colors() [c(55,150,300)][as.factor(test_iris$Species)])                               
p+ggtitle("Prediction on pc1 and pc2 for test data")

#norm_test = as.data.frame(lapply(test_data[1:85], normalize))

#two_pca = compute.pca$x[,1:2]



