############################ SVM Digit Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation
#####################################################################################

#####################################################################################

# 1. Business Understanding: 

# The objective is to develop a model using Support Vector Machine which should correctly 
# classify the handwritten digits based on the pixel values given as features

#####################################################################################

#Loading Neccessary libraries

#install.packages("caret")

#install.packages("kernlab")

#install.packages("dplyr")

#install.packages("readr")

#install.packages("ggplot2")

#install.packages("gridExtra")

library(caret)
library(kernlab)
library(ggplot2)
library(dplyr)
library(readr)
library(gridExtra)

#Loading Data

Digrec_train <- read.csv(file.choose(), stringsAsFactors = F, header =FALSE)
Digrec_test <- read.csv(file.choose(), stringsAsFactors = F, header = FALSE)

#Understanding Dimensions

dim(Digrec_train) # 59999 obs. 785 variables
dim(Digrec_test)  # 9999 obs 785 variables 

#Structure of the dataset

str(Digrec_train)

#printing first few rows

head(Digrec_train)

#Exploring the data

summary(Digrec_train)

#checking missing value

sapply(Digrec_train, function(x) sum(is.na(x))) # no, missing values

# checking duplicate values

sum(duplicated(Digrec_train))
sum(duplicated(Digrec_test))

# First column of train data set containing the "digit" that is represented by the other 784 columns
names(Digrec_train)[1] <- "digit"
names(Digrec_test)[1] <- "digit"

#Making our target class to factor

Digrec_train$digit <- as.factor(Digrec_train$digit)
Digrec_test$digit <- as.factor(Digrec_test$digit)

levels(Digrec_train[, 1])

#All the other columns are numeric:
sapply(Digrec_train[1,], class)

#----------------------------------------- Exploratory Data Analysis --------------------------------------#

# Create a 28*28 matrix with pixel color values
m = matrix(unlist(Digrec_train[10,-1]),nrow = 28,byrow = T)
# Plot that matrix
image(m,col=grey.colors(255))
rotate <- function(x) t(apply(x, 2, rev))
# Plot a bunch of images
par(mfrow=c(2,3))
lapply(1:6, 
       function(x) image(
         rotate(matrix(unlist(Digrec_train[x,-1]),nrow = 28,byrow = T)),
         col=grey.colors(255),
         xlab=Digrec_train[x,1]
       )
)
par(mfrow=c(1,1))



# Using 15% of train and test data set for model building and evaluation 

digirec_train_samp <- Digrec_train[sample(nrow(Digrec_train), 9000), ]
digirec_test_samp <- Digrec_test[sample(nrow(Digrec_test),1500), ]

#Constructing Model

#Using Linear Kernel
Model_linear <- ksvm(digit~ ., data = digirec_train_samp, scale = FALSE, kernel = "vanilladot", C = 1)
Eval_linear<- predict(Model_linear, digirec_test_samp, type = "response")

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,digirec_test_samp$digit)

# Accuracy : 92%
# Sensitivity : 85%
# specificity : 99%

#Using RBF Kernel
Model_RBF <- ksvm(digit~ ., data = digirec_train_samp, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, digirec_test_samp)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,digirec_test_samp$digit)

# Accuracy : 96.53
# sensitivity : 94.9
# Specificity : 99

# Acuracy has increased. It shows data have non-linearity. 

############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 2 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
grid <- expand.grid(.sigma=c(0.025, 0.05), .C=c(0.1,0.5,1,2) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(digit~., data=digirec_train_samp, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)
print(fit.svm)
plot(fit.svm)

# Observation
# Accuracy : 11%

grid <- expand.grid(.sigma=c(0.001), .C=c(1,2,3,4,5,6,7,8,9,10) )

fit.svm2 <- train(digit~., data=digirec_train_samp, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm2)

# Observation:
# Accuracy didn't improve it's still 11%
# Will reduce sigma more to check non-collinearity 

grid <- expand.grid(.sigma=c(0.0000001), .C=c(1,2,3,4,5,6,7,8,9,10) )
fit.svm3 <- train(digit~., data=digirec_train_samp, method="svmRadial", metric=metric, 
                  tuneGrid=grid, trControl=trainControl)
print(fit.svm3)
plot(fit.svm3)

# Observation
# Accurracy is high with sigma = 0.0000001 and C = 3

grid <- expand.grid(.sigma=c(0.00000001), .C=c(1,2,3,4,5,6,7,8,9,10) )
fit.svm4 <- train(digit~., data=digirec_train_samp, method="svmRadial", metric=metric, 
                  tuneGrid=grid, trControl=trainControl)
print(fit.svm4)

# Observation 
# Accuracy again decreased. 

# Final model

# fit.svm3 

# Model verification through example, we need to increase row count to see current and predicted value

row <- 8
prediction.digit <- as.vector(predict(Model_RBF, newdata = digirec_test_samp[row, 
                                                                        ], type = "response"))
print(paste0("Current Digit: ", as.character(digirec_test_samp$digit[row])))
print(paste0("Predicted Digit: ", prediction.digit))
z <- array(as.vector(as.matrix(digirec_test_samp[row, -1])), dim = c(28, 28))
z <- z[, 28:1]  ##right side up
par(mfrow = c(1, 3), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
image(1:28, 1:28, z, main = digirec_test_samp[row, 1], col = grey.colors(255))
