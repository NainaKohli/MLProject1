##MARKETING cAMPAIGN FOR TERM DEPOSIT
rm(list=ls(all=TRUE))
library(ggplot2)    # We'll need to use ggplot to create some graphs.
library(stringr)    # This is used for string manipulations.
install.packages("glmnet")
library(glmnet) # This is where ridge and LASSO reside
install.packages("doParallel")
library(doParallel) # Install parallel processing for R.  This allows multiple processor codes to be used at once.
library(class)
set.seed(45)        # Since we're going to split our data we need to ensure the split is repeatable. 

getwd()
bank<-read.csv("C:/Users/naina/Downloads/bank.csv", stringsAsFactors = FALSE, header = T)
View(bank)
# This code of chunks create extra column for variables with unknown values
bank$job_unk <- ifelse(bank$job == "unknown", 1, 0)
bank$edu_unk <- ifelse(bank$education == "unknown", 1, 0)
bank$cont_unk <- ifelse(bank$contact == "unknown", 1, 0)
bank$pout_unk <- ifelse(bank$poutcome == "unknown", 1, 0)
View(bank)
# This code of chunk make the character data into numeric format
bank$job <- as.numeric(as.factor(bank$job))
bank$marital <- as.numeric(as.factor(bank$marital))
bank$education <- as.numeric(as.factor(bank$education))
bank$default<- ifelse(bank$default == "yes", 1, 0)
bank$housing <- ifelse(bank$housing== "yes", 1, 0)
bank$loan<- ifelse(bank$loan== "yes", 1, 0)
bank$month <- as.numeric(as.factor(bank$month))
bank$contact <- as.numeric(as.factor(bank$contact))
bank$poutcome <- as.numeric(as.factor(bank$poutcome))
bank$y <- ifelse(bank$y== "yes", 1, 0)
View(bank)
# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
# normalize the data to get rid of outliers if present in the data set
bank <- as.data.frame(lapply(bank, normalize))
View(bank)
# Creating design matrix and target vector
mydata.X <- model.matrix(y ~ -1+., data= bank)
View(mydata.X)
mydata.X <- as.data.frame(mydata.X)
mydata.Y <- bank$y

#Now we split the data into training and test.
cuts <- c(training = .8, test = .2)
g <- sample(cut(seq(nrow(mydata.X)), nrow(mydata.X)*cumsum(c(0,cuts)), labels = names(cuts)))
final.X <- split(mydata.X, g)
final.Y <- split(mydata.Y, g)
#we perform regression using LASSO setting alpha to be 1.
bank.lasso <- cv.glmnet(x = as.matrix(final.X$training), y = as.matrix(final.Y$training), nfolds=10, 
                        type.measure="class", parallel=TRUE, family='binomial', alpha = 1, nlambda=100)
print(bank.lasso$lambda.min)
plot(bank.lasso)
##Since best vallue of value of lambda is close to zero. 
##thus from the above plot of misclassification error versus lambda, it is conformed that we do not need to do the regularization.
# Create a dataframe with the coefficient values
lasso.coefs <- as.data.frame(as.vector(coef(bank.lasso, s = bank.lasso$lambda.min)), 
                             row.names = rownames(coef(bank.lasso)))
print(lasso.coefs)
names(lasso.coefs) <- 'coefficient'
features <- rownames(lasso.coefs)[lasso.coefs != 0]
print(features)

# Creates a new matrix with only the non-zero features
lasso_bank <- bank[, intersect(colnames(bank), features)]
# Re-do the split into training and test
bank <- as.matrix(lasso_bank)
bank <- as.data.frame(bank)
bank$Y <- mydata.Y
bank_1 <- split(bank, g)
View(bank_1)
#Now standard logistic regression is run using non zero features identified by a LASSO

model_std <- glm(Y ~ ., family = binomial(link = "logit"),  data = bank_1$training)
summary(model_std)

##Prediction and misclassification of the model
predictions <- predict.glm(model_std, newdata=bank_1$test, type= "response")
p=predictions
predictions[predictions > 0.5] <- 1
predictions[predictions <= 0.5] <- 0

1 - length(predictions[predictions == bank_1$test$Y]) / length(predictions)

##Confusion matrix from the test data
pred.lm=prediction(p,bank_1$test$Y)
#Performance Object
perf.lm=performance(pred.lm,"tpr","fpr")
perf.lm1=performance(pred.lm,"acc","prec")
perf.lm@y.values  # y value shows tpr values
perf.lm@x.values  # x value shows fpr values
#Plot the ROC
plot(perf.lm,xlim=c(0,1),ylim=c(0,1))
plot(perf.lm1,xlim=c(0,1),ylim=c(0,1))
library(caret)
##Confusion matrix from the test data
table(predictions, bank_1$test$Y)
library(gmodels)
CrossTable(predictions, bank_1$test$Y, prop.chisq = FALSE)

