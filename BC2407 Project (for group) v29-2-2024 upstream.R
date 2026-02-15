# Setting Path
setwd("/Users/alexshienhowkhoo/Documents")

# Step 1: Install external package "data.table"
# install.packages("data.table")
library(data.table)
set.seed(100)
readdataset.dt <- fread("predictive_maintenance.csv", check.names = FALSE)
names(readdataset.dt)[names(readdataset.dt) == "Air temperature [K]"] <- "Air_temperature_K"
names(readdataset.dt)[names(readdataset.dt) == "Process temperature [K]"] <- "Process_temperature_K"
names(readdataset.dt)[names(readdataset.dt) == "Rotational speed [rpm]"] <- "Rotational_speed_rpm"
names(readdataset.dt)[names(readdataset.dt) == "Torque [Nm]"] <- "Torque_nm"
names(readdataset.dt)[names(readdataset.dt) == "Tool wear [min]"] <- "Tool_wear_min"

View(readdataset.dt)

# Step 2: The Type of Machines
#install.packages("ggplot2")
library(ggplot2)
frequency_table <- table(readdataset.dt$Type)
frequency_df_Type <- as.data.frame(frequency_table)
frequency_df_Type


ggplot(frequency_df_Type , aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  xlab("Machine Size") +
  ylab("Frequency") +
  ggtitle("Frequency Barplot for Machines") +
  theme(plot.title = element_text(face="bold"))



# Step 3: 
dataset.dt <- readdataset.dt
View(dataset.dt)


# Step 4: Addressing class imbalance to reduce model bias
library(ggplot2)
frequency_table <- table(dataset.dt$`Target`)
frequency_df_target <- as.data.frame(frequency_table)
frequency_df_target

ggplot(frequency_df_target , aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  xlab("Machine Failure") +
  ylab("Frequency") +
  ggtitle("Frequency Barplot for Machine Failure") +
  theme(plot.title = element_text(face="bold"))


# Undersampling 
majority_class <- subset(dataset.dt, dataset.dt$'Target' == 0)
minority_class <- subset(dataset.dt, dataset.dt$'Target' == 1)
nrow(minority_class)
nrow(majority_class)
filtered_indices <- sample(nrow(majority_class), size = nrow(minority_class))
filtered_majority_class <- majority_class[filtered_indices,]
new_dataset.dt <- rbind(minority_class, filtered_majority_class)
dataset.dt <- new_dataset.dt


# Step 5: Checking missing data
View(dataset.dt)
summary(dataset.dt)
sum(is.na(dataset.dt)) 


# Step 6: Checking skewness - resolving skewness
# install.packages("moments")
library(moments)
skewness(dataset.dt$`Air_temperature_K`)
skewness(dataset.dt$`Process_temperature_K`)
skewness(dataset.dt$`Rotational_speed_rpm`)
skewness(dataset.dt$`Torque_nm`)
skewness(dataset.dt$`Tool_wear_min`)


dataset.dt$`Rotational_speed_rpm` <- log(dataset.dt$`Rotational_speed_rpm`)
skewness(dataset.dt$`Rotational_speed_rpm`)



# Step 7: Encode categorical variable
class(dataset.dt$`Failure Type`)
dataset.dt$`Target` <- factor(dataset.dt$`Target`)
dataset.dt <- dataset.dt[, 3:9]
View(dataset.dt)


# One-hot encoding
dataset_transformed <- model.matrix(~ Type - 1, data = dataset.dt)
dataset.dt <- cbind(dataset.dt[, -1], dataset_transformed)
View(dataset.dt)


# Step 8: Min-Max Scale
dataset.dt$Air_temperature_K = (dataset.dt$Air_temperature_K - min(dataset.dt$Air_temperature_K))/(max(dataset.dt$Air_temperature_K)-min(dataset.dt$Air_temperature_K))
dataset.dt$Torque_nm = (dataset.dt$Torque_nm - min(dataset.dt$Torque_nm))/(max(dataset.dt$Torque_nm)-min(dataset.dt$Torque_nm))
dataset.dt$Tool_wear_min = (dataset.dt$Tool_wear_min - min(dataset.dt$Tool_wear_min))/(max(dataset.dt$Tool_wear_min)-min(dataset.dt$Tool_wear_min))
dataset.dt$Process_temperature_K = (dataset.dt$Process_temperature_K - min(dataset.dt$Process_temperature_K))/(max(dataset.dt$Process_temperature_K)-min(dataset.dt$Process_temperature_K))
dataset.dt$Rotational_speed_rpm = (dataset.dt$Rotational_speed_rpm - min(dataset.dt$Rotational_speed_rpm))/(max(dataset.dt$Rotational_speed_rpm)-min(dataset.dt$Rotational_speed_rpm))

View(dataset.dt)

# ---------------------------------------------------------------------------------------------------------------------

# Step 8: Split into training set and test set
library(caTools)
split = sample.split(dataset.dt$Target, SplitRatio =  0.70)
training_set = subset(dataset.dt, split == TRUE)
test_set = subset(dataset.dt, split == FALSE)
View(training_set)
View(test_set)

target_index = 6

df_result <- data.frame(Model = character(), 
                        Accuracy = numeric(), 
                        F1_Score = numeric(),
                        F2_Score = numeric(),
                        Log_Loss = numeric(),
                        stringsAsFactors = FALSE)


# ---------------------------------------------------------------------------------------------------------------------

## Logistic Regression

# Fitting Logistic Regression to the Dataset (Build Logistic Regression Classifier)
classifier = glm(formula = Target ~ ., family = binomial, data = training_set)


# Predicting the test results - probability of new test set
prob_pred = predict(classifier, type = "response", newdata = test_set[,-6]) #Remove the last column
prob_pred
y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred

# Making the confusion matrix
actual_values =  test_set$Target
class(actual_values)
y_pred = as.factor(y_pred)
cm = table(actual_values, y_pred)
cm
accuracy_LR = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])


library(MLmetrics)
f1_LR <- F1_Score(test_set$Target, y_pred, positive = "1") 
f1_LR
f2_LR <- FBeta_Score(test_set$Target, y_pred, positive = "1", beta = 2)
f2_LR
log_loss_LR <- LogLoss(as.numeric(test_set$Target), as.numeric(prob_pred))
log_loss_LR
df_result <- rbind(df_result , data.frame(Model = "Logistic Regression", Accuracy = accuracy_LR, F1_Score = f1_LR, F2_Score = f2_LR, Log_Loss = log_loss_LR))




# -------------------------------------------------------------------------------------------------------

# Support Vector Machine
# Fitting classifier to the Dataset 
#install.packages("e1071")
library(e1071)


classifier = svm(formula = `Target` ~ ., data = training_set, type = "C-classification", kernel = "linear")


# Making the confusion matrix
y_pred = predict(classifier, newdata = test_set[, -6]) 
y_pred
prob_svc = predict(classifier, newdata = test_set, probability = TRUE)
prob_svc
y_pred = as.factor(y_pred)
cm = table(test_set$'Target', y_pred)
cm
accuracy_SVM = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])


library(MLmetrics)
f1_SVM <- F1_Score(test_set$Target, y_pred, positive = "1") 
f1_SVM
f2_SVM <- FBeta_Score(test_set$Target, y_pred, positive = "1", beta = 2)
f2_SVM
log_loss_SVM <- LogLoss(as.numeric(test_set$Target), as.numeric(prob_svc))
log_loss_SVM

df_result  <- rbind(df_result , data.frame(Model = "Support Vector Machine", Accuracy = accuracy_SVM, F1_Score = f1_SVM, F2_Score = f2_SVM, Log_Loss = log_loss_SVM))


# -------------------------------------------------------------------------------------------------------

# Decision Tree Classification

# Step 1: Fitting Decision Tree Classification to the Dataset 
# install.packages("rpart")
library(rpart)
classifier <- rpart(formula = Target ~ . , 
                    data = training_set, method = 'class',
                    control = rpart.control(minsplit = 2, cp = 0))


# Step 2: Extract the optimal tree
# Compute min CVerror + 1SE in maximal tree

classifier$cptable
?which.min()
CVerror.xerror <- classifier$cptable[which.min(classifier$cptable[,"xerror"]), "xerror"]
CVerror.xerror

CVerror.std <- classifier$cptable[which.min(classifier$cptable[,"xerror"]), "xstd"]
CVerror.std

CVerror.cap <- classifier$cptable[which.min(classifier$cptable[,"xerror"]), "xerror"] + classifier$cptable[which.min(classifier$cptable[,"xerror"]), "xstd"]
CVerror.cap


# Step 3: Find the most optimal CP region whose CV error is just below CVerror.cap in maximal tree
i <- 1
j <- 4

while(classifier$cptable[i,j] > CVerror.cap){
  i <- i + 1
}

# Step 4: Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split
cp.opt <-  ifelse(i > 1, sqrt(classifier$cptable[i,1] * classifier$cptable[i-1,1]),1)
cp.opt

# Step 5: Get the optimal tree 
classifier2 <- prune(classifier, cp = cp.opt)
printcp(classifier2, digits = 3)
?printcp()


# Step 6: Plot the CART model and corresponding variable importance bar chart
library("rpart.plot")
rpart.plot(classifier2, nn = T, main = "Optimal CART Model.csv") 

# Step 7: Print the summary of CART model and the CART model
print(classifier2)
summary(classifier2)


# Step 8: Print the variable importance bar chart
var_importance <- classifier2$variable.importance
var_importance
sorted_var_importance <- var_importance[order(var_importance, decreasing = TRUE)]
sorted_var_importance
rownames <- colnames(classifier2$variable.importance)
rownames

barplot(sorted_var_importance, 
        names.arg = names(classifier2$variable.importance),
        xlab = "Variable Importance",
        #ylab = "Variable",
        col = "darkblue",  # Change the color as needed
        horiz = TRUE,
        las = 2,
        main = "Variable Importance Bar Chart (Optimal CART Model)") 
par(mar = c(5.1,15,4.1,2.1)) # bottom, left, top right


# Step 9: Checking prediction accuracy by making the confusion matrix table
classifier2.predict <- predict(classifier2, newdata = test_set[,-6], type = "class")
classifier2.predict


results <- data.frame(test_set, classifier2.predict)
results

cm = table(results[,6], results[,10])
cm 
accuracy_CART = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])



# Step 10: Check the f1 score of the predicted results
library(MLmetrics)
f1_CART <- F1_Score(test_set$Target, classifier2.predict,  positive = "1")
f1_CART

f2_CART <- FBeta_Score(test_set$Target, classifier2.predict,  positive = "1", beta = 2)
f2_CART


# Step 11: Predict the class probability 
classifier2.predictprob <- predict(classifier2, newdata = test_set[,-6], type = "prob")
classifier2.predictprob

log_loss_CART <- LogLoss(as.numeric(test_set$Target), as.numeric(classifier2.predictprob[,2]))
log_loss_CART

df_result <- rbind(df_result, data.frame(Model = "CART", Accuracy = accuracy_CART, F1_Score = f1_CART, F2_Score = f2_CART, Log_Loss = log_loss_CART))


# -----------------------------------------

# Random Forest Classification

#install.packages("randomForest")
library(randomForest)
?randomForest


View(dataset.dt)
classifier3 = randomForest(x = training_set[,-6], y = training_set$Target, ntree = 30, keep.forest = TRUE) #Note ntree large will cause overfitting

# Step 1: Predict the test results - probability of new test set
y_pred = predict(classifier3, newdata = test_set[,-6]) 
y_pred

# Step 2: Predict the class probability
y_pred_prob = predict(classifier3, newdata = test_set[,-6], type = "prob") 
y_pred_prob

# Step 3: Plot confusion matrix
cm = table(test_set$Target, y_pred)
cm 
accuracy_RF = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])


# Step 4: Plot a variable importance plot
varImpPlot(classifier3, main = "Variable Importance Plot of Random Forest", pch = 16, col = "black", las = 1)



library(MLmetrics)
f1_RF <- F1_Score(test_set$Target, y_pred, positive = "1") 
f1_RF 
f2_RF <- FBeta_Score(test_set$Target, y_pred, positive = "1", beta = 2)
f2_RF 
log_loss_RF <- LogLoss(as.numeric(test_set$Target), as.numeric(y_pred_prob[,2]))
log_loss_RF


df_result <- rbind(df_result, data.frame(Model = "Random Forest", Accuracy = accuracy_RF, F1_Score = f1_RF, F2_Score = f2_RF, Log_Loss = log_loss_RF))


# --------------------
# 10-Fold Random Forest

library(caret)
folds = createFolds(dataset.dt$Target, k = 10)
cv = lapply(folds, function(x) {
  training_fold = dataset.dt[-x, ]
  test_fold = dataset.dt[x, ]
  classifier3 = randomForest(x = training_set[,-6], y = training_set$Target, ntree = 30, keep.forest = TRUE)
  classifier3.predict = predict(classifier3, newdata = test_set[,-6]) 
  cm = table(test_set$Target, classifier3.predict)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])    
  return(accuracy)
})
accuracy_10RF = mean(as.numeric(cv))
accuracy_10RF

print(df_result)

df_result <- rbind(df_result, data.frame(Model = "10-Fold Random Forest", Accuracy = accuracy_10RF, F1_Score = f1_RF, F2_Score = f2_RF, Log_Loss = log_loss_RF))
print(df_result)


# ------------------------
# Ridge and Lasso Regression

#install.packages("glmnet")
library(glmnet)

?glmnet

classifier <- cv.glmnet(as.matrix(training_set[,-6]),  training_set$Target, alpha = 0,  family = "binomial")
lambda_best <- classifier$lambda.min
y_pred_prob = predict(classifier, newx = as.matrix(test_set[,-6]), s = lambda_best, type = "response")
y_pred_prob
threshold <- 0.5
y_pred <- ifelse(y_pred_prob > threshold, 1, 0)
cm = table(test_set$Target, y_pred)
cm
accuracy_RR = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])

library(MLmetrics)
f1_RR <- F1_Score(test_set$Target, y_pred, positive = "1") 
f1_RR 
f2_RR <- FBeta_Score(test_set$Target, y_pred, positive = "1", beta = 2)
f2_RR 
log_loss_RR <- LogLoss(as.numeric(test_set$Target), as.numeric(y_pred_prob))
log_loss_RR

df_result <- rbind(df_result, data.frame(Model = "Ridge Regression", Accuracy = accuracy_RR, F1_Score = f1_RR, F2_Score = f2_RR, Log_Loss = log_loss_RR))



classifier <- cv.glmnet(as.matrix(training_set[,-6]),  training_set$Target, alpha = 1,  family = "binomial")
lambda_best <- classifier$lambda.min
y_pred = predict(classifier, newx = as.matrix(test_set[,-6]), s = lambda_best, type = "response")
y_pred
threshold <- 0.5
y_pred <- ifelse(y_pred > threshold, 1, 0)
cm = table(test_set$Target, y_pred)
cm
accuracy_LSR = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])

library(MLmetrics)
f1_LSR <- F1_Score(test_set$Target, y_pred, positive = "1") 
f1_LSR
f2_LSR <- FBeta_Score(test_set$Target, y_pred, positive = "1", beta = 2)
f2_LSR 
log_loss_LSR <- LogLoss(as.numeric(test_set$Target), as.numeric(y_pred_prob))
log_loss_LSR

df_result <- rbind(df_result, data.frame(Model = "Lasso Regression", Accuracy = accuracy_LSR, F1_Score = f1_LSR, F2_Score = f2_LSR, Log_Loss = log_loss_LSR))

print(df_result)




# ---------------------------------------------------------------------

# XG-Boost
#install.packages('xgboost')
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[,-6]), label = as.numeric(as.factor(training_set$Target)) - 1 , nrounds = 10, objective = "binary:logistic", eval_metric = "error") 
y_pred_prob = predict(classifier, newdata = as.matrix(test_set[,-6]))
y_pred = (y_pred_prob >= 0.5)
y_pred = ifelse(y_pred == TRUE,1,0)
cm = table(test_set$Target, y_pred)
accuracy_XG = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])    



library(MLmetrics)
f1_XG <- F1_Score(test_set$Target, y_pred, positive = "1") 
f1_XG 
f2_XG <- FBeta_Score(test_set$Target, y_pred, positive = "1", beta = 2)
f2_XG 
log_loss_XG <- LogLoss(as.numeric(test_set$Target), as.numeric(y_pred_prob))
log_loss_XG

df_result <- rbind(df_result, data.frame(Model = "XG Boost", Accuracy = accuracy_XG, F1_Score = f1_XG, F2_Score = f2_XG, Log_Loss = log_loss_XG))


# ------------------------------------------------------------------

# K-Fold Cross Validation - XG Boost
#install.packages('caret')

library(caret)
folds = createFolds(dataset.dt$Target, k = 10)
cv = lapply(folds, function(x) {
  training_fold = dataset.dt[-x, ]
  test_fold = dataset.dt[x, ]
  classifier = xgboost(data = as.matrix(training_set[,-6]), label = as.numeric(as.factor(training_set$Target)) - 1 , nrounds = 10, objective = "binary:hinge", eval_metric = "error") 
  y_pred = predict(classifier, newdata = as.matrix(test_fold[,-6]))
  y_pred = (y_pred >= 0.5)
  y_pred = ifelse(y_pred == TRUE,1,0)
  cm = table(test_fold$Target, y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])    
  return(accuracy)
})
accuracy_10XG = mean(as.numeric(cv))
accuracy_10XG

print(df_result)

df_result <- rbind(df_result, data.frame(Model = "10-Fold XG Boost", Accuracy = accuracy_10XG, F1_Score = f1_XG, F2_Score = f2_XG, Log_Loss = log_loss_XG))


----------------------
# Neural Network
  
#install.packages("neuralnet")
library(neuralnet)
View(dataset.dt)



new.dt <- dataset.dt

# Train-test split
set.seed(100)
library(caTools)
split = sample.split(new.dt$Target, SplitRatio =  0.70)
training_set = subset(new.dt, split == TRUE)
test_set = subset(new.dt, split == FALSE)
View(training_set)
View(test_set)


training_set$Target <- as.numeric(training_set$Target)-1
test_set$Target <- as.numeric(test_set$Target)-1

# Modelling
neuralnetwork <- neuralnet(Target~., data=training_set, hidden=c(6,6,4,2), err.fct="ce", linear.output=FALSE, stepmax=1e6)


# Prediction
y_pred_prob <- predict(neuralnetwork, test_set)
y_pred_prob
y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)
y_pred
cm = table(test_set$Target, y_pred)
cm
accuracy_nn = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])   
accuracy_nn
par(mfrow=c(1,1))
plot(neuralnetwork)

summary(dataset.dt)

par(mfrow=c(3,3))
gwplot(neuralnetwork,selected.covariate="Air_temperature_K", min=-2.5, max=5)
gwplot(neuralnetwork,selected.covariate="Process_temperature_K", min=-2.5, max=5)
gwplot(neuralnetwork,selected.covariate="Rotational_speed_rpm", min=-2.5, max=5)
gwplot(neuralnetwork,selected.covariate="Torque_nm", min=-2.5, max=5)
gwplot(neuralnetwork,selected.covariate="Tool_wear_min", min=-2.5, max=5)
gwplot(neuralnetwork,selected.covariate="TypeH", min=-2.5, max=5)
gwplot(neuralnetwork,selected.covariate="TypeL", min=-2.5, max=5)
gwplot(neuralnetwork,selected.covariate="TypeM", min=-2.5, max=5)


library(MLmetrics)
f1_nn <- F1_Score(test_set$Target, y_pred, positive = "1") 
f1_nn 
f2_nn <- FBeta_Score(test_set$Target, y_pred, positive = "1", beta = 2)
f2_nn 
log_loss_nn <- LogLoss(as.numeric(test_set$Target), as.numeric(y_pred_prob))
log_loss_nn

df_result <- rbind(df_result, data.frame(Model = "Neural Network", Accuracy = accuracy_nn, F1_Score = f1_nn, F2_Score = f2_nn, Log_Loss = log_loss_nn))
print(df_result)



