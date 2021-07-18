library(readr)
library(caret)
library(ggplot2)
library(MLmetrics)
library(ROCR)
library(pROC)
library(dplyr)
bank <- read.csv("B:/GLOBAL/2063-BASUS/FLORHAM-PARK/NTH/NTH-T/TalentManagement/03_Data & Analytics (Quinn & Jake)/Quinn/Syracuse/Marketing Analytics/Final Project/bank.csv", stringsAsFactors=TRUE)
summary(bank)
#bank <- read.csv("B:/GLOBAL/2063-BASUS/FLORHAM-PARK/NTH/NTH-T/TalentManagement/03_Data & Analytics (Quinn & Jake)/Quinn/Syracuse/Marketing Analytics/Final Project/bank_2.csv", stringsAsFactors=TRUE)

round(prop.table(table(bank$deposit)),2) #Take a peak at our DV balance 

#Age Grouping
bank$AgeGroup <- with(bank,ifelse(age>55,8,ifelse(age>50,7,ifelse(age>45,6,ifelse(age>40,5,
                                                                              ifelse(age>35,4,ifelse(age>30,3,ifelse(age>25,2,1))))))))
bank$AgeGroup <- as.ordered(bank$AgeGroup)
#Married Grouping
bank$MarriedGroup <- with(bank,ifelse(marital=="married",1,0)) 
bank$MarriedGroup <- as.factor(bank$MarriedGroup)

#Season Grouping
bank <- bank %>%
  mutate(
    season = case_when(
      month %in% c("oct","nov","dec") ~ "Fall",
      month %in% c("jan","feb","mar")  ~ "Winter",
      month %in%  c("apr","may","jun")  ~ "Spring",
      TRUE ~ "Summer"))
bank$season <- as.factor(bank$season)

#Time of the Month Grouping 
bank$TimeoftheMonth <- with(bank,ifelse(day>14,1,0))


#Log of balance
bank$balance_log <- log(bank$balance+6848)

#Log of Duration 
bank$duration_log <- log(bank$duration)

#Log of campaign
bank$campaign_log <- log(bank$campaign)

#Log of previous 
bank$previous_log <- log(bank$previous)

#Log of pdays
bank$pdays_log <- log(bank$pdays+2)

hist(as.numeric(bank$duration),col="orange")
hist(as.numeric(bank$duration_log), col="orange")


bank_cluster <- bank[c(6,12:15)]
bank_scale <- scale(bank_cluster)

#Distance Function and Visualization 
library(factoextra)



#Clustering
set.seed(123)
k2 <- kmeans(bank_scale, centers = 2, nstart = 25, iter.max = 100, algorithm = "Hartigan-Wong")
fviz_cluster(k2, data = bank_scale)


k3 <- kmeans(bank_scale, centers = 3, nstart = 25)
k4 <- kmeans(bank_scale, centers = 4, nstart = 25)
k5 <- kmeans(bank_scale, centers = 5, nstart = 25)

# plots to compare
p1 <- fviz_cluster(k2, geom = "point", data = bank_scale) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = bank_scale) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = bank_scale) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = bank_scale) + ggtitle("k = 5")

library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)






#Elbow Chart
wss <- function(k){
  return(kmeans(bank_scale, k, nstart = 25)$tot.withinss)
}

k_values <- 1:15

wss_values <- purrr::map_dbl(k_values, wss)

plot(x = k_values, y = wss_values, 
     type = "b", pch=19, frame = F,
     xlab = "Number of Clusters K",
     ylab = "Total within-clusters sum of square")

set.seed(123)

fviz_nbclust(bank_scale, kmeans, method = "wss")

fviz_nbclust(bank_scale, kmeans, method = "silhouette")


# compute gap statistic
set.seed(123)
library(cluster)
gap_stat <- clusGap(bank_scale, FUN = kmeans, nstart = 25,
                    K.max = 10, B = 50)
# Print the result
print(gap_stat, method = "firstmax")

fviz_gap_stat(gap_stat)

# Compute k-means clustering with k = 3
set.seed(123)
final <- kmeans(bank_scale, 3, nstart = 25,iter.max = 100, algorithm = "Hartigan-Wong")
print(final)
fviz_cluster(final, data = bank_scale)

k6 <- kmeans(bank_scale, centers = 3, nstart = 25, iter.max = 100, algorithm = "Hartigan-Wong")
k7 <- kmeans(bank_scale, centers = 3, nstart = 25, iter.max = 100, algorithm = "Lloyd")
k8 <- kmeans(bank_scale, centers = 3, nstart = 25, iter.max = 100, algorithm = "Forgy")
k9 <- kmeans(bank_scale, centers = 3, nstart = 25, iter.max = 100, algorithm = "MacQueen")

# plots to compare
p5 <- fviz_cluster(k6, geom = "point", data = bank_scale) + ggtitle("Hartigan-Wong")
p6 <- fviz_cluster(k7, geom = "point",  data = bank_scale) + ggtitle("Lloyd")
p7 <- fviz_cluster(k8, geom = "point",  data = bank_scale) + ggtitle("Forgy")
p8 <- fviz_cluster(k9, geom = "point",  data = bank_scale) + ggtitle("MacQueen")

grid.arrange(p5, p6, p7, p8, nrow = 2)

cluster <- kmeans(bank_scale, centers = 2, nstart = 25, iter.max = 100, algorithm = "Hartigan-Wong")
bank$cluster <- cluster$cluster 

#write.csv(bank, file="/Users/KnudseQ/Desktop/Final Project Cluster Summary.csv", row.names=FALSE)



# Partioning the data train 80% test 20%
set.seed(123)
ind <- sample(2, nrow(bank), replace =T, prob = c(0.8, 0.2))
train <- bank[ind==1,]
test <- bank[ind==2,]


tune <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE,
                     summaryFunction = multiClassSummary) 
rf_grid <- expand.grid(mtry = seq(from = 5, to = 15, by = 5))





model1_glm <- train(deposit~.,
                   data = train,
                   method = "glm",
                   family = "binomial",
                   #preProcess = "pca",
                   trControl = tune)
summary(model1_glm)
print(model1_glm)

glm.prd <- predict(model1_glm, test)
confusionMatrix(glm.prd, as.factor(test$deposit))
#Plot Logistic
log.plot <- plot.roc (as.numeric(test$deposit),
                      as.numeric(glm.prd),lwd=2, type="b", print.auc=TRUE, col ="blue")

model2_rf <- train(deposit~.,
                  data = train,
                  method = "rf",
                  #method = "rpart",  
                  metric = "ROC",
                  ntree = 25,
                  tuneLength = 5,
                  trControl = tune,
                  #control = rpart.control(minsplit=10,maxdepth=30),
                  tuneGrid = rf_grid)

print(model2_rf)

# Prediction & Confusion Matrix - test data 
rf.prd <- predict(model2_rf, test)
test$deposit <- as.factor(test$deposit)
confusionMatrix(rf.prd, as.factor(test$deposit))

#plot ROC
rf.plot<- plot.roc(as.numeric(test$deposit), as.numeric(rf.prd ),lwd=2, type="b",print.auc=TRUE,col ="blue")

# Simple GBM
library(gbm)
ctrl <- trainControl(method = "cv", number=10,savePred=T, classProb=T)

gbmfit <- train(deposit ~., 
                data = train, 
                method = "gbm", 
                verbose = FALSE, 
                metric = "ROC", 
                trControl = ctrl)
print(gbmfit)
GB.prd <- predict(gbmfit,test)
confusionMatrix(GB.prd, test$deposit)
GB.plot <- plot.roc (as.numeric(test$deposit),
                      as.numeric(XGB.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")


xgbGrid <- expand.grid(nrounds = 100,
                       
                       max_depth = 12,
                       eta = .03,
                       gamma = 0.1,
                       colsample_bytree = .7,
                       min_child_weight = .08,
                       subsample = 1
)
gb_ctrl <- trainControl(method = "cv",
                        number = 10,
                        summaryFunction = multiClassSummary,
                        classProbs = TRUE
) 

XGB.model <- train(deposit~., data = train,
                   method = "xgbTree"
                   ,trControl = gb_ctrl
                   , verbose=0
                   , maximize=FALSE
                   ,tuneGrid = xgbGrid
)


XGB.prd <- predict(XGB.model,test)
confusionMatrix(XGB.prd, test$deposit)
XGB.plot <- plot.roc (as.numeric(test$deposit),
                      as.numeric(XGB.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")
ggplot(varImp(XGB.model)) + 
  geom_bar(stat = 'identity', fill = 'steelblue', color = 'black') + 
  scale_y_continuous(limits = c(0, 105), expand = c(0, 0)) +
  theme_light()




#Support Vector Machine
library("e1071")
set.seed(123)
svm<-svm(
  as.factor(train$deposit)~.,
  type="C-classification",
  data=train
)

svm.prd <- predict(svm,newdata=test)
confusionMatrix(svm.prd,test$deposit)
svm.plot <-plot.roc (as.numeric(test$deposit), 
                     as.numeric(svm.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")


svm2 <- train(as.factor(deposit)~., data=train, method = "svmLinear", trControl = ctrl)
svm.prd2 <- predict(svm2,newdata=test)
confusionMatrix(svm.prd2,test$deposit)
svm.plot2 <-plot.roc (as.numeric(test$deposit), 
                     as.numeric(svm.prd2),lwd=2, type="b", print.auc=TRUE,col ="blue")


# Custom Control Parameters
custom <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 5,
                       verboseIter =  T)

# Ridge Regression
set.seed(123)
ridge <- train(deposit ~.,
               train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 0,
                                      lambda = seq(0.001, 1, length=5)),
               trControl = tune)
plot(ridge)
plot(ridge$finalModel, xvar = 'lambda', label = T)
plot(ridge$finalModel, xvar = 'dev', label = T)
plot(varImp(ridge, scale=T))
#For ROC
ridge.prd <- predict(ridge,newdata=test)
confusionMatrix(ridge.prd,test$deposit)
test$deposit <- as.factor(test$deposit)
ridge.plot <-plot.roc (as.numeric(test$deposit), 
                       as.numeric(ridge.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")
# Lasso Regression 
set.seed(123)
lasso <- train(deposit ~.,
               train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 1,
                                      lambda = seq(0.001, 0.2, length=5)),
               trControl = custom)
# Plot Results
lasso
plot(lasso)
plot(lasso$finalModel, xvar = 'lambda', label = T)
plot(lasso$finalModel, xvar = 'dev', label = T)
#For ROC
lasso.prd <- predict(lasso,newdata=test)
confusionMatrix(lasso.prd,test$deposit)
test$deposit <- as.factor(test$deposit)
lasso.plot <-plot.roc (as.numeric(test$deposit), 
                       as.numeric(lasso.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")
#Elastic Net Regression 
set.seed(123)
en <- train(deposit ~.,
            train,
            method = 'glmnet',
            tuneGrid = expand.grid(alpha = seq(0,1, length=10),
                                   lambda = seq(0.001, 0.2, length=5)),
            trControl = tune)
plot(en)
plot(en$finalModel, xvar = 'lambda', label = T)
plot(en$finalModel, xvar = 'dev', label = T)
plot(varImp(en, scale=T))

en.prd <- predict(en,newdata=test)
confusionMatrix(en.prd,test$deposit)
test$deposit <- as.factor(test$deposit)
en.plot <-plot.roc (as.numeric(test$deposit), 
                    as.numeric(en.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")


#Compare Models
model_list <- list(Ridge = ridge, Lasso = lasso, ElasticNet = en)
res <- resamples(model_list)
summary(res)
bwplot(res)
xyplot(res, 'RMSE')


#Ensemble Model
library(caretEnsemble)
library(iml)
library(caret)
library(randomForest)
set.seed(123)

control_stacking <- trainControl(method="cv", number=10, repeats=2, savePredictions=TRUE, classProbs=TRUE)

algorithms_to_use <- c('rpart', 'glm', 'knn', 'svmRadial','xgbTree','nnet')

stacked_models <- caretList(deposit ~., data=train, trControl=control_stacking, methodList=algorithms_to_use)

stacking_results <- resamples(stacked_models)

summary(stacking_results)

# stack using glm
stackControl <- trainControl(method="cv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)

set.seed(123)

glm_stack <- caretStack(stacked_models, method="glm", metric="Accuracy", trControl=stackControl)

print(glm_stack)
stack.prd <- predict(glm_stack,newdata=test)
confusionMatrix(stack.prd,test$deposit)
test$deposit <- as.factor(test$deposit)
stack.plot <-plot.roc (as.numeric(test$deposit), 
                       as.numeric(stack.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")


glm_stack2 <- caretStack(stacked_models, method="rf", metric="Accuracy", trControl=stackControl)
stack.prd2 <- predict(glm_stack2,newdata=test)
confusionMatrix(stack.prd2,test$deposit)
test$deposit <- as.factor(test$deposit)
stack.plot2 <-plot.roc (as.numeric(test$deposit), 
                       as.numeric(stack.prd2),lwd=2, type="b", print.auc=TRUE,col ="blue")



# Plot the ROC curves
pred <- prediction(preds_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
legend(x = "bottomright", 
       legend = c("Logistic Regression", "Neural Network", "Random Forest", "GBM", "Elastic Net"),
       fill = 1:m)

plot(rocrf, ylim = c(0,1), print.thres = T, print.thres.cex = 0.8, col = "darkolivegreen", add = T)
plot(rocgbm, ylim = c(0,1), print.thres = T, print.thres.cex = 0.8, col = "burlywood", add = T)


#plotting the ROC curves (better view) 
par(mfrow=c(5,2))
plot.roc (as.numeric(test$deposit), as.numeric(GB.prd),main="GBoost",lwd=2, type="b", print.auc=TRUE, col ="blue")
plot.roc (as.numeric(test$deposit), as.numeric(XGB.prd), main="eXtreme Gradient Boost",lwd=2, type="b", print.auc=TRUE, col ="brown")
plot.roc (as.numeric(test$deposit), as.numeric(glm.prd),main="Logistic",lwd=2, type="b", print.auc=TRUE, col ="green")
plot.roc (as.numeric(test$deposit), as.numeric(svm.prd),main="SVM",lwd=2, type="b", print.auc=TRUE, col ="red")
plot.roc (as.numeric(test$deposit), as.numeric(stack.prd), main="Ensemble 1",lwd=2, type="b", print.auc=TRUE, col ="seagreen4")
plot.roc (as.numeric(test$deposit), as.numeric(rf.prd), main="Random Forest",lwd=2, type="b", print.auc=TRUE, col ="slateblue4")
plot.roc (as.numeric(test$deposit), as.numeric(stack.prd2), main="Ensemble 2",lwd=2, type="b", print.auc=TRUE, col ="orange")
plot.roc (as.numeric(test$deposit), as.numeric(ridge.prd), main="Ridge",lwd=2, type="b", print.auc=TRUE, col ="purple")
plot.roc (as.numeric(test$deposit), as.numeric(lasso.prd), main="Lasso",lwd=2, type="b", print.auc=TRUE, col ="goldenrod4")
plot.roc (as.numeric(test$deposit), as.numeric(en.prd), main="Elastic Net",lwd=2, type="b", print.auc=TRUE, col ="gray")




model_list <- list(glm = model1_glm,
                   rf = model2_rf,
                   xgb = XGB.model,
                   en = en)

res <- resamples(model_list)

summary(res)
bwplot(res , metric = c("Accuracy", "AUC", "F1"))
compare_models(model2_rf, XGB.model)



