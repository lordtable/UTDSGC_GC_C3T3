# Load CompleteResponses and SurveyIncomplete data
library(readr)
df_exsprod<-read.csv('existingproductattributes2017.csv')
df_newprod<-read.csv('newproductattributes2017.csv')

# EDA of the raw data

library(summarytools)
view(dfSummary(df_exsprod),
     file="df_exsprod_RAW.html")

view(dfSummary(df_newprod),
     file="df_newprod_RAW.html")

library(explore)
explore(df_exsprod)

library(dlookr)
#plot_normality(Data_Complete)
#plot_correlate(Data_Complete)

eda_report(df_exsprod,target=Volume,output_format="html",
           output_file="EDA_dlookr_report.html")


# labeling categorical data
library(caret)
#dummy_var<-dummyVars("~.",data=df_exsprod)
#df_exsprod_ed<-data.frame(predict(dummy_var,
                                  #newdata=df_exsprod))

# try to do something different here, a more
# compact coding ORDINAL

df_exsprod_ed2=df_exsprod_ed
df_exsprod_ed2$ProductType<-as.numeric(factor(df_exsprod_ed$ProductType))

# create a dictionary of ProductType mapping
df_ProductType_dict<-data.frame(df_exsprod_ed$ProductType,
                                df_exsprod_ed2$ProductType)
df_ProductType_dict<-distinct(df_ProductType_dict)

names(df_ProductType_dict)<-c("ProductType","ID")

df_ProductType_dict<-arrange(df_ProductType_dict,ID)

#apply this dictionary to df_newprod_ed

#add ID column from the dictionary to newprod

#based on common column:ProductType
df_newprod_ed2<-right_join(df_newprod_ed,
                df_ProductType_dict,by="ProductType")

#replace ProductType with their corresponding
#value based on dictionary
df_newprod_ed2$ProductType<-df_newprod_ed2$ID

#remove ID column, no longer needed
df_newprod_ed2<-subset(df_newprod_ed2,select=-c(ID))

# CORRELATION

corrData<-cor(df_exsprod_ed2)
library(corrplot)
corrplot(corrData)

library(dlookr)
#plot_correlate(df_exsprod_ed2)
eda_report(df_exsprod_ed2,target=Volume,output_format="html",
           output_file="EDA_dlookr_report.html")

# FEATURE SELECTION: BASED ON CORR HEATMAP

#remove highly correlated variables and 
# other useless from file
df_exsprod_ed3<-subset(df_exsprod_ed2,
                       select=-c(ProductNum,x5StarReviews,x4StarReviews,
                                 x3StarReviews,x2StarReviews,
                                 x1StarReviews,BestSellersRank,
                                 ShippingWeight,ProductDepth,ProductWidth,
                                 ProductHeight,ProfitMargin))

df_newprod_ed3<-subset(df_newprod_ed2,
                       select=-c(ProductNum,x5StarReviews,x4StarReviews,
                                 x3StarReviews,x2StarReviews,
                                 x1StarReviews,BestSellersRank,
                                 ShippingWeight,ProductDepth,ProductWidth,
                                 ProductHeight,ProfitMargin))


# remove missing and NA
#according to EDA, BestSellersRank has 15 NAs
# repeat on df_newprod
df_exsprod_ed4<-na.omit(df_exsprod_ed3)
df_newprod_ed4<-na.omit(df_newprod_ed3)

# remove duplicated rows
library(dplyr)
df_exsprod_ed5<-distinct(df_exsprod_ed4)
df_newprod_ed5<-distinct(df_newprod_ed4)

corrData_ed5<-cor(df_exsprod_ed5)
corrplot(corrData_ed5)

library(dlookr)
#plot_normality(Data_Complete)
plot_correlate(df_exsprod_ed5)


# DATA SPLITTING

set.seed(123)
trainSize<-round(nrow(df_exsprod_ed5)*0.7) 
testSize<-nrow(df_exsprod_ed5)-trainSize

training_indices<-sample(seq_len(nrow(df_exsprod_ed5)),size =trainSize)

trainSet<-df_exsprod_ed5[training_indices,]

testSet<-df_exsprod_ed5[-training_indices,] 


# DEVELOP MULTIPLE REGRESSION MODELS

Vol_LM<-lm(Volume~.,data=trainSet)

#summary(Vol_LM)

hist(residuals(Vol_LM),col="darkgreen",breaks=10)

Vol_LM_resd<-residuals(Vol_LM)

qqnorm(Vol_LM_resd, pch = 1, frame = FALSE)

qqline(Vol_LM_resd, col = "red", lwd = 2)

Vol_LM_preds<-predict(Vol_LM,newdata=trainSet,
                      interval='none',
                      type="response")

plot(trainSet$Volume,Vol_LM_preds,
     xlab="Actual Volumes",
     ylab="Predicted Volumes",
     xlim=c(-1000,8000),
     ylim=c(-1000,8000))
abline(a=0,b=1,col="red")

# BUILD NON-PARAMETRIC MODELS

# SVM
library(e1071)
Vol_SVM<-svm(Volume~.,data=trainSet)

# grid search for tuning
Vol_SVM_tuning<-tune(svm,Volume~.,data=trainSet,
                     ranges=list(epsilon=seq(0,0.4,0.01),
                                 cost=2^(2:8)))

plot(Vol_SVM_tuning)

#select the best model from the grid search
Vol_SVM_tuned<-Vol_SVM_tuning$best.model
Vol_SVM_preds<-predict(Vol_SVM_tuned,trainSet)

plot(trainSet$Volume,Vol_SVM_preds,
     xlab="Actual Volumes",
     ylab="Predicted Volumes",
     xlim=c(-1000,8000),
     ylim=c(-1000,8000))
abline(a=0,b=1,col="red")

r2_Vol_SVM_tuned<-R2(trainSet$Volume,Vol_SVM_preds)



# Random forest

library(randomForest)

control<-trainControl(method = "repeatedcv",
                      number=5, repeats=3)
metric_RF<-"RMSE"
tunegrid<-expand.grid(.mtry=c(1:5))
Vol_RF_tuning<-train(Volume~.,data=trainSet,method="rf",
                  metric=metric_RF,tuneGrid=tunegrid,
                  trControl=control)

plot(Vol_RF_tuning)

Vol_RF_preds<-predict(Vol_RF_tuning,trainSet)

plot(trainSet$Volume,Vol_RF_preds,
     xlab="Actual Volumes",
     ylab="Predicted Volumes",
     xlim=c(-1000,8000),
     ylim=c(-1000,8000))
abline(a=0,b=1,col="red")

Vol_RF_VarImp<-varImp(Vol_RF_tuning)

# Gradient Boosting
Vol_GBM_tuning<-train(Volume~.,data=trainSet,method='gbm',
                 preProcess=c('scale','center'),
                 trControl=control)



Vol_GBM_tuning #this displays model summary in console

plot(Vol_GBM_tuning)

Vol_GBM_tuned<-Vol_GBM_tuning$finalModel

Vol_GBM_preds<-predict(Vol_GBM_tuned,trainSet)

plot(trainSet$Volume,Vol_GBM_preds,
     xlab="Actual Volumes",
     ylab="Predicted Volumes",
     xlim=c(-1000,8000),
     ylim=c(-1000,8000))
abline(a=0,b=1,col="red")

library(gbm)
Vol_GBM_VarImp<-varImp(Vol_GBM_tuned)

# COMPARE MODELS, CHECK FOR OVERFITTING
# by getting metrics on test data-> larger errors
#than in training data

# obtain each model predictions using the test set

# LM
Test_Vol_LM_preds<-predict(Vol_LM,newdata=testSet,
                           interval='none',
                           type="response")

plot(testSet$Volume,Test_Vol_LM_preds,
     xlab="Actual Volumes",
     ylab="Predicted Volumes",
     xlim=c(-1000,8000),
     ylim=c(-1000,8000))
abline(a=0,b=1,col="blue")

Train_RMSE_LM<-RMSE(Vol_LM_preds,trainSet$Volume)
Test_RMSE_LM<-RMSE(Test_Vol_LM_preds,testSet$Volume)

# SVM
Test_Vol_SVM_preds<-predict(Vol_SVM_tuned,newdata=testSet,
                           interval='none',
                           type="response")

plot(testSet$Volume,Test_Vol_SVM_preds,
     xlab="Actual Volumes",
     ylab="Predicted Volumes",
     xlim=c(-1000,8000),
     ylim=c(-1000,8000))
abline(a=0,b=1,col="blue")

Train_RMSE_SVM<-RMSE(Vol_SVM_preds,trainSet$Volume)
Test_RMSE_SVM<-RMSE(Test_Vol_SVM_preds,testSet$Volume)


# RF
Test_Vol_RF_preds<-predict(Vol_RF_tuning,newdata=testSet)

plot(testSet$Volume,Test_Vol_RF_preds,
     xlab="Actual Volumes",
     ylab="Predicted Volumes",
     xlim=c(-1000,8000),
     ylim=c(-1000,8000))
abline(a=0,b=1,col="blue")

Train_RMSE_RF<-RMSE(Vol_RF_preds,trainSet$Volume)
Test_RMSE_RF<-RMSE(Test_Vol_RF_preds,testSet$Volume)

# GBM
Test_Vol_GBM_preds<-predict(Vol_GBM_tuned,newdata=testSet)

plot(testSet$Volume,Test_Vol_GBM_preds,
     xlab="Actual Volumes",
     ylab="Predicted Volumes",
     xlim=c(-1000,8000),
     ylim=c(-1000,8000))
abline(a=0,b=1,col="blue")

Train_RMSE_GBM<-RMSE(Vol_GBM_preds,trainSet$Volume)
Test_RMSE_GBM<-RMSE(Test_Vol_GBM_preds,testSet$Volume)

# MODEL SELECTED: Random Forest
# apply RF to new products data

#generate a copy
df_newprod_ed6<-df_newprod_ed5
df_newprod_ed6$Predicted_Volume<-predict(Vol_RF_tuning,
                               newdata=df_newprod_ed6)

#convert ProductType back to characters using dictionary

df_newprod_ed7<-df_newprod_ed6
df_newprod_ed7$ProductType<-with(df_ProductType_dict,
                                 ProductType[match(df_newprod_ed6$ProductType,ID)])


write.csv(df_newprod_ed7,"RamsesMeza_C3T3_newprod_pred.csv",
          row.names=FALSE)

