# Machine-Learning

library(rpart); library(rattle); library(caret); library(randomForest)

# Download the datasets
UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
UrlTest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(UrlTrain))
testing  <- read.csv(url(UrlTest))

# Partiting the training dataset 
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainSet <- training[inTrain, ]
TestSet  <- training[-inTrain, ]
dim(TrainSet)

# Cleaning the data
## removing columns with many NAs
NAcol    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, NAcol==FALSE]
TestSet  <- TestSet[, NAcol==FALSE]
## removing low-variance columnes
zerocol <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -zerocol]
TestSet  <- TestSet[, -zerocol]
## removing unnecessary identification columns
TrainSet <- TrainSet[, -(1:6)]
TestSet  <- TestSet[, -(1:6)]
dim(TrainSet); dim(TestSet)

# Classification tree model
set.seed(987)
ClassTree <- rpart(classe ~ ., data=TrainSet, method="class")
fancyRpartPlot(ClassTree,cex=.5,under.cex=1,shadow.offset=0)
predictTree <- predict(ClassTree, newdata=TestSet, type="class")
cmtree<-confusionMatrix(predictTree, TestSet$classe)
cmtree

# Random forest model
set.seed(1475)
Forest<-train(classe~., data=TrainSet, method="rf", trControl=trainControl(method="cv", number=3, verboseIter=FALSE))
Forest$finalModel
predictForest <- predict(Forest, newdata=TestSet)
cmforest<-confusionMatrix(predictForest, TestSet$classe)
cmforest

# visualisation
par(mfrow=c(1,2))
plot(cmtree$table, main=paste(c("Classific Tree Accuracy Rate=",round(cmtree$overall['Accuracy'],3))), col="red")
plot(cmforest$table, main=paste(c("Random Forest Accuracy Rate=",round(cmforest$overall['Accuracy'],3))), col="green")

# Answering the quiz
quiz <- predict(Forest, newdata=testing)
quiz


