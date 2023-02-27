package main

import (
	"fmt"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	fmt.Println("Loading CSV data")
	rawData, err := base.ParseCSVToInstances("iris_headers.csv", true)
	if err != nil {
		panic(err)
	}

	fmt.Println("Initializing KNN Classifer")
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)

	fmt.Println("Performing Train-Test Split")
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)
	cls.Fir(trainData)

	fmt.Println("Calculating the Euclidean Distance and Return the Most Popular Label")
	predictions, err := cls.Predict(testData)
	if err != nil {
		panic((err))
	}

	fmt.Println(predictions)

	fmt.Println("Printing Metrics Summary")

	confusionMatrix, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get Confusion Matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMatrix))
}
