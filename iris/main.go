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

	fmt.Println("Init