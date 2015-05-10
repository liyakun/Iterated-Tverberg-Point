#! /usr/bin/pythonw
#This file contains the main function

import Regression
import Data

#filter data with unknown features && create dummy mat matrix
myData = Data.Data()
myData.loadDataSet()
myData.filterDataSet()
myData.convertAttrToMatrix()
matrix_data, teacher_label = myData.returnPartDataMatrix(1, 500)

#regression algorithm
myRegression = Regression.Regression()
weights = myRegression.gradAscent(matrix_data, teacher_label, 300)




