#! /usr/bin/pythonw
#This file process data

import numpy as np
import random
import csv
class Data:
    """
    load data, filter unknown features,
    """

    data_line_full = []
    filtered_data = []
    teacher_label = []
    data_matrix = np.mat
    test_matrix = np.mat
    train_matrix = np.mat
    train_class_list = []
    test_class_list = []
    header_line = []

    def __init__(self):
        pass

    #load data from disk
    def loadDataSet(self):
        fr = open("../resources/bank-additional-full.csv");
        self.header_line = next(fr).split(";")
        for line in fr:
            self.data_line_full.append(line.split(";"))

    #omit data with both unknown features attributes && with duration attribute column
    def filterDataSet(self):
        lines_omit = 0
        lines_left = 0
        duration_idx = self.header_line.index('"duration"')
        for lists in self.data_line_full:
            if lists.count('"unknown"'):
                lines_omit += 1
            else:
                del lists[duration_idx]
                self.filtered_data.append(lists)
                lines_left += 1

    # convert all value into double
    def convertCatToNumerical(self, index, attribute):

        if index == 1:
            if attribute == '"admin."':
                return 1
            elif attribute == '"blue-collar"':
                return 2
            elif attribute == '"entrepreneur"':
                return 3
            elif attribute == '"housemaid"':
                return 4
            elif attribute == '"management"':
                return 5
            elif attribute == '"retired"':
                return 6
            elif attribute == '"self-employed"':
                return 7
            elif attribute == '"services"':
                return 8
            elif attribute == '"student"':
                return 9
            elif attribute == '"technician"':
                return 10
            elif attribute == '"unemployed"':
                return 11
            else:
                raise NameError('error in converCatToNumerical function, no this job title!')

        elif index == 2:
            if attribute == '"divorced"':
                return 1
            elif attribute == '"married"':
                return 2
            elif attribute == '"single"':
                return 3
            else:
                raise NameError('error in converCatToNumerical function, no this marital status!')


        elif index == 3:
            if attribute == '"basic.4y"':
                return 1
            elif attribute == '"basic.6y"':
                return 2
            elif attribute == '"basic.9y"':
                return 3
            elif attribute == '"high.school"':
                return 4
            elif attribute == '"illiterate"':
                return 5
            elif attribute == '"professional.course"':
                return 6
            elif attribute == '"university.degree"':
                return 7
            else:
                raise NameError('error in converCatToNumerical function, no this education level!')

        elif index == 4:
            if attribute == '"no"':
                return 1
            elif attribute == '"yes"':
                return 2
            else:
                raise NameError('error in converCatToNumerical function, index==4!')

        elif index == 5:
            if attribute == '"no"':
                return 1
            elif attribute == '"yes"':
                return 2
            else:
                raise NameError('error in converCatToNumerical function, index==5!')

        elif index == 6:
            if attribute == '"no"':
                return 1
            elif attribute == '"yes"':
                return 2
            else:
                raise NameError('error in converCatToNumerical function, index==6!')

        elif index == 7:
            if attribute == '"cellular"':
                return 1
            elif attribute == '"telephone"':
                return 2
            else:
                raise NameError('error in converCatToNumerical function, index==7!')

        elif index == 8:
            if attribute == '"jan"':
                return 1
            elif attribute == '"feb"':
                return 2
            elif attribute == '"mar"':
                return 3
            elif attribute == '"apr"':
                return 4
            elif attribute == '"may"':
                return 5
            elif attribute == '"jun"':
                return 6
            elif attribute == '"jul"':
                return 7
            elif attribute == '"aug"':
                return 8
            elif attribute == '"sep"':
                return 9
            elif attribute == '"oct"':
                return 10
            elif attribute == '"nov"':
                return 11
            elif attribute == '"dec"':
                return 12
            else:
                raise NameError('error in converCatToNumerical function, index==8!')

        elif index == 9:
            if attribute == '"mon"':
                return 1
            elif attribute == '"tue"':
                return 2
            elif attribute == '"wed"':
                return 3
            elif attribute == '"thu"':
                return 4
            elif attribute == '"fri"':
                return 5

        elif index == 13:
            if attribute == '"failure"':
                return 1
            elif attribute == '"nonexistent"':
                return 2
            elif attribute == '"success"':
                return 3
            else:
                raise NameError('error in converCatToNumerical function, index==14!')

        else:
            return float(attribute)

    #parse teacher label and return
    def parseTeacherLabel(self, str_in):
        if str_in == '"no"\r\n':
            return 0
        else:
            return 1

    # convert all value into numpy mat
    def convertAttrToMatrix(self):
        temp_array = np.zeros((len(self.filtered_data), len(self.header_line)-2), dtype=float)
        for idx_list, data_line in enumerate(self.filtered_data):
            self.teacher_label.append(self.parseTeacherLabel(data_line.pop()))
            for idx_line, element in enumerate(data_line):
                temp_array[idx_list][idx_line] = self.convertCatToNumerical(idx_line, element)
        self.data_matrix = np.asmatrix(temp_array)

    #split data into train(2/3), and test(1/3)
    def splitToTrainAndTest(self):
        m, n = np.shape(self.data_matrix)
        self.train_matrix, self.test_matrix = np.vsplit(self.data_matrix, np.array([2*m/3]))
        m_tr, n_tr = np.shape(self.train_matrix)
        self.train_class_list = self.teacher_label[0:m_tr]  #get the label of training example
        self.test_class_list = self.teacher_label[m_tr:m] #get the label of testing example

    #get part of the train data matrix by index
    def getRandomIndexList(self, length):
        m, n = np.shape(self.train_matrix)
        return random.sample(range(m), length)

    def writeToCsvFile(self, file_path, content):
        with open(file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerows(content)


