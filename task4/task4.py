# Predicting the speech sound (“phoneme”) from an acoustic feature vector (describing the acoustic content in a small frame of audio). This is also a classification task.

#**********************************
# get data
#**********************************

import pandas as pd

# filepath strings:
params = "proj_data/task4_phoneme/task4.config"
trainfeats = "proj_data/task4_phoneme/train.X"
traintargs = "proj_data/task4_phoneme/train.CT"
devfeats = "proj_data/task4_phoneme/dev.X"
devtargs = "proj_data/task4_phoneme/dev.CT"
# list of all files
filelist = [params, trainfeats, traintargs, devfeats, devtargs]
# lists that dataframes will be appended to
dataframelist = []
dfnum = 0

# this should loop through each file and make it into a dataframe
for i in range(len(filelist)):
    # open the file, create a new dataframe
    openfile = open(filelist[i], "r")
    dataframename = filelist[i] + "DF"
    dataframename = pd.DataFrame({'type': ["C"]})
    count = 1
    
    colnum = 0
    firstline = openfile.readline().strip() # get the first line to see how many dimensions we need
    for num in firstline.split():
            # make a new column for each dimension called D#
            colname = "D" + str(colnum)
            dataframename.insert(loc=colnum+ 1, column=colname, value="NULL")
            colnum += 1

    # make sure the first line we took out gets added!
    appendtoline = firstline.split()
    appendtoline.insert(0, "C") # mine is C for classification, you don't really need this but you have 
                                # to initialize the dataframe with something so i used the type of task
    dataframename.loc[0] = appendtoline # loc returns that row or sets its value

    # for each line in the file, read it, split it and insert the type of task, then use loc to enter it in DF
    for line in openfile:
        readline = line.strip()
        appendtoline = line.split()
        appendtoline.insert(0, "C")
        dataframename.loc[count] = appendtoline
        count += 1
    # add to our list of dataframes
    dataframelist.append(dataframename)

    # i just like checking that it looks right, probs only useful for the config file since everything else is too big
    print(dataframelist[dfnum])
    dfnum += 1

#**********************************
# put into usable format
#**********************************

#**********************************
# set up model
#**********************************

#**********************************
# train?
#**********************************

# um. ??? idk