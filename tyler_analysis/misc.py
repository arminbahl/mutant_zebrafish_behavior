from plotFigs import *
from all_analysis import *

def plotAllAll(printPDF): # Plot everything
    for i in range(0,10):
        eval("p1('het', " + str(i) + ", " + str(printPDF) + ")")
        eval("p1('hom', " + str(i) + ", " + str(printPDF) + ")")
        eval("p2('het', " + str(i) + ", " + str(printPDF) + ")")
        eval("p2('hom', " + str(i) + ", " + str(printPDF) + ")")
        eval("p2a('het', " + str(i) + ", " + str(printPDF) + ")")
        eval("p2a('hom', " + str(i) + ", " + str(printPDF) + ")")
        eval("p2b('het', " + str(i) + ", " + str(printPDF) + ")")
        eval("p2b('hom', " + str(i) + ", " + str(printPDF) + ")")
        eval("p2v('het', " + str(i) + ", " + str(printPDF) + ")")
        eval("p2v('hom', " + str(i) + ", " + str(printPDF) + ")")
        eval("p3('het', " + str(i) + ", " + str(printPDF) + ")")
        eval("p3('hom', " + str(i) + ", " + str(printPDF) + ")")
        eval("p4('het', " + str(i) + ", " + str(printPDF) + ")")
        eval("p4('hom', " + str(i) + ", " + str(printPDF) + ")")

def plotAll(figIndex, printPDF): # Plot a figure for all fileIndex 0 through 9 and for all genotypes
    for i in range(0,10):
        print("Fig" + str(figIndex) + ", fileIndex:",i)
        eval("p"+str(figIndex)+"('het', "+str(i)+", "+str(printPDF)+")")
        if i<8:
        eval("p"+str(figIndex)+"('hom', "+str(i)+", "+str(printPDF)+")")

def plotAllGen(figIndex, gen, printPDF): # Plot a figure for all fileIndex 0 through 9 for a single genotype
    for i in range(0,10):
        print("Fig" + str(figIndex) + ", fileIndex:", i)
        eval("p"+str(figIndex)+"('"+gen+"', "+str(i)+", "+str(printPDF)+")")

def runAll(functionName): # Run analysis for all fileIndex 0 through 9 and for all genotypes
    for i in range(0,10):
        print(eval(str(functionName)+"('wt',"+str(i)+")"))
        print(eval(str(functionName)+"('het',"+str(i)+")"))
        print(eval(str(functionName) + "('hom'," + str(i) + ")"))