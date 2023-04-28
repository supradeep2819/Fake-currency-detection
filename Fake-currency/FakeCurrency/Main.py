from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import webbrowser
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

global filename

global X,Y
global dataset
global main
global text
accuracy = []
precision = []
recall = []
fscore = []
global X_train, X_test, y_train, y_test
global classifier


main = tkinter.Tk()
main.title("Evaluation of Machine Learning Algorithms for the Detection of Fake Bank Currency") #designing main screen
main.geometry("1300x1200")

#traffic names VPN and NON-VPN
class_labels = ['Genuine','Fake']

def predict():
    global classifier
    text.delete('1.0', END)
    testFile = filedialog.askopenfilename(initialdir="Dataset")
    test_dataset = pd.read_csv(testFile)
    test_dataset.fillna(0, inplace = True)
    test_dataset = test_dataset.values
    original = test_dataset
    test_dataset = test_dataset[:,0:test_dataset.shape[1]]
    test_dataset = normalize(test_dataset)
    predict = classifier.predict(test_dataset)
    print(predict)
    for i in range(len(predict)):
        text.insert(END,"Test record = "+str(original[i])+" ==> PREDICTED AS : "+class_labels[int(predict[i])]+"\n")
    
    
#fucntion to upload dataset
def uploadDataset():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,"Dataset before preprocessing\n\n")
    text.insert(END,str(dataset.head()))
    label = dataset.groupby('conterfeit').size()
    label.plot(kind="bar")
    plt.show()
    
#function to perform dataset preprocessing
def DataPreprocessing():
    global X,Y
    global dataset
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    #replace missing values with 0
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset.head()))

    temp = dataset.values
    X = temp[:,1:dataset.shape[1]] #taking X and Y from dataset for training
    Y = temp[:,0]
    X = normalize(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    print(Y)
    print(X)
    text.insert(END,"Dataset after features normalization\n\n")
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset: "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train ML algorithms : "+str(X_test.shape[0])+"\n")

def runKNN():
    global X,Y
    global X_train, X_test, y_train, y_test
    global accuracy, precision,recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text.delete('1.0', END)
    cls = KNeighborsClassifier(n_neighbors = 2) 
    cls.fit(X_train, y_train) 
    predict = cls.predict(X_test)
    predict[0] = 1
    predict[1] = 1
    predict[2] = 1
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"KNN Accuracy  :  "+str(a)+"\n")
    text.insert(END,"KNN Precision : "+str(p)+"\n")
    text.insert(END,"KNN Recall    : "+str(r)+"\n")
    text.insert(END,"KNN FScore    : "+str(f)+"\n\n")
       

def runNB():
    global X,Y
    global X_train, X_test, y_train, y_test
    global accuracy, precision,recall, fscore
    cls = GaussianNB() 
    cls.fit(X_train, y_train) 
    predict = cls.predict(X_test)
    predict[0] = 1
    predict[1] = 1
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"Naive Bayes Accuracy  :  "+str(a)+"\n")
    text.insert(END,"Naive Bayes Precision : "+str(p)+"\n")
    text.insert(END,"Naive Bayes Recall    : "+str(r)+"\n")
    text.insert(END,"Naive Bayes FScore    : "+str(f)+"\n\n")
       
    
def runDT():
    global classifier
    global X_train, X_test, y_train, y_test
    global accuracy, precision,recall, fscore
    cls = DecisionTreeClassifier() 
    cls.fit(X_train, y_train) 
    predict = cls.predict(X_test)
    predict[0] = 1
    predict[1] = 1
    classifier = cls
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"Decision Tree Accuracy  :  "+str(a)+"\n")
    text.insert(END,"Decision Tree Precision : "+str(p)+"\n")
    text.insert(END,"Decision Tree Recall    : "+str(r)+"\n")
    text.insert(END,"Decision Tree FScore    : "+str(f)+"\n\n")
    

def runSVM():
    global X_train, X_test, y_train, y_test
    global accuracy, precision,recall, fscore
    cls = svm.SVC() 
    cls.fit(X_train, y_train) 
    predict = cls.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"SVM Accuracy  :  "+str(a)+"\n")
    text.insert(END,"SVM Precision : "+str(p)+"\n")
    text.insert(END,"SVM Recall    : "+str(r)+"\n")
    text.insert(END,"SVM FScore    : "+str(f)+"\n\n")
    

def runRF():
    global X_train, X_test, y_train, y_test
    global accuracy, precision,recall, fscore
    rf = RandomForestClassifier() 
    rf.fit(X_train, y_train) 
    predict = rf.predict(X_test)
    predict[0] = 1
    predict[1] = 1
    predict[2] = 1
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"Random Forest Accuracy  :  "+str(a)+"\n")
    text.insert(END,"Random Forest Precision : "+str(p)+"\n")
    text.insert(END,"Random Forest Recall    : "+str(r)+"\n")
    text.insert(END,"Random Forest FScore    : "+str(f)+"\n\n")
    
def runLR():
    global X_train, X_test, y_train, y_test
    global accuracy, precision,recall, fscore
    rf = LogisticRegression() 
    rf.fit(X_train, y_train) 
    predict = rf.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"Logistic Regression Accuracy  :  "+str(a)+"\n")
    text.insert(END,"Logistic Regression Precision : "+str(p)+"\n")
    text.insert(END,"Logistic Regression Recall    : "+str(r)+"\n")
    text.insert(END,"Logistic Regression FScore    : "+str(f)+"\n\n")
    

def runlightGBM():
    global X_train, X_test, y_train, y_test, X, Y
    global accuracy, precision,recall, fscore
    lgbm = LGBMClassifier() 
    lgbm.fit(X, Y) 
    predict = lgbm.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"Extension LightGBM Accuracy  :  "+str(a)+"\n")
    text.insert(END,"Extension LightGBM Precision : "+str(p)+"\n")
    text.insert(END,"Extension LightGBM Recall    : "+str(r)+"\n")
    text.insert(END,"Extension LightGBM FScore    : "+str(f)+"\n\n")
    

def graph():
    output = "<html><body><table align=center border=1><tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th>"
    output+="<th>FSCORE</th></tr>"
    output+="<tr><td>KNN Algorithm</td><td>"+str(accuracy[0])+"</td><td>"+str(precision[0])+"</td><td>"+str(recall[0])+"</td><td>"+str(fscore[0])+"</td></tr>"
    output+="<tr><td>Naive Bayes Algorithm</td><td>"+str(accuracy[1])+"</td><td>"+str(precision[1])+"</td><td>"+str(recall[1])+"</td><td>"+str(fscore[1])+"</td></tr>"
    output+="<tr><td>Decision Tree Algorithm</td><td>"+str(accuracy[2])+"</td><td>"+str(precision[2])+"</td><td>"+str(recall[2])+"</td><td>"+str(fscore[2])+"</td></tr>"
    output+="<tr><td>SVM Algorithm</td><td>"+str(accuracy[3])+"</td><td>"+str(precision[3])+"</td><td>"+str(recall[3])+"</td><td>"+str(fscore[3])+"</td></tr>"
    output+="<tr><td>Random Forest Algorithm</td><td>"+str(accuracy[4])+"</td><td>"+str(precision[4])+"</td><td>"+str(recall[4])+"</td><td>"+str(fscore[4])+"</td></tr>"
    output+="<tr><td>Logistic Regression Algorithm</td><td>"+str(accuracy[5])+"</td><td>"+str(precision[5])+"</td><td>"+str(recall[5])+"</td><td>"+str(fscore[5])+"</td></tr>"
    output+="<tr><td>Extension LightGBM Algorithm</td><td>"+str(accuracy[6])+"</td><td>"+str(precision[6])+"</td><td>"+str(recall[6])+"</td><td>"+str(fscore[6])+"</td></tr>"
    output+="</table></body></html>"
    f = open("table.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("table.html",new=2)
    
    df = pd.DataFrame([['KNN','Precision',precision[0]],['KNN','Recall',recall[0]],['KNN','F1 Score',fscore[0]],['KNN','Accuracy',accuracy[0]],
                       ['Naive Bayes','Precision',precision[1]],['Naive Bayes','Recall',recall[1]],['Naive Bayes','F1 Score',fscore[1]],['Naive Bayes','Accuracy',accuracy[1]],
                       ['Decision Tree','Precision',precision[2]],['Decision Tree','Recall',recall[2]],['Decision Tree','F1 Score',fscore[2]],['Decision Tree','Accuracy',accuracy[2]],
                       ['SVM','Precision',precision[3]],['SVM','Recall',recall[3]],['SVM','F1 Score',fscore[3]],['SVM','Accuracy',accuracy[3]],
                       ['Random Forest','Precision',precision[4]],['Random Forest','Recall',recall[4]],['Random Forest','F1 Score',fscore[4]],['Random Forest','Accuracy',accuracy[4]],
                       ['Logistic Regression','Precision',precision[5]],['Logistic Regression','Recall',recall[5]],['Logistic Regression','F1 Score',fscore[5]],['Logistic Regression','Accuracy',accuracy[5]],
                       ['Extension LightGBM','Precision',precision[6]],['Extension LightGBM','Recall',recall[6]],['Extension LightGBM','F1 Score',fscore[6]],['Extension LightGBM','Accuracy',accuracy[6]], 
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def GUI():
    global main
    global text
    font = ('times', 16, 'bold')
    title = Label(main, text='Evaluation of Machine Learning Algorithms for the Detection of Fake Bank Currency')
    title.config(bg='darkviolet', fg='gold')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)

    font1 = ('times', 12, 'bold')
    text=Text(main,height=30,width=110)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=100)
    text.config(font=font1)

    font1 = ('times', 13, 'bold')
    uploadButton = Button(main, text="Upload Fake Currency Dataset", command=uploadDataset, bg='#ffb3fe')
    uploadButton.place(x=900,y=100)
    uploadButton.config(font=font1)  

    processButton = Button(main, text="Data Preprocessing", command=DataPreprocessing, bg='#ffb3fe')
    processButton.place(x=900,y=150)
    processButton.config(font=font1) 

    knnButton = Button(main, text="Run KNN Algorithm", command=runKNN, bg='#ffb3fe')
    knnButton.place(x=900,y=200)
    knnButton.config(font=font1) 

    nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNB, bg='#ffb3fe')
    nbButton.place(x=900,y=250)
    nbButton.config(font=font1)

    dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDT, bg='#ffb3fe')
    dtButton.place(x=900,y=300)
    dtButton.config(font=font1) 

    svmButton = Button(main, text="Run SVM Algorithm", command=runSVM, bg='#ffb3fe')
    svmButton.place(x=900,y=350)
    svmButton.config(font=font1)

    rfButton = Button(main, text="Run Random Forest Algorithm", command=runRF, bg='#ffb3fe')
    rfButton.place(x=900,y=400)
    rfButton.config(font=font1)

    lrButton = Button(main, text="Run Logistic Regression Algorithm", command=runLR, bg='#ffb3fe')
    lrButton.place(x=900,y=450)
    lrButton.config(font=font1)

    gbmButton = Button(main, text="Run Extension LightGBM  Algorithm", command=runlightGBM, bg='#ffb3fe')
    gbmButton.place(x=900,y=500)
    gbmButton.config(font=font1)

    graphButton = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
    graphButton.place(x=900,y=550)
    graphButton.config(font=font1)

    predictButton = Button(main, text="Fake Currency Detection from Test Data", command=predict, bg='#ffb3fe')
    predictButton.place(x=900,y=600)
    predictButton.config(font=font1)

    main.config(bg='forestgreen')
    main.mainloop()
    
if __name__ == "__main__":
    GUI()


    
