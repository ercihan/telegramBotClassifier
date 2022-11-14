from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import SVC
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from random import randrange
from sklearn.model_selection import train_test_split
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from joblib import dump, load

import os
from PIL import Image
import PIL.ImageOps 

labels = ['T-shirt', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal',
            'Shirt', 'Sneaker', 'Bag', 'Ankle boat']

def import_file_train(file_path):
    return pd.read_csv(file_path)

def import_file_test(file_path):
    return pd.read_csv(file_path)

def normalisiere(data): #Hier wird der Datenarray gegeben
    mean = np.mean(data) # Das ist der durchschnitt ueber den Datenarray
    std = np.std(data) #Standartabweichung wird berechnet
    return (data - mean) / std #Das ist nun die eigentliche normalisierung aehnlich wie Vektornormalisierung -> vektor durch betrag von vektor -> heisst datensets-array - mittelwert durch standardabweichung (=mittlere Abweichung der Streuung)

def normalize_from_algo(x_train, x_valid, x_test):
    x_train = normalisiere(x_train)
    x_valid = normalisiere(x_valid)
    x_test = normalisiere(x_test)
    return x_train, x_valid, x_test

def solveAPRF(expect, pred, user,algo):
    accuracy = accuracy_score(expect, pred)
    print('Accuracy: ', accuracy) # Accuracy:  Korrekte Test Samples / Alle Samples
    print('Precision MICRO: ', precision_score(expect, pred, average='micro')) #Precision: true_positives / (true_positives + false_positives) 
    print('Precision MACRO: ', precision_score(expect, pred, average='macro'))
    print('Recall MICRO: ', recall_score(expect, pred, average='micro')) # Recall:    (alle Elemente die als true klassifiziert wurden und die auch true sind) / (Alle tatsächlich positiven Dokumente)
    print('Recall MACRO: ', recall_score(expect, pred, average='macro'))
    print('f1Score MICRO: ', f1_score(expect, pred, average='micro')) #f1Score:   2 * ((Precision * Recall) / (Precision + Recall))
    print('f1Score MACRO: ', f1_score(expect, pred, average='macro'))
    confusion_matrix_df = pd.DataFrame(confusion_matrix(expect, pred), index=labels, columns=labels) #generiert uns anahnd der exprectation und prediction (expectation = values in db) (prediction = values die unser Modell klassifiziert) eine Confusion Matrix anhand der Labels (=kategorien)
    """A confusion matrix is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. This is the key to the confusion matrix."""
    plt.figure(figsize=(10,6))
    sns.heatmap(confusion_matrix_df, annot=True, fmt="d", annot_kws={"size": 10})
    plt.savefig('./TelegramStorage/'+user+'_'+algo+'.png')
    return accuracy

def init(file_path_test,file_path_train):
    df_train = import_file_train(file_path_train)
    df_test = import_file_test(file_path_test)
    x_train = df_train.drop(['Id', 'Category'], axis=1) # Wir haben bei der Anzeige 2 mal die ID die gleich wie der Index ist -> also loeschen wir diese raus
    y_train = df_train['Category']
    x_test = df_test.drop(['Id'],axis=1)
    
    x_train = np.array(x_train, dtype='float32') #implizites Konvertieren in einen array
    y_train = np.array(y_train, dtype='int64')
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42) 

    x_train = np.array(x_train, dtype='float32') #implizites Konvertieren in einen array
    x_train, x_valid, x_test = normalize_from_algo(x_train, x_valid, x_test)

    train_indic = np.full((x_train.shape[0],), -1, dtype=int) # Return a new array of given shape and type, filled with fill_value.
    test_indic  = np.full((x_valid.shape[0],), 0, dtype=int)
    ps = PredefinedSplit(np.append(train_indic, test_indic))
    return df_train, df_test, x_train, y_train, x_test, x_valid, y_valid, ps

def dt(file_path_test, file_path_train, user):
    df_train, df_test, x_train, y_train, x_test, x_valid, y_valid, ps = init(file_path_test, file_path_train)

    parameters = {'max_depth': [3,6], 'criterion': ['gini', 'entropy']}
    rfc = RandomForestClassifier(random_state=0) #Nur die bereits vorhandene Daten verwenden, und nichts dazuerfinden (Random-maessig)
    clf = GridSearchCV(rfc, param_grid=parameters, cv=ps)
    beforeTrainDT = datetime.now()
    clf.fit(np.append(x_train, x_valid, axis=0), np.append(y_train, y_valid, axis=0))
    afterTrainDT = datetime.now()
    print(clf.best_params_)

    y_valid_prediction = clf.predict(x_valid)
    dump(clf, './TelegramStorage/'+user+'_dt.joblib') 
    return solveAPRF(y_valid, y_valid_prediction, user,'dt')

def lr(file_path_test, file_path_train, user):
    df_train, df_test, x_train, y_train, x_test, x_valid, y_valid, ps = init(file_path_test, file_path_train)
    parameters = {'C': [1., 1.3, 1.5], 'solver': ['lbfgs', 'sag']} #Parameters welche solver sollen verwendet werden welche Gewichtungen
    # Die Gewichtung ist abhaengig vom Datensatz je nach dem wie die Verteilung von den Samples von den beiden oder mehreren Klassen ist muss eine Klasse mehr als die andere gewichtet werden

    confMatr = LogisticRegression(random_state=0, max_iter=150) # Logisitische Regression = Solves a classification task -> heisst hier koennen wir mit statischen daten random_state=0 ; max_iter=Maximum number of iterations taken for the solvers to converge.
    clf = GridSearchCV(confMatr, param_grid=parameters, cv=ps) # cv = splitter function
    beforeTrainLogisRegr = datetime.now()
    clf.fit(np.append(x_train, x_valid, axis=0), np.append(y_train, y_valid, axis=0)) # Das definierte Model wird nun mit den Training und validation sets gefittet -> Trainiert
    afterTrainLogisRegr = datetime.now()
    print(clf.best_params_)
    y_valid_prediction = clf.predict(x_valid)
    accuracy = solveAPRF(y_valid, y_valid_prediction,user,'lr')
    dump(clf, './TelegramStorage/'+user+'_lr.joblib') 
    return accuracy



def svm(file_path_test, file_path_train, user):
    df_train, df_test, x_train, y_train, x_test, x_valid, y_valid, ps = init(file_path_test, file_path_train)
    
    parameters = {'C': [0.7, 1., 1.3], 'gamma': ['scale', 'auto']} #Gewichtung der Klassen ändern, wenn sie mehr Datasets haben sind Fehler irrelevanter mit weniger Datasets sind Fehler relevanter -> Gewichtung der Klassen muss unterschiedlich sein
    svc = SVC(random_state = 0) #Nur die bereits vorhandene Daten verwenden, und nichts dazuerfinden (Random-maessig)
    clf = GridSearchCV(svc, param_grid=parameters, cv=ps)
    beforeTrainSVC = datetime.now()
    clf.fit(np.append(x_train, x_valid, axis=0), np.append(y_train, y_valid, axis=0))
    afterTrainSVC = datetime.now()
    y_predictions = clf.predict(x_valid)
    dump(clf, './TelegramStorage/'+user+'_svm.joblib') 
    return solveAPRF(y_valid, y_predictions,user,'svm')

def cnn(file_path_test, file_path_train, user):
    df_train, df_test, x_train, y_train, x_test, x_valid, y_valid, ps = init(file_path_test, file_path_train)
    modell = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding="same", input_shape=(28,28,1)),#heisst es sind 64 Kernels a 3x3 Werte
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding="same",), # also wird von 64 auf 128 Neuronen erhoeht, da man nun ja weniger pixel hat welche verarbeitet werden muessen also weniger komplex ist und man dem Faktor 2 reduziert wurde muss man die neuronen mit faktor 2 multiplizieren um dem entgegenzuwirken
        tf.keras.layers.MaxPooling2D(2,2), #Es wird die Komplexitaet um faktor 2 reduziert
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding="same",),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding="same",),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding="same",),
        tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding="same",),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3), # Wir loeschen randomisiert 30% der Neuronen damit dsa Neurale Netz selbst besser funzt -> unnuetze Neuronen reduzieren ist ds Ziel
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'), # Da wir am schluss 10 Kategorien haben wollen muessen wir den Softmax-Algorithmus verwednen
    ])
    
    
    parent_dir = "./TelegramStorage/"
    path = os.path.join(parent_dir, user)
    os.mkdir(path)

    file_path = './TelegramStorage/'+user#'vars/vars.ckpt'
    modell_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath = file_path,
        save_weight_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    x_train_tensflo = x_train.reshape(x_train.shape[0], 28, 28)
    x_train_tensflo = np.expand_dims(x_train_tensflo, -1) # ndim - 1 will be treated as axis == 0
    x_valid_tensflo = x_valid.reshape(x_valid.shape[0], 28, 28)
    x_valid_tensflo = np.expand_dims(x_valid_tensflo, -1)

 
    modell.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    beforeTrainCNN = datetime.now()
    histogram = modell.fit(
        x_train_tensflo,
        y_train,
        batch_size=1000,
        epochs=30,
        verbose=1,
        validation_data=(x_valid_tensflo, y_valid),
        callbacks=[modell_cb],
    )
    afterTrainCNN = datetime.now()

    modell.load_weights(file_path)
    y_valid_predi = modell.predict(x_valid_tensflo)
    return solveAPRF(y_valid,np.argmax(y_valid_predi, axis=1),user,'cnn')


def classify(user): 
    clf = load('./TelegramStorage/'+user+'_dt.joblib') 
    print('classifying')
    filename = './TelegramStorage/'+user+'.jpg'
    
    image = Image.open(filename)
    p = plt.imshow(np.asarray(image), cmap=plt.cm.gray,)
    p = plt.title('Shape: ' + str(np.asarray(image).shape))

    # convert to grayscale image and resize
    image_bw = image.convert('L')
    p = plt.imshow(np.asarray(image_bw), cmap=plt.cm.gray,)
    p = plt.title('Shape: ' + str(np.asarray(image_bw).shape))
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    p = plt.imshow(np.asarray(image_bw_resized), cmap=plt.cm.gray,)
    p = plt.title('Shape: ' + str(np.asarray(image_bw_resized).shape))

    # invert image to match training data
    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    p = plt.imshow(np.asarray(image_bw_resized_inverted), cmap=plt.cm.gray,)
    p = plt.title('Shape: ' + str(np.asarray(image_bw_resized_inverted).shape))

    # adjust contrast and scale
    pixel_filter = 20 # value from 0 to 100
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    p = plt.imshow(np.asarray(image_bw_resized_inverted_scaled), cmap=plt.cm.gray,)
    p = plt.title('Shape: ' + str(np.asarray(image_bw_resized_inverted_scaled).shape))

    # finally, reshape to 1 sample and 784 features
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    p = plt.imshow(np.reshape(test_sample, (28,28)), cmap=plt.cm.gray,)
    p = plt.title('Shape: ' + str(test_sample.shape))
    p = plt.imshow(np.reshape(test_sample, (28,28)), cmap=plt.cm.gray,)
    p = plt.title('Shape: ' + str(test_sample.shape))
    plt.savefig('./TelegramStorage/debugimg.png')

    test_probs = clf.predict_proba(test_sample)
    import matplotlib.pyplot as plt2
    plt2.figure()
    sns.barplot(labels, test_probs.squeeze())
    plt2.ylabel("Probability")
    plt2.xlabel("Class")
    plt2.xticks(rotation=90)
    plt2.savefig('./TelegramStorage/fig.png')
        
    test_pred = clf.predict(test_sample)
    print("Predicted class is: ", test_pred)

    print(labels[test_pred[0]])
    return labels[test_pred[0]], p
