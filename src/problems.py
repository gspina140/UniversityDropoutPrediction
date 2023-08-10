import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from statistics import mean
from sklearn.metrics import classification_report

#Function to modify the Additional learning requirements as explained in the report
def ofa_values_filter (row):
    if row['OFA_assegnati'] == 0 :
        return 1
    if row['OFA_assegnati'] == 1 and row['OFA_superati'] == 0:
        return 2
    if row['OFA_assegnati'] == 1 and row['OFA_superati'] == 1:
        return 3
    return 0    

#Static data
df  = pd.read_excel('../dataset/data_dropout.xlsx')
#drop NaN values, reset column indexes
df = df.dropna().reset_index(drop=True)

#regarding the high school, when unknown the value was 99, converting it to 10 so the possible values: 1-10 (to remove "outliers")
df['Diploma_scuola_superiore'] = np.where(df['Diploma_scuola_superiore'] == 99, 10, df['Diploma_scuola_superiore']) 

#work on OFA to obtain 1 (no ofa), 2 (with ofa but not exceeded), 3 (with ofa and exceeded)
df['OFA'] = df.apply(lambda row: ofa_values_filter(row), axis=1)
df = df.drop('OFA_assegnati', axis=1) 
df = df.drop('OFA_superati', axis=1)

#extract birth year from date
df['DataNascita']=df['DataNascita'].dt.year

#Only master
lm = df.loc[df['TipoCorso'] == 'LM'].reset_index()
#3 range of ages at the start of university, for master 1 (lt 22), 2 (bt 22,25), 3(gt 25)
lm['DataNascita'] = np.where((lm['Coorte']-lm['DataNascita']) < 22, 1, lm['DataNascita'])
lm['DataNascita'] = np.where((lm['Coorte']-lm['DataNascita']).between(22,25), 2, lm['DataNascita'])
lm['DataNascita'] = np.where((lm['Coorte']-lm['DataNascita']) > 25, 3, lm['DataNascita'])

#3 range of ages at the start of university, 1 (lt 20), 2 (bt 20,23), 3(gt 23)
df['DataNascita'] = np.where((df['Coorte']-df['DataNascita']) < 20, 1, df['DataNascita'])
df['DataNascita'] = np.where((df['Coorte']-df['DataNascita']).between(20,23), 2, df['DataNascita'])
df['DataNascita'] = np.where((df['Coorte']-df['DataNascita']) > 23, 3, df['DataNascita'])

#Only bachelor
l = df.loc[df['TipoCorso'] == 'L'].reset_index()
#Only unique cycle
lmcu = df.loc[df['TipoCorso'] == 'LMCU'].reset_index()

#Dynamic data
df1 = pd.read_excel('../dataset/data_dropout_exams.xlsx')
df1 = df1.dropna().reset_index(drop=True)

#Replace all the exams of a single student with the mean, sum for cfus (the mean would be incorrect as the information about the number of exams taken will be lost)
#df1 = df1.groupby('ID_Studente').agg({'Voto_se_numerico':mean, 'CFUsuperati': sum}).reset_index()

#Merge static data with dynamic one, all problems
all = pd.merge(df, df1, on='ID_Studente')
#l = pd.merge(l, df1, on='ID_Studente')
#lm = pd.merge(lm, df1, on='ID_Studente')
#lmcu = pd.merge(lmcu, df1, on='ID_Studente')

#Get the year when exam was taken
all['Giorno_esame'] = all['Giorno_esame'].dt.year

#Keep only exams from the year of subscription
all = all.loc[all['Coorte'] != all['Giorno_esame']]
group = all.groupby('ID_Studente').agg({'Voto_se_numerico':mean, 'CFUsuperati':sum})
all = pd.merge(group, all, on='ID_Studente')
all = all.drop('Voto_se_numerico_y', axis=1)
all = all.drop('CFUsuperati_y', axis=1)
all = all.drop_duplicates(subset=['ID_Studente'])

#Datasets for other problems formulations
#Only master
lm = all.loc[all['TipoCorso'] == 'LM'].reset_index()
#3 range of ages at the start of university, for master 1 (lt 22), 2 (bt 22,25), 3(gt 25)
lm['DataNascita'] = np.where((lm['Coorte']-lm['DataNascita']) < 22, 1, lm['DataNascita'])
lm['DataNascita'] = np.where((lm['Coorte']-lm['DataNascita']).between(22,25), 2, lm['DataNascita'])
lm['DataNascita'] = np.where((lm['Coorte']-lm['DataNascita']) > 25, 3, lm['DataNascita'])

#3 range of ages at the start of university, 1 (lt 20), 2 (bt 20,23), 3(gt 23)
all['DataNascita'] = np.where((all['Coorte']-all['DataNascita']) < 20, 1, all['DataNascita'])
all['DataNascita'] = np.where((all['Coorte']-all['DataNascita']).between(20,23), 2, all['DataNascita'])
all['DataNascita'] = np.where((all['Coorte']-all['DataNascita']) > 23, 3, all['DataNascita'])

#Only bachelor
l = all.loc[all['TipoCorso'] == 'L'].reset_index()
#Only unique cycle
lmcu = all.loc[all['TipoCorso'] == 'LMCU'].reset_index()

features = ['Genere','Diploma_scuola_superiore','DataNascita',
            'OFA','voto_scuola_superiore',
            'Ambito','CFUsuperati_x','Voto_se_numerico_x']

#Dataset, all degrees
X = all[features]
y = all['Abbandoni']

#Dataset, only bachelor
Xl = l[features]
yl = l['Abbandoni']

#Dataset, only master
Xlm = lm[features]
ylm = lm['Abbandoni']

#Dataset, only unique cycle
Xlmcu = lmcu[features]
ylmcu = lmcu['Abbandoni']

#Cycle on random seeds
for i in range(1):
    print('#########################################################\n Seed ' + str(i) + '\n')
    #Train-test splits
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=i)

    X_trainl, X_testl, y_trainl, y_testl = train_test_split(Xl,yl,test_size=0.2, random_state=i)

    X_trainlm, X_testlm, y_trainlm, y_testlm = train_test_split(Xlm,ylm,test_size=0.2, random_state=i)

    X_trainlmcu, X_testlmcu, y_trainlmcu, y_testlmcu = train_test_split(Xl,yl,test_size=0.2, random_state=i)

    #Balance dataset, undersampling
    under_sampler = RandomUnderSampler(random_state=i)
    X_res, y_res = under_sampler.fit_resample(X_train,y_train)

    print('\n---All degrees---\n')
    #KNN model 
    model = KNeighborsClassifier()
    model.fit(X_res,y_res)
    y_pred = model.predict(X_test)
    print('-KNN:\n')
    print(classification_report(y_test, y_pred) + '\n')

    #DT model
    model = DecisionTreeClassifier()
    model.fit(X_res,y_res)
    y_pred = model.predict(X_test)
    print("-DT:\n")
    print(classification_report(y_test, y_pred) + '\n')

    #RF model
    model = RandomForestClassifier()
    model.fit(X_res,y_res)
    y_pred = model.predict(X_test)
    print("-RF:\n")
    print(classification_report(y_test, y_pred) + '\n')

    #Balance dataset, undersampling
    under_sampler = RandomUnderSampler(random_state=i)
    X_res, y_res = under_sampler.fit_resample(X_trainl,y_trainl)

    print('\n---Only bachelor---\n')
    #KNN model 
    model = KNeighborsClassifier()
    model.fit(X_res,y_res)
    y_pred = model.predict(X_test)
    print('-KNN:\n')
    print(classification_report(y_test, y_pred))

    #DT model
    model = DecisionTreeClassifier()
    model.fit(X_res,y_res)
    y_pred = model.predict(X_test)
    print("-DT:\n")
    print(classification_report(y_test, y_pred) + '\n')

    #RF model
    model = RandomForestClassifier()
    model.fit(X_res,y_res)
    y_pred = model.predict(X_test)
    print("-RF:\n")
    print(classification_report(y_test, y_pred) + '\n')

    #Balance dataset, undersampling
    under_sampler = RandomUnderSampler(random_state=i)
    X_res, y_res = under_sampler.fit_resample(X_trainlm,y_trainlm)

    print('\n---Only master---\n')
    #KNN model 
    model = KNeighborsClassifier()
    model.fit(X_res,y_res)
    y_pred = model.predict(X_test)
    print('-KNN:\n')
    print(classification_report(y_test, y_pred))

    #DT model
    model = DecisionTreeClassifier()
    model.fit(X_res,y_res)
    y_pred = model.predict(X_test)
    print("-DT:\n")
    print(classification_report(y_test, y_pred) + '\n')

    #RF model
    model = RandomForestClassifier()
    model.fit(X_res,y_res)
    y_pred = model.predict(X_test)
    print("-RF:\n")
    print(classification_report(y_test, y_pred) + '\n')

    #Balance dataset, undersampling
    under_sampler = RandomUnderSampler(random_state=i)
    X_res, y_res = under_sampler.fit_resample(X_trainlmcu,y_trainlmcu)

    print('\n---Only unique cycle---\n')
    #KNN model 
    model = KNeighborsClassifier()
    model.fit(X_res,y_res)
    y_pred = model.predict(X_test)
    print('-KNN:\n')
    print(classification_report(y_test, y_pred))

    #DT model
    model = DecisionTreeClassifier()
    model.fit(X_res,y_res)
    y_pred = model.predict(X_test)
    print("-DT:\n")
    print(classification_report(y_test, y_pred) + '\n')

    #RF model
    model = RandomForestClassifier()
    model.fit(X_res,y_res)
    y_pred = model.predict(X_test)
    print("-RF:\n")
    print(classification_report(y_test, y_pred) + '\n')
