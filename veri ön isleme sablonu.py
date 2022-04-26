import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split #eğitim ve test verilerini ayırmak için
from sklearn.preprocessing import StandardScaler #oznitelik olcekleme icin

#veri yukleme
veriler= pd.read_csv("eksikveriler.csv")

#eksik veri
imputer = SimpleImputer(missing_values= np.nan, strategy="mean")
yas= veriler.iloc[:,1:4].values   
imputer= imputer.fit(yas[:,1:4])
yas[:,1:4]= imputer.transform(yas[:,1:4])

#kategorik veriyi sayısala cevirme
ulke= veriler.iloc[:,0:1].values  
le= preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
ohe= preprocessing.OneHotEncoder()
ulke= ohe.fit_transform(ulke).toarray()

#dataframe oluşturma ve birleştirme
parca1= pd.DataFrame(data=ulke, index=range(22), columns=["fr","tr","us"])
parca2= pd.DataFrame(data=yas, index= range(22), columns= ["boy", "kilo", "yas"])
cinsiyet= veriler.iloc[:,-1].values
parca3= pd.DataFrame(data=cinsiyet, index= range(22), columns= ["cinsiyet"])
birlestir12= pd.concat([parca1,parca2], axis=1)
birlestir123=pd.concat([birlestir12,parca3], axis=1)
print(birlestir123)

#train ve test ayırma
x_train, x_test, y_train, y_test= train_test_split(birlestir12, parca3, test_size=0.33, random_state=0)

"""OZNİTELİK OLCEKLEME"""
sc= StandardScaler()
#x_train ve x_test teki verileri birbiriyle uyumladık. X_train ve X_test e atadık. variable explorerdan bak
X_train= sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)