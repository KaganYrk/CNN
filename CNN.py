import tensorflow as tf
import sys
from keras import callbacks
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os
import pandas
import numpy 
import matplotlib.pyplot as plt
import cv2
import numpy as np  
import pandas as pd
from skimage import transform
from keras.models import load_model
import imageio
from keras.callbacks import ModelCheckpoint
import time




TRAIN_DIR = "data/train"
VALIDATION_DIR = "data/validation"
TEST_DIR = "data/test"
IMAGE_SIZE=224
BATCH_SIZE=64


def get_filelist(myDir, format='.jpg'):
    fileList = []
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList



def get_images_and_labels():
    species=[]
    source=[]
    path=os.getcwd()
    lab= get_filelist("%s\dataset2\images\lab"% path)
    field= get_filelist("%s\dataset2\images\ield"% path)
    imagepath=lab+field

    for i in range(len(imagepath)):
        speciespath=os.path.dirname(imagepath[i])
        words= speciespath.split("\\")
        species.append(words[12])
        source.append(words[11])
    return imagepath,species,source


df=pd.DataFrame()

df["path"],df["species"],df["source"]=get_images_and_labels()
training_set = df.groupby("species").sample(frac = 0.8)
validation_set = df.drop(training_set.index).groupby("species").sample(frac = 0.5)
test_set = df.drop(training_set.index).drop(validation_set.index)

#Manuel olarak one hot encoding
# =============================================================================
# x=pd.DataFrame(df["path"])
# y=pd.DataFrame(df["species"])
# y_numeric = np.asarray(y.drop_duplicates())
# #One hot encoding labels
# u_one_hot = keras.utils.to_categorical(y, num_classes=184)
# y_one_hot= pd.get_dummies(df.species.drop_duplicates().reset_index().drop(columns=["index"]), prefix='species')
# =============================================================================

       

def resize_padded(img, new_shape, fill_cval=None, order=1):
    import numpy as np
    fill_cval = img[1,1] # np.min(img)
    if img.shape[0] > img.shape[1]:
        ratio = img.shape[0] / new_shape[0]
    else:
        ratio = img.shape[1] / new_shape[1]
    img2 = cv2.resize(img, (0,0), fx = 1/ratio, fy = 1/ratio)
    new_img = np.empty(new_shape)
    new_img.fill(fill_cval)
    new_img[0:img2.shape[0],0:img2.shape[1]] = img2        
    return(new_img)



def convert_resize_and_export(dataset,directory):
    if not os.path.exists(directory): os.makedirs(directory)
    for index,item in dataset.iterrows():
        imagepath = item["path"]
        th1 = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
        
        # crop color bars in 'lab' images
        if item['source'] == 'lab':
            h=th1.shape[1]-160
            w=th1.shape[0]-100
            time.sleep(0.1)
            plt.pause(0.0001)
            th1 = th1[0:w, 0:h]
            
        name = os.path.basename(imagepath)
        outfn = "%s/%s/%s" % (directory,item["species"],name)
        outfn2 = "%s/%s" % (directory, item["species"])
        if not os.path.exists(outfn2): os.makedirs(outfn2)
        img3 = resize_padded(th1, (IMAGE_SIZE,IMAGE_SIZE))
        img4 = (img3 / np.max(img3) * 255).astype(int)
        cv2.imwrite(outfn, img4)

convert_resize_and_export(training_set,TRAIN_DIR)
convert_resize_and_export(validation_set,VALIDATION_DIR)
convert_resize_and_export(test_set,TEST_DIR)


trainingdata = ImageDataGenerator(
   rescale = 1./255,
    rotation_range=90
)

validationdata = ImageDataGenerator(
           rescale = 1./255,
)

testdata = ImageDataGenerator(
        rescale=1./255
        )

train_generator = trainingdata.flow_from_directory(
        TRAIN_DIR,  
        target_size=(IMAGE_SIZE, IMAGE_SIZE), 
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        interpolation="lanczos"
        )

#Data Augmentation sonrası oluşan çıktıları incelemek için
# =============================================================================
# 
# i = 0
# for batch in trainingdata.flow_from_directory(   TRAIN_DIR,  
#         target_size=(IMAGE_SIZE, IMAGE_SIZE), 
#         batch_size=BATCH_SIZE,
#         color_mode='grayscale',
#         class_mode='categorical',
#         interpolation="lanczos",
#     save_to_dir="preview", save_prefix='hi'):
# 
#     i += 1
#     if i > 20:
#         break  
# =============================================================================
        

validation_generator = validationdata.flow_from_directory(
        VALIDATION_DIR,  
        target_size=(IMAGE_SIZE, IMAGE_SIZE), 
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical'
        )


test_generator = testdata.flow_from_directory(
        TEST_DIR,  
        target_size=(IMAGE_SIZE, IMAGE_SIZE),  
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical'
        )

#Modelin oluşturulup eğitildiği ve eğitim grafiğinin çizildiği bölüm.
# =============================================================================
# num_classes = len(train_generator.class_indices)
# 
# model = Sequential()
# 
# #MODEL MİMARİSİ 
# 
# model.add(Dense(158, activation='softmax'))
# model.compile(optimizer=keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=False),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# best_model_file = MODEL_FILE
# best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)
# history=model.fit(train_generator,
#            steps_per_epoch =len(train_generator),
#            batch_size=64,
#       verbose=1,
#         epochs =150,
#       validation_data=validation_generator,
#      validation_steps=len(validation_generator),
#      callbacks =[best_model]
#       )
# model.save("model.h5")
# 
# #  "Accuracy"
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# # "Loss"
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# 
# =============================================================================

model = load_model("model.h5")

#Modelin test edilmesi
# =============================================================================
# c=model.evaluate(
#   test_generator,
# )
# 
# 
# =============================================================================

#Modelin manuel olarak ve türlere göre test edilmesi
species_list=df[['species']].drop_duplicates(subset=['species']).sort_values(by=['species']).reset_index().drop(columns=["index"])
prediction=[]
gr=[]
percent=2645
değertablosu=species_list

değertablosu["tam"]=0
değertablosu["doğru"]=0
for step in range(2645):
    X, y = test_generator.next()
    tür=(species_list.loc[np.argmax(y[0])]).to_string()
    ayrıtür=tür.split()
    gr.append(ayrıtür[1])
    tahmin=model.predict(X)
    tahminedilentür=(species_list.loc[np.argmax(tahmin[0])]).to_string()
    ayrıtahminedilentür=tahminedilentür.split()
    prediction.append(ayrıtahminedilentür[1]) 
    if(tür!=tahminedilentür):
       percent= percent-1
       
       
       değertablosu.loc[değertablosu.species == ayrıtür[1],"tam"]=( değertablosu.loc[değertablosu.species == ayrıtür[1],"tam"]+1)
    else:
       değertablosu.loc[değertablosu.species == ayrıtür[1],"doğru"]= değertablosu.loc[değertablosu.species == ayrıtür[1],"doğru"]+1
       değertablosu.loc[değertablosu.species == ayrıtür[1],"tam"]=( değertablosu.loc[değertablosu.species == ayrıtür[1],"tam"]+1)

değertablosu["sonuç"]=0
i=0
for step in range(158):
    
    doğrutahmin=(değertablosu.loc[i,"doğru"])
    toplamgirdi=(değertablosu.loc[i,"tam"])
    değertablosu.loc[i,"yüzde"]=(doğrutahmin/toplamgirdi)*100
    i=i+1
result=(percent/2645)


    
    
# Tek fotoğrafı test etmek için
# =============================================================================
# print("--Single test--")
# species_list=df[['species']].drop_duplicates(subset=['species']).sort_values(by=['species']).reset_index().drop(columns=["index"])
# path=PATH
# image = cv2.imread(path,0)
# 
# plt.imshow(image)
# 
# plt.show()
# imageslast = np.array(image)
# img=[]
# img.append(imageio.imread(image))
# imageslast = [transform.resize(image, (224,224)) for image in img] 
# imageslast = np.array(imageslast)
# 
# imageslast = [transform.resize(imageslast, (224,224))]
# imageslast = np.array(imageslast)
# imageslast = imageslast.reshape(*imageslast.shape, 1)   
# singletest= model.predict(imageslast)
# gr=species_list.loc[np.argmax(singletest[0])]
# 
# singletest[0][np.argmax(singletest[0])]=0
# gr2=species_list.loc[np.argmax(singletest[0])]
# 
# singletest[0][np.argmax(singletest[0])]=0
# gr3=species_list.loc[np.argmax(singletest[0])]
# 
# singletest[0][np.argmax(singletest[0])]=0
# gr4=species_list.loc[np.argmax(singletest[0])]
# 
# singletest[0][np.argmax(singletest[0])]=0
# gr5=species_list.loc[np.argmax(singletest[0])]
# print(np.argmax(singletest[0]))
# print(gr)
# print(gr2)
# print(gr3)
# print(gr4)
# print(gr5)
# 
# =============================================================================


