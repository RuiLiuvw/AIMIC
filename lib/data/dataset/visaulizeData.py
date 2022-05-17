from glob import glob

import numpy as np
import pandas as pd
from math import ceil

import imageio
# import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


# Show one sample for each class
def drawSamplebyClass(DatasetPath, ClassNames, Mode=1):
    NumClass = len(ClassNames)
    ImgPath = []
    if Mode == 1:
        '''
        root/split1/dog_1.png
        root/split2/cat_1.png
        '''
        for ClassName in ClassNames:
            ImgPath.append(DatasetPath + '/split1/' + ClassName + '_1.png')
    elif Mode == 2:
        '''
        root/dog/xxx.png
        root/cat/123.png
        '''
        for ClassName in ClassNames:
            ImgPath.append(glob(DatasetPath + '/' + ClassName + '/*')[0])
    
    for Pos, SamplePath in enumerate(ImgPath):
            
        Img = imageio.imread(SamplePath)

        plt.figure("Figure 1")
        plt.subplot(2, ceil(NumClass / 2), Pos + 1)
        plt.suptitle("Examples of the dataset",size = 16)
        plt.imshow(Img)
        plt.axis('off')
        plt.title(ClassNames[Pos], loc='center', size=14)

    plt.tight_layout()
    plt.show()


# drawing five samples from each class to see what the images look like in the 450x600 size
def drawSample(metadata, HomePath = "./HAM10000"):
    label = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']
    label_images = []
    names = ['benign keratosis-like lesions', 'melanocytic nevi', 'dermatofibroma', 'melanoma',
            'vascular lesions','basal cell carcinoma','actinic keratoses']

    fig = plt.figure(figsize=(20, 20))
    k = range(7)

    for i in label:
        
        sample = metadata[metadata['dx'] == i]['image_id'][:5]
        label_images.extend(sample)
        

    for position,ID in enumerate(label_images):
            
        im_sample = HomePath + '/images/' + str(ID) + '.jpg'
        im_sample = imageio.imread(im_sample)

        plt.subplot(7,5,position+1)
        plt.imshow(im_sample)
        plt.axis('off')

        if position%5 == 0:
            title = int(position/5)
            plt.title(names[title], loc='left', size=20)

    plt.tight_layout()
    plt.show()

    fig.savefig('large_sample.png')


# Getting a sense of what the distribution of each column looks like
def visaulizeData(HomePath, metadata, SetName, DestDataPath, Mode):
    if Mode == 0:
        print('Without visulizing')
        return
        
    elif Mode == 1:
        
        fig = plt.figure(figsize=(15,10))

        ax1 = fig.add_subplot(221)
        
        # temp = metadata['gender'].value_counts()
        labels = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc', 'scc']
        men_means = np.zeros([1, len(labels)])
        for idx, ClassName in enumerate(labels):
            men_means[0][idx] = np.sum(metadata[ClassName])
        
        x = np.arange(len(labels))
        width = 0.35  # the width of the bars
        ax1.bar(x, men_means[0], width)
        
        ax1.set_ylabel('Count')
        ax1.set_title('Cell Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        
        metadata2 = pd.read_csv(HomePath + '/ISIC_2019_Training_Metadata.csv')
        ax2 = fig.add_subplot(222)
        metadata2['sex'].value_counts().plot(kind='bar', ax=ax2)
        ax2.set_ylabel('Count', size=15)
        ax2.set_title('Sex')

        ax3 = fig.add_subplot(223)
        metadata2['anatom_site_general'].value_counts().plot(kind='bar')
        ax3.set_ylabel('Count',size=12)
        ax3.set_title('Localization')


        ax4 = fig.add_subplot(224)
        sample_age = metadata2[pd.notnull(metadata2['age_approx'])]
        sns.distplot(sample_age['age_approx'], fit=stats.norm, color='red')
        ax4.set_title('Age')

        plt.tight_layout()
        plt.show()
        
    elif Mode == 2:
               
        fig = plt.figure(figsize=(15,10))

        ax1 = fig.add_subplot(221)
        metadata['dx'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_ylabel('Count')
        ax1.set_title('Cell Type')
        
        ax2 = fig.add_subplot(222)
        metadata['sex'].value_counts().plot(kind='bar', ax=ax2)
        ax2.set_ylabel('Count', size=15)
        ax2.set_title('Sex')

        ax3 = fig.add_subplot(223)
        metadata['localization'].value_counts().plot(kind='bar')
        ax3.set_ylabel('Count',size=12)
        ax3.set_title('Localization')


        ax4 = fig.add_subplot(224)
        sample_age = metadata[pd.notnull(metadata['age'])]
        sns.distplot(sample_age['age'], fit=stats.norm, color='red')
        ax4.set_title('Age')

        plt.tight_layout()
        plt.show()

    elif Mode == 3:
        fig = plt.figure(figsize=(15,10))

        ax1 = fig.add_subplot(221)
        metadata['diagnostic'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_ylabel('Count')
        ax1.set_title('Cell Type')

        ax2 = fig.add_subplot(222)
        temp = metadata['gender'].value_counts()
        Unk = metadata.shape[0] - (temp[0] + temp[1])
        labels = ['male', 'female', 'unkown']
        men_means = [temp[0], temp[1], Unk]
        x = np.arange(len(labels))
        width = 0.35  # the width of the bars
        ax2.bar(x, men_means, width)
        ax2.set_ylabel('Count', size=15)
        ax2.set_title('sex')

        ax3 = fig.add_subplot(223)
        metadata['region'].value_counts().plot(kind='bar')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax3.set_ylabel('Count',size=12)
        ax3.set_title('Localization')


        ax4 = fig.add_subplot(224)
        sample_age = metadata[pd.notnull(metadata['age'])]
        sns.distplot(sample_age['age'], fit=stats.norm, color='red')
        ax4.set_title('Age')

        plt.tight_layout()
        plt.show()
    
    fig.savefig(DestDataPath + '/' + SetName + '_dataDistr.jpg')
    return