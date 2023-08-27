"""
Created on Sun Apr 9 2023

@author: nsingla
"""

import numpy as np
import util.util as util, plotly.express as px
import plotly.graph_objects as go



#dcmData, dcmPixelData = util.data_loader("Cleaned_image")

## Clinical Data Analysis

#Distribution between male vs female
def genderDistribution(df, name):
    values = df["PatientSex"].value_counts()
    fig = go.Figure(data=[go.Pie(labels = ['M', 'F'], values = values, hole=0.3)])
    filePath = "EDA_Result/" + name + ".png"
    fig.write_image(filePath)



#Age boxplot
def agedistribution(df, name):
    #fig = px.histogram(df, x="PatientAge")
    fig = px.box(df, 
                 y="PatientAge", 
                 points="all",
                 title="Age distribution of patients")
    fig.update_layout(title_x=0.5)
    filePath = "EDA_Result/" + name + ".png"
    fig.write_image(filePath)

## Data Visualization: 

'''
Data visualization techniques, such as histograms, box plots, scatter plots, and 
heat maps, can be used to visualize the distribution of pixel values, 
anatomical structures, and image features in DI
'''

## TO DO

## Statistical Analysis
'''
Various statistical techniques can be applied to DICOM data, 
such as mean, median, standard deviation, variance, and correlation analysis, 
to understand the relationships between different variables and identify any patterns or anomalies
'''

#Mean

def meanImage(dcmPixelData, name="Mean_image"):
    meanImage = np.zeros(dcmPixelData[0].shape, np.float64)
    for im in dcmPixelData:
        meanImage = meanImage + im/(len(dcmPixelData))

    meanImage = np.array(np.round(meanImage), dtype = np.uint8)
    util.saveImage(meanImage, name)

#Variancemat

'''
Variance Map: 
A variance map is a grayscale image that displays the variance of pixel values 
for each pixel location across all images in the dataset. 
The variance values are usually normalized to the range [0, 255] for display purposes. 
Darker areas on the map correspond to regions of low variance, 
while brighter areas correspond to regions of high variance.
'''
def varianceMap(dcmPixelData, name="Variance_map"):
    variance = np.var(np.array(dcmPixelData), axis=0)
    util.saveImage(variance, name)

#Select a random image from dataset
#randImageInd = np.random.choice(range(270))
#Overlay the variance map on one of the images from dataset. 
#util.display_gradcam(dcmPixelData[randImageInd], variance)


#Corelation Analysis - Application out of memory error - DO NOT RUN
#data = np.array(dcmPixelData)
#cov_matrix = np.cov(data.reshape(-1, data.shape[0]))
#util.saveImage(cov_matrix, "Covariance_matrix")

#Variance by components in image



#Distribution of the pixels


##Dimensionality Reduction
'''
Dimensionality reduction techniques, such as principal component analysis (PCA), 
can be used to reduce the complexity of DICOM data by transforming high-dimensional data 
into lower-dimensional representations while preserving the most important features.
'''
#PCAs




#TO DO

## Segmentation: 
'''
Segmentation is the process of separating an image into regions or segments based on pixel intensity values, 
anatomical structures, or other image features. DICOM data can be segmented to extract important features, 
such as tumors, blood vessels, or other abnormalities.
'''

## TO DO


# a few ideas that I can run with is 

# run a k-means clustering to see what sort of clusters do you get
    # Can you get some clusters on the basis of fat and all. 



'''
https://datascience.stackexchange.com/questions/29223/exploratory-data-analysis-with-image-datset
https://towardsdatascience.com/exploratory-data-analysis-ideas-for-image-classification-d3fc6bbfb2d2
https://www.kaggle.com/code/manabendrarout/md-with-computer-vision-eda-and-domain-knowledge

Image/ML specific stuff

Things you can do with images:

Compute the mean image
Mean image by class
Eigenfaces (or rather "Eigenimages")
Fisher-Faces
You can compute the correlation of pixels
'''

'''
pylab.imshow(im, cmap=pylab.cm.gist_gray)
pylab.axis('off')
pylab.savefig("test.png")
'''

##Runcommand 
# python3 exploratory_data_analysis.py