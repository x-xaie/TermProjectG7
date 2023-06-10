**NeuroSense**


NeuroSense is a user-friendly web application built with Python, Streamlit, and Tensorflow. It uses the power of Convolutional Neural Networks (CNNs) to classify a brain MRI into one of four classes: No tumor, Pituitary tumor, Meningioma tumor, or Glioma tumor.

**Features**
  *Upload an MRI image and the app will classify it into one of the four categories.
  *Uses an EfficientNetB0 model trained on a large dataset of brain MRIs for accurate classification.
  *Prerequisites
  *Python 3.7 or higher
  *Streamlit
  *Tensorflow
  *OpenCV
  *PIL
  *Numpy
  *Matplotlib

**Dependencies**
pip install -r requirements.txt

**Run**
streamlit run app.py


**Problem Statement:
**The problem is to classify brain tumor images into different classes using machine learning and deep learning. The aim is to create an accurate and reliable model to aid in the diagnosis of brain tumors.

**Method**:
This method is optimized for a pre-trained "microsoft/swin-tiny-patch4-window7-224" sample of the brain tumor image dataset. The model has been trained on large datasets with good results before. Using transfer learning, knowledge of a previous learning model can be transferred to the specific task of classifying brain tumors.
Motivation:
The motivation behind this project is to solve problems in brain tumor diagnosis and provide tools that can help doctors classify brain tumors accurately. Manually diagnosing brain diseases based on medical images can be time consuming and lead to human error. It can increase the efficiency and accuracy of brain diagnosis by creating automatic classification models.

Insufficient information provided for sample description. More details about the architecture, specific procedures and adjustments will be helpful in the renovation process.
In addition, information about the data used for training and evaluation, including classes, image resolution, and distribution data, is important for understanding the model's resources and functions.

Also details about usage pattern and restrictions would be helpful. For example, will the model be used as a stand-alone diagnostic tool or as an aid to clinicians? Are there certain limitations or possible biases to the modeling? These considerations help determine the validity and reliability of the model in real-world situations.
Regarding the training process, the hyperparameters provided some insights into the training configuration. However, a good understanding of the training method requires additional details such as the amplification process performed, the sequencing procedure, and preliminary information.

Finally, the training results are provided, including loss, accuracy, F1 score, recall, and accuracy index obtained in the benchmark. While these measurements are an indicator of the model's performance, it is useful to know the baseline performance in classifying brain tumors and compare it with other existing model or models.
