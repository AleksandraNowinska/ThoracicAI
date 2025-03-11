# ThoracicAI - Chest X-ray Disease Classification  

ThoracicAI is a deep learning project aimed at **classifying thoracic diseases from chest X-ray images**.  
This project uses **Convolutional Neural Networks (CNNs)** to assist radiologists by providing **automated, data-driven diagnoses**.  

---

## **📌 Features**
✅ **Deep Learning for Medical Imaging** – Uses a custom CNN model optimized for **feature extraction and interpretability**.  
✅ **Class Imbalance Handling** – Implements **weighted loss functions and stratified batch sampling** for better generalization.  
✅ **Performance Optimizations** – Uses **Adam optimizer, learning rate scheduling (`ReduceLROnPlateau`)**, and **dropout regularization**.  
✅ **Explainability & Ethical Considerations** – Supports **saliency maps (Grad-CAM) and ethical AI deployment strategies**.  

---

## **🛠 Project Structure**

ThoracicAI/
│── dc1/                 # Project source code
│   ├── batch_sampler.py  # Custom batch sampling strategy for handling class imbalance
│   ├── image_dataset.py  # Dataset loading and preprocessing
│   ├── net.py            # Convolutional Neural Network (CNN) model
│   ├── train_test.py     # Training and evaluation functions
│   ├── main.py           # Main script to train and test the model
│── data/                 # Folder containing preprocessed datasets
│   ├── X_train.npy       # Training images
│   ├── Y_train.npy       # Training labels
│   ├── X_test.npy        # Test images
│   ├── Y_test.npy        # Test labels
│── model_weights/        # Saved model weights from training
│── artifacts/            # Training results, plots, and logs
│── README.md             # Project documentation
│── requirements.txt      # Dependencies for running the project


---

## **📂 Dataset**
- The dataset consists of **grayscale chest X-ray images** (128×128 resolution).  
- It contains **six different classes** representing **various thoracic diseases** or the absence of disease.  
- **Preprocessing applied:**  
  - **Contrast Enhancement**: CLAHE for better feature visibility.  
  - **Data Augmentation**: Flipping and slight rotations to improve generalization.  
  - **Balanced Sampling**: Stratified batch sampling to prevent class imbalance issues.  

---

## **🚀 Installation & Setup**
### **1️⃣ Clone the Repository**
git clone https://github.com/your-username/ThoracicAI.git
cd ThoracicAI

### **2️⃣ Create Virtual Environment (Optional)**
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows

### **3️⃣ Install Dependencies**
pip install -r requirements.txt

### **4️⃣ Run Training**
python dc1/main.py --nb_epochs 10 --batch_size 25 --balanced_batches True
👉 The model will train and save weights automatically in model_weights/.


The CNN follows a 4-layer structure optimized for medical imaging:

Layer	Type	Filters	Kernel	Activation
Conv1	Convolution (2D)	64	4×4	ReLU
Conv2	Convolution (2D)	128	4×4	ReLU
Conv3	Convolution (2D)	64	4×4	ReLU
Conv4	Convolution (2D)	32	4×4	ReLU
FC1	Fully Connected (FC)	256	-	ReLU
FC2	Fully Connected (FC)	128	-	ReLU
Output	Softmax Classifier	6	-	Softmax
✔ Optimized for feature extraction and medical interpretability.

## **📈 Training Results**
The model was trained for 10 epochs, and results are logged in artifacts/.
Training and validation loss trends are stored as plots:
artifacts/session_MM_DD_HH_MM.png

## **📌 Next Steps**
🔹 Implement Grad-CAM visualization to analyze feature importance.
🔹 Evaluate the model on external datasets for generalization testing.
🔹 Explore fine-tuning with pre-trained models (ResNet, DenseNet).

## **👨‍💻 Contributors**
Aleksandra Nowinska – Lead Developer
DC1 – Research & Model Evaluation

## **📜 License**
This project is released under the MIT License.
Feel free to contribute or modify for research purposes.
