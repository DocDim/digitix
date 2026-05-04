# 🔢 Digitix: Handwritten Digit Recognition System

Digitix is a full-stack machine learning application that identifies handwritten digits (0–9) using a Convolutional Neural Network (CNN). The project features a robust image-processing pipeline and an interactive web interface built with Streamlit.

## 🚀 Live Preview
*(Optional: Add a link here if you deploy to Streamlit Cloud, or include a screenshot of your GUI from `app.py` here)*

## 🛠️ Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** OpenCV / Pillow
- **Web Interface:** Streamlit
- **Data:** MNIST Dataset (70,000 grayscale images)

## 🧠 Model Architecture
The system utilizes a CNN architecture, which is superior to standard feed-forward networks for image tasks because it preserves spatial hierarchies.

| Layer | Type | Configuration | Purpose |
| :--- | :--- | :--- | :--- |
| 1 | **Conv2D** | 32 filters, 3x3 kernel | Detects basic edges and curves |
| 2 | **MaxPooling** | 2x2 pool size | Reduces dimensionality |
| 3 | **Conv2D** | 64 filters, 3x3 kernel | Detects complex patterns/shapes |
| 4 | **Dropout** | 0.2 rate | Prevents overfitting |
| 5 | **Dense** | 128 units, ReLU | Learned classification logic |
| 6 | **Output** | 10 units, Softmax | Probability distribution (0-9) |

**Performance:** Achieved **~99% accuracy** on the MNIST test set.

## ⚙️ Features
- **Smart Preprocessing:** Automatically handles user uploads by converting to grayscale, resizing to 28x28, and inverting colors to match training data (white-on-black).
- **Real-time Prediction:** Instant classification with a confidence score.
- **Probability Distribution:** Visualizes how the model "thinks" across all possible digits using a bar chart.
- **Model Transparency:** Displays the processed image as seen by the CNN.

## 📂 Project Structure
```text
digitix/
├── app.py                # Streamlit Web GUI
├── requirements.txt      # Project dependencies
├── models/               # Saved CNN model (.h5)
├── src/
│   ├── model_training.py # CNN training script
│   └── inference.py      # Image processing pipeline
└── README.md             # Project documentation
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)DocDim/digitix.git
   cd digitix
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model:**
   ```bash
   python src/model_training.py
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 👤 Author
- **Tchifou M. Dieffi** - https://www.linkedin.com/in/dieffi-m-tchifou-3608649/
- Graduate Researcher at **Hood College**

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
