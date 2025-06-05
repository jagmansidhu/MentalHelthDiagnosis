# MentalHelthAanalyzer

---

## ⚠️ Important Disclaimer ⚠️

This project implements a Natural Language Processing (NLP) model for text classification into mental health-related categories. **It is developed purely for educational and experimental purposes and should NOT be used for any form of medical diagnosis, advice, or treatment.**

**The model's predictions are based on patterns learned from a small, limited, and potentially biased dataset. Its outputs are NOT professional medical diagnoses and may be inaccurate or misleading. Always consult with a qualified mental health professional for any concerns about your mental health or any medical condition.**

---

## Project Description

This project demonstrates how to fine-tune a specialized BERT-based language model (https://huggingface.co/mental/mental-bert-base-uncased) for multi-class text classification related to mental health. The model is trained to classify input text into one of seven categories: **Anxiety, Bipolar, Stress, Depression, Normal, Personality Disorder, and Suicidal**.

It provides a command-line interface for interactive classification, allowing users to input text and receive a predicted category along with a confidence score.

## Features

* **Custom Fine-tuning:** Adapts a pre-trained (https://huggingface.co/mental/mental-bert-base-uncased) model to specific mental health categories.
* **Multi-Class Classification:** Classifies text into one of 7 distinct mental health-related labels.
* **Interactive Interface:** A command-line tool to quickly test the model with custom text inputs.
* **Leverages Hugging Face Transformers:** Built on the powerful `transformers` library for easy model handling and training.
* **GPU/MPS Acceleration:** Utilizes NVIDIA GPUs or Apple Silicon's MPS for faster training and inference (if available).

## Setup and Installation

To get this project running on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[jagmansidhu]/[MentalHelthDiagnosis].git
    cd [MentalHelthDiagnosis]
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
3. **Hugging Face Access Token:**
    The https://huggingface.co/mental/mental-bert-base-uncased is a gated model on Hugging Face. You'll need to:
    * Log in to [Hugging Face Hub](https://huggingface.co/login).
    * Navigate to the [mental/mental-bert-base-uncased](https://huggingface.co/mental/mental-bert-base-uncased) model page and **request access**.
    * Once granted, go to your [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens) and generate a **"Read"** role token.
    * Set this token as an environment variable (recommended) before running your scripts:
        ```bash
        export HF_TOKEN="hf_YOUR_COPIED_TOKEN"
        ```
        (Replace `hf_YOUR_COPIED_TOKEN` with your actual token. On Windows, use `set HF_TOKEN="hf_YOUR_COPIED_TOKEN"`).

## Usage

### 1. Data Preparation

* Ensure you have your labeled dataset. It should be a CSV file (e.g., `mental_health_data.csv`) with at least two columns:
    * One for text content (e.g., `text` or `statement`).
    * One for your mental health categories (e.g., `category` or `status`).
* The categories in your dataset's label column *must* exactly match one of these 7 labels: `anxiety`, `bipolar`, `stress`, `depression`, `normal`, `personality disorder`, `suicidal`.

### 2. Fine-tuning the Model

The fine-tuning script (`TrainModel.ipynb` - assuming this is what you named your training script) will train the `mental/mental-bert-base-uncased` model on your dataset and save the fine-tuned version locally.

* **Place your CSV:** Put your `mental_health_data.csv` (or whatever you named it) in the correct path relative to your script, as specified in the `DATA_CSV_PATH` variable within the script.
* **Run the fine-tuning script:**
    ```bash
    python fine_tune_model.py
    ```
    This script will:
    * Load your data.
    * Randomly sample 150 data points for training (as configured in the script).
    * Split the data into training and evaluation sets.
    * Train the model for a few epochs.
    * Save the fine-tuned model and tokenizer into the `./my_mental_health_classifier` directory.

### 3. Using the Interactive Classifier

Once the model is fine-tuned and saved, you can use the interactive script (`iteractic.py`) to test it.

* **Run the interactive script:**
    ```bash
    python interactive_diagnosis.py
    ```

Open `index.html` and play with the bot on web!

## Data

The model was fine-tuned on a custom dataset, which is expected to be a CSV file with text and corresponding mental health labels. **Please note that the example code uses a very small subset (150 data points) for demonstration purposes.** For robust performance, a much larger, diverse, and carefully annotated dataset is required.

## Limitations

* **Extremely Limited Data:** The model is trained on a tiny dataset (150 samples). This severely impacts its ability to generalize, leading to low accuracy and unreliable predictions.
* **Bias:** The model's performance and biases are entirely dependent on the quality, diversity, and representativeness of the training data.
* **Non-Clinical Use Only:** This model is not a diagnostic tool. Mental health is complex, and AI models cannot replace professional judgment.

## Contributing

Contributions are welcome! If you have suggestions for improvements, feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'feat: Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature`).
6.  Open a Pull Request.