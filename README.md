# action-recognition-pipeline

This project is an action recognition pipeline that uses OpenCV and TensorFlow to detect anomaly behavior among the following classes: 'Normal', 'Fighting', 'Robbery', 'Shoplifting', 'Stealing'.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with this project, you need to have Python installed on your system. Then, follow the steps below to set up the environment:

1. Clone the repository:
    ```bash
    git clone
    cd action-recognition-pipeline
    ```

2. Create a virtual environment:
    ```bash
    python -m venv env
    ```

3. Activate the virtual environment:

    On Windows:
    ```bash
    .\env\Scripts\activate
    ```

    On macOS/Linux:
    ```bash
    source env/bin/activate
    ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Download the pre-trained weights for action recognition and save them as `action_recognition_weights.h5` in the project directory.

2. Run the action recognition pipeline on a video file:
    ```bash
    python action_recognition.py --video_path path_to_your_video.mp4
    ```

3. The script will process the video, detect anomalies, and display the video with detected anomalies in real-time.

## Project Structure
    action-recognition-pipeline/
    ├──action_recognition.py # Main script for action recognition
    ├──README.md # Project README file
    ├──requirements.txt # Python packages required for the project
    └──action_recognition_weights.h5 # Pre-trained model weights

## Model Training

If you want to train the model from scratch, you need a dataset containing videos labeled with the respective classes. The dataset should be organized in directories named after the classes.

1. Prepare your dataset with the following structure:
    ```
    dataset/
    ├── Normal/
    ├── Fighting/
    ├── Robbery/
    ├── Shoplifting/
    └── Stealing/
    ```

2. Use a script to train the model on your dataset. The training script should perform data preprocessing, model training, and save the trained model weights.

3. Save the trained model weights as `action_recognition_weights.h5`.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.
