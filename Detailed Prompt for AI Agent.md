# Detailed Prompt for AI Agent: Building a Modulation Signal Classification System

## 1. Introduction & Role

You will act as a **Senior Machine Learning Engineer** specializing in signal processing and Deep Learning. Your task is to create a series of professional, well-structured, readable, and reusable Jupyter Notebooks to solve the problem of Automatic Modulation Classification (AMC).

## 2. General Standards and Requirements

### 2.1. Code Quality and Documentation

- **Language Requirement:** All explanatory text (Markdown), chart titles, axis labels, and notes must be written in **English**. Code comments and variable/function names should remain in English to adhere to programming best practices.
- **Docstrings:** All custom-defined functions must include a clear docstring explaining their purpose, parameters (arguments), and return values.

### 2.2. General Aesthetic Requirements for All Plots

- **Plotting Workflow (CRITICAL):** For every plot generated, the code must first **save the figure to a file** (`plt.savefig`) and then **display it inline** in the notebook (`plt.show()`). This ensures that a visual record is saved while maintaining an interactive notebook experience.
- **Layout & Margins:**
  - **Auto-adjust Layout:** Use `plt.tight_layout()` after plotting each figure to ensure components do not overlap or get clipped.
  - **Data Margins:** Increase axis limits (`xlim`, `ylim`) so that data points are not too close to the plot edges.
- **General Setup:**
  - **Consistent Style:** Set a global style for Matplotlib/Seaborn at the beginning of each notebook (e.g., `plt.style.use('seaborn-v0_8-whitegrid')`) for a professional and consistent look.
  - **Reasonable Figure Size:** Use a consistent and appropriate figure size (`figsize`). For plots where the horizontal axis represents time or frequency (e.g., I/Q Waveform, Amplitude, Phase, Spectrum), **use a rectangular aspect ratio (e.g., `figsize=(12, 4)`)** for clarity. For the Constellation Diagram, a square aspect ratio (e.g., `figsize=(6, 6)`) is more appropriate.
- **Labels and Titles:**
  - **Clear Titles and Labels:** **Every plot** must have a main title describing its content and clear labels for the x-axis and y-axis.
  - **Legend:** Must be placed in the best possible location ('best' or a specific location) without obscuring the data.
  - **Axis Ticks:** Labels should not be automatically rotated and must be easy to read.
- **Colors:**
  - **Colormap:** Use the `Magma` colormap for heatmaps and matrices. Provide other options like `Viridis`, `Plasma`, `Cividis` as variables in the code for easy switching.
  - **Palette for Line Plots:** For I/Q plots and other multi-class line plots, use a high-contrast, colorblind-friendly palette.
- **Details for Heatmaps/Confusion Matrices:**
  - Cells must be square (`square=True`), with white separating lines (`linewidths=0.5, linecolor='white'`), and no grid (`grid=False`).
  - Annotation font size should be legible and fit within the cell (`annot_kws={'size': 9}`).
  - The color bar (cbar) must have the same height as the heatmap.

## 3. Project Goal & Deliverables

**Goal:** To build, train, and evaluate various Deep Learning models to classify 10 types of modulation signals from the **RadioML 2016.10b** dataset. The project will be deployed in the Google Colab environment.

**Structure:** The project will consist of two main types of notebooks, designed to clearly separate the data preparation and model training workflows.

- **Deliverable 1: `01_Data_Preparation_and_Visualization.ipynb`**
  - **Purpose:** This notebook is to be run ONLY ONCE. It will perform all steps from loading, exploring, visualizing, and preprocessing the data, then save the prepared and formatted data, ready for training.
- **Deliverable 2: `02_Model_Training_and_Evaluation_[ModelName].ipynb`**
  - **Purpose:** There will be multiple notebooks following this template, each dedicated to a specific model architecture (e.g., `02_Model_Training_and_Evaluation_ResNet1D.ipynb`). These notebooks will load the preprocessed data from the previous step and focus solely on building, training, and evaluating the model.

## 4. Detailed Requirements for `01_Data_Preparation_and_Visualization.ipynb`

### 4.1. Environment Setup

- Mount Google Drive to access and store data.
- Import necessary libraries (TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Pickle, `tqdm`, `os`, etc). **Plotly** should also be imported for advanced visualization options.
- **Set Random Seed:** Set a seed for NumPy and TensorFlow (e.g., `seed = 42`) to ensure reproducibility.
- **Check for GPU Availability:** Add code to list available physical devices and confirm that a GPU is detected by TensorFlow. Print a clear message indicating whether a GPU is being used.
- **Define Paths Flexibly:**
  - **Define a single Project Root Path:** This should be the main folder for the project. All other paths will be constructed from this root.
    - `PROJECT_ROOT_PATH = '/content/drive/MyDrive/DTVT_IUH_2025/AMC_RML2016_10b/'`
  - **Define Raw Data Path:** This path is separate as the file may be stored outside the project folder.
    - `RAW_DATA_PATH = '/content/drive/MyDrive/RML2016.10b.dat'`
  - **Construct Subfolder Paths:** Use `os.path.join()` to create absolute paths for all subdirectories from the `PROJECT_ROOT_PATH`. This ensures flexibility.
    - `PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, 'data/')`
    - `VISUALIZATIONS_PATH = os.path.join(PROJECT_ROOT_PATH, 'visualizations/')`
    - `PREPROCESSING_OBJECTS_PATH = os.path.join(PROJECT_ROOT_PATH, 'preprocessing_objects/')`
  - *Check for and automatically create these directories if they do not exist.*

### 4.2. Data Loading and Exploratory Data Analysis (EDA)

- Load the `RML2016.10b.dat` file from `RAW_DATA_PATH`.
- **Convert data from dictionary to a Pandas DataFrame** for easier manipulation and exploration. The DataFrame should include columns for the signal (as an object), modulation label, and SNR.
- Display the first 5 rows (`.head()`) and general info (`.info()`) of the DataFrame.
- **Display the total number of samples** in the entire dataset.
- **Overview of Classes and SNRs:**
  - Print the list of unique modulation types present in the data.
  - Print the list of unique SNR levels present in the data, sorted in ascending order.
- **Data Distribution Analysis:**
  - Display the sample count for each modulation type (`.value_counts()`).
  - Display the sample count for each SNR level.
  - Display a summary table (e.g., a pivot table) showing the number of samples for each (modulation, SNR) pair to verify detailed balance.
  - Visualize the distribution of modulation types using a countplot to check for data balance.
- Check for any missing (null/NaN) values in the data.
- **Extract Data to NumPy:** After exploration, convert the relevant columns from the DataFrame into three separate NumPy arrays: I/Q signals (X), modulation labels (Y), and SNR levels (Z). Ensure the sample order is preserved and the link between the three arrays is maintained correctly.

### 4.3. Raw Data Visualization

- **Adherence to Standards:** All plots must adhere to the **General Aesthetic Requirements for All Plots** defined in Section 2.2.
- **Content Requirements:**
  - All images must be saved to the `VISUALIZATIONS_PATH` directory with high resolution (`dpi=300`).
  - Use only samples with **SNR = 18dB** for visualization.
- **Improving Line Plot Appearance (Choice for AI):** For time-domain plots (I/Q, Amplitude, Phase), the line graphs must be visually clear and smooth to interpret.
  - **CRITICAL NOTE:** Any technique used here is **strictly for visualization purposes**. It **must not** modify the underlying NumPy data arrays that will be preprocessed and saved for model training.
  - You can choose one of the following approaches:
    - **Option A (Geometric Smoothing with Matplotlib):** Apply a smoothing technique like cubic spline interpolation to the data points *before* passing them to Matplotlib for plotting.
    - **Option B (Geometric Smoothing with Plotly):** Use the Plotly library, which has built-in smoothing capabilities, such as the `line_shape='spline'` parameter.
    - **Option C (Visual Styling with Matplotlib):** If geometric smoothing is not desired, improve the plot's clarity by adjusting visual parameters like increasing the `linewidth` to make the signal easier to see.
  - *This choice only applies to line plots; heatmaps and other plot types should still follow the general standards.*
- **Note on Input Data Types (CRITICAL):** The raw signal for each sample is a NumPy array of shape `(2, 128)`, where the first row is the I component and the second is the Q component. To avoid errors, ensure the correct data format is used for each visualization:
  - For **I/Q Waveform** and **Constellation Diagram**, use the I and Q components directly as two separate real-valued arrays.
  - For **Amplitude** and **Phase** plots, first construct a single complex-valued signal array of shape `(128,)` using the formula `signal = I + 1j*Q` before performing calculations (e.g., `numpy.abs`, `numpy.angle`).
- **Required Plots (one plot per modulation type):**
  1. **I/Q Waveform:** Plot I (real) and Q (imaginary) signals on the same chart.
  2. **Amplitude:** Plot amplitude over time.
  3. **Wrapped Phase:** Plot the phase over time, ensuring the values are "wrapped" within the range of -π to +π. This is the standard output of functions like `numpy.angle`.
  4. **Constellation Diagram:** I-Q plot.

### 4.4. Data Splitting

- Split the dataset into `train`, `validation`, and `test` sets with a ratio of **80% : 10% : 10%**.
- **CRITICAL REQUIREMENT (Stratification):** The split must be **stratified** by both **modulation type** AND **SNR level**. Ensure that each set (train/val/test) contains the exact 80/10/10 proportion of samples for every (modulation, SNR) pair. Approximate results are not acceptable.
- **CRITICAL REQUIREMENT (Reproducibility):** When using `train_test_split` or a similar function, you **must** set the `random_state` parameter to the same value as the global seed (e.g., `random_state=42`) to ensure the split is reproducible.
- **Note:** When splitting `X`, the corresponding `Y` (modulation labels) and `Z` (SNR) arrays must also be split accordingly to maintain the link for each sample.
- After splitting, print the number of samples in each set for confirmation.

### 4.5. Data Preprocessing

1. **Label Encoding:**
   - Use `LabelEncoder` and `OneHotEncoder` (or Keras's `to_categorical`) to create one-hot encoded labels.
   - **CRITICAL:** The encoders must **ONLY** be `fit` on the `y_train` set to learn the label mapping.
   - Use the fitted encoders to `transform` `y_train`, `y_val`, and `y_test`.
2. **Signal Normalization (Z-Score):**
   - Use `sklearn.preprocessing.StandardScaler` to perform Z-score normalization on the signal data.
   - **CRITICAL:** The scaler must **ONLY** be `fit` on the `X_train` data to learn the mean and standard deviation.
   - Use the fitted scaler to `transform` all three datasets: `X_train`, `X_val`, and `X_test`. This ensures all data is scaled consistently without data leakage.

### 4.6. Reshaping and Saving Data

- From the preprocessed data, create and save different versions with various shapes to cater to different model types. Use clear filenames.
  - **1D Format (for CNN-1D, GRU, LSTM, ResNet-1D, CGDNN, CLDNN, etc.):** shape `(samples, steps, channels)` = `(samples, timesteps, feature)` = `(samples, 128, 2)`.
  - **2D Format (for CNN-2D, VGGNet-2D, ResNet-2D, etc.):** shape `(samples, height, width, channels)` = `(samples, 2, 128, 1)`.
- **Save all datasets** (`X_train`, `X_val`, `X_test` for each format, `y_train`, `y_val`, `y_test`, and `snr_train`, `snr_val`, `snr_test`) to the `PROCESSED_DATA_PATH` directory in `.npy` format.
- **Save the fitted `LabelEncoder` and `StandardScaler` objects** to the `PREPROCESSING_OBJECTS_PATH` directory in `.pkl` format for future reuse in inference.

## 5. Detailed Requirements for `02_Model_Training_and_Evaluation_[ModelName].ipynb`

### 5.1. Environment Setup

- Similar to notebook 1, mount Drive, import libraries, set random seed, and define paths using the same flexible structure (`PROJECT_ROOT_PATH` and `os.path.join`).
- **Check for GPU Availability:** Add code to list available physical devices and confirm that a GPU is detected by TensorFlow. Print a clear message indicating whether a GPU is being used.
- **Construct Model Paths:**
  - `MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT_PATH, 'models/', '[ModelName]/')`
  - `MODEL_VISUALIZATIONS_PATH = os.path.join(MODEL_SAVE_PATH, 'visualizations/')`
- *Check for and automatically create these directories if they do not exist.*

### 5.2. Load Preprocessed Data

- Load the corresponding `.npy` and `.pkl` files from the `PROCESSED_DATA_PATH` and `PREPROCESSING_OBJECTS_PATH`. **This includes the SNR files and the pre-fitted `StandardScaler` object.** Select the correct data format (1D, 2D, etc.) appropriate for the model to be built.
- Display the shapes of the loaded data for verification.

### 5.3. Build Model Architecture

- Define the model architecture (e.g., ResNet-1D, LSTM, Vertically Stacked CNN-2D, ResNet-2D, VGGNet-2D, CLDNN, CNN-LSTM).
- **Notes:**
  - Prioritize architectures of moderate complexity, not excessively deep, to avoid overfitting and reduce training time.
  - Ensure the model's input shape exactly matches the shape of the loaded data.
- Print `model.summary()`.

### 5.4. Train the Model

- **Compile the model:**
  - Optimizer: Adam, `learning_rate=0.008`.
  - Loss: `CategoricalCrossentropy`.
  - Metrics: `accuracy`.
- **Callbacks:**
  - `ModelCheckpoint`: Save the **entire model** (not just the weights) that achieves the best `val_loss`. The saved file should be in the `.keras` format. This serves as a backup.
  - `ReduceLROnPlateau`: Automatically reduce the learning rate if `val_loss` plateaus.
  - `EarlyStopping`: Stop training early if `val_loss` does not improve. **Set `restore_best_weights=True`** to ensure the model's weights from the best epoch are restored after training.
- **Training:**
  - `batch_size=1024`.
  - `epochs <= 100`.
- Use `tqdm` to display the progress bar.

### 5.5. Evaluate and Visualize Results

- **Adherence to Standards:** All plots must adhere to the **General Aesthetic Requirements for All Plots** defined in Section 2.2.
- **Reload the Best Model:** This step is now a safeguard. The model in memory should already have the best weights thanks to `restore_best_weights=True`, but loading the model from the file saved by `ModelCheckpoint` is a good practice to ensure evaluation is done on the definitively best-saved state.
- **Required Tasks:**
  1. **Training History Plot:** Plot two charts: `accuracy` vs `val_accuracy` and `loss` vs `val_loss` over epochs.
  2. **Evaluation on Test Set:** Calculate the overall `loss` and `accuracy` on the test set.
  3. **Classification Report:** Display a detailed classification report from `sklearn.metrics.classification_report`, including Precision, Recall, and F1-score for each modulation type.
  4. **Detailed Accuracy Heatmap:** Create a heatmap showing accuracy for each **modulation type** (y-axis) vs each **SNR level** (x-axis). Display values as decimals with 3 places (`.3f`).
  5. **Overall Confusion Matrix:**
     - Display two matrices side-by-side:
       - Matrix 1: Shows raw counts of correct/incorrect predictions. The color bar ranges must be fixed from **0 to the total number of test samples per class** (e.g., use `vmin=0, vmax=12000` if there are 12000 samples per class).
       - Matrix 2: Shows accuracy (normalized), with values as decimals with 3 places (`.3f`). The color bar ranges must be fixed from **0.0 to 1.0** (e.g., use `vmin=0.0, vmax=1.0`).
  6. **Comprehensive Confusion Matrix Analysis per SNR (NEW):**
     - To fully understand model performance under different noise conditions, you must generate confusion matrices for **every single** unique SNR level present in the test set.
     - **Loop through all unique SNR values** without exception.
     - For **each and every SNR level**, plot and save the corresponding pair of confusion matrices (raw count and normalized). Do not skip any SNR or select only a few representatives.
     - **CRITICAL for Comparison:** To allow for a fair visual comparison of performance as SNR changes, the color bar ranges must be fixed across all SNR levels.
       - For the **normalized** (accuracy) matrices, the range must be fixed from **0.0 to 1.0** (e.g., use `vmin=0.0, vmax=1.0`).
       - For the **raw count** matrices, the range must be fixed from **0 to the total number of test samples per class at a single SNR level** (e.g., use `vmin=0, vmax=600` if there are 600 samples per class).
  7. **Accuracy vs. SNR Plot:**
     - Plot two separate charts:
       - Chart 1: A single line showing the **overall** accuracy as it changes with SNR.
       - Chart 2: Multiple lines on one chart, where each line represents the accuracy of a **single modulation type** as it changes with SNR. Use distinct, bold colors and a clear legend.
- **Results Summary:**
  - **Create a final Markdown cell** to summarize the key results:
    - Overall accuracy on the test set.
    - The SNR level at which the model begins to perform well (e.g., accuracy > 90%).
    - **Analyze Top Confusion Pairs:** Based on the overall confusion matrix (using raw counts), programmatically identify and list the top 2-3 confusion pairs.
      - A "confusion pair" is defined as a `(True Label, Predicted Label)` tuple where the True Label is not equal to the Predicted Label.
      - The goal is to find the off-diagonal cells with the highest values.
      - Present the findings clearly. For example: "1. Top Confusion: `8PSK` signals were most frequently misclassified as `QPSK` (occurred X times)."
- **Storage:** All generated images, plots, and the trained model must be saved to their corresponding defined directories.