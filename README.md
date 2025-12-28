# CLIPStyler - Fast Neural Style Transfer with CLIP

## Project Overview

CLIPStyler is a neural style transfer application that uses OpenAI's CLIP model combined with advanced loss functions to rapidly apply artistic styles to images using natural language descriptions. The project leverages directional CLIP loss, content preservation loss, total variation regularization, and patch-level consistency to achieve high-quality style transfer results.

**Key Features:**
- Fast style transfer training using CLIP embeddings
- Natural language style descriptions (e.g., "oil painting", "watercolor")
- Multiple training modes (individual images, dataset directories, inference)
- Real-time GUI with Streamlit
- Checkpoint support for model persistence
- Batch processing capabilities

---

## Project Architecture

The project follows a standard deep learning pipeline with 7 main phases:

### 1. Problem Definition and Data Collection
**Objective:** Define the neural style transfer problem and prepare image datasets.

**What it does:**
- Accepts user input images through the Streamlit GUI
- Supports both single image uploads and batch processing from directories
- Stores images in the `/data` directory for processing
- Allows flexible dataset scaling through the `scale_data` parameter

**Input Requirements:**
- Images in JPG, PNG format
- Any resolution (automatically resized to specified `img_size`: 256-1024px)
- Dataset can be single or multiple images

**Output:** Organized image dataset ready for preprocessing

---

### 2. Data Cleaning and Analysis

**Objective:** Prepare and curate high-quality image datasets, remove irrelevant images, and balance dataset representation.

**Why High-Resolution Images Matter:**
Our style transfer model relies on advanced loss functions that benefit significantly from high-resolution input images. Therefore, we use:
- **DIV2K Dataset:** ~800 high-quality diverse images (typically 2K resolution)
- **Flickr2K Dataset:** ~2,500 high-resolution images from Flickr (various artistic styles)

These datasets provide the image quality and detail needed for our CLIP-based loss functions to work effectively.

**The Challenge:**
However, during training we faced several issues:
- **Noisy/Messy Images:** Low-quality, corrupted, or irrelevant photos
- **Class Imbalance:** Too many images of one type (e.g., many landscapes, few portraits)
- **Unwanted Styles:** Images that don't match our target style domains

**Our Solution - Smart Filtering Using CLIP + K-Means Clustering:**

**Step 1: Extract Image Features with CLIP**
- Use CLIP's vision encoder to convert each image into a numerical "fingerprint" (512-dimensional vector)
- This fingerprint captures what the image is about (content, objects, style)

**Step 2: Identify Target Classes**
- Specify which image types we want (e.g., "persons", "landscapes", "sea")
- CLIP automatically filters images matching these descriptions
- This removes unwanted/messy images automatically

**Step 3: Balance Dataset Using K-Means Clustering**
- **Problem:** After filtering, we still have imbalance (many similar images, few diverse ones)
- **Solution:** Group similar images together using K-Means clustering on CLIP features
  - Cluster 1: All similar landscape photos → pick 50 representative images
  - Cluster 2: Different painting styles → pick 50 representative images
  - Cluster 3: Portrait variations → pick 50 representative images
- This ensures diversity across the dataset

**Processing Steps:**
```python
1. Load all images from DIV2K + Flickr2K
2. Extract CLIP embeddings for each image
3. Filter by class descriptions using CLIP
4. Apply K-Means clustering (typically 10-50 clusters)
5. Select balanced samples from each cluster
6. Final dataset: High-quality, balanced, diverse images
---

### 3. Feature Engineering

**Objective:** Extract meaningful features from images and text.

**What it does:**
- **CLIP Image Encoder:** Extracts image features using pre-trained OpenAI CLIP vision encoder
- **CLIP Text Encoder:** Converts style descriptions (e.g., "oil painting") into embedding space
- **VGG19 Content Encoder:** Extracts semantic content features for content preservation
- **Patch Extraction:** Creates overlapping patches for spatial consistency checking

**Key Components:**
- `clip_model`: `openai/clip-vit-base-patch32` for image/text embeddings
- `vgg19`: Pre-trained VGG19 (layers 0-21) for content loss
- `processor`: CLIPProcessor for tokenization and image preprocessing
- `patch_transform`: Extracts N×N patches for patch-level consistency

**Output:** 
- Image feature embeddings (512-dim CLIP vectors)
- Text feature embeddings (512-dim for each style description)
- Content feature pyramids (multi-scale VGG features)
- Patch embeddings for spatial validation

---

### 4. Model Design

**Objective:** Design the neural style transfer architecture.

**Model Components:**

#### A. **Style Transfer Network** (Encoder-Decoder)
- **Downsampling Path:** Reduces spatial dimensions while increasing channels
  - `DownBlock`: Conv → InstanceNorm → ReLU (3×3 kernels)
  - Progressive resolution reduction with feature accumulation
  
- **Bottleneck:** Processes full resolution features
  
- **Upsampling Path:** Restores spatial dimensions with skip connections
  - `UpBlock`: Bilinear upsampling + concatenation + Conv
  - Skip connections from corresponding downsampling layers
  - Maintains fine details while applying style

#### B. **Loss Functions**

1. **DirectionalCLIPLoss** (`DirectionalCLIPLoss` class)
   - **Purpose:** Enforce style transfer through CLIP embedding space
   - **Mechanism:**
     - Computes directional difference: `diff = (style_embedding - photo_embedding)`
     - Aligns generated image features with style direction
     - Loss formula: `1 - cosine_similarity(generated_diff, style_diff)`
   - **Patch Loss:** Validates patch-level consistency with threshold filtering
   - **Parameters:** `threshold=0.7` (ignores low patch loss which could cause overfitting)

2. **ContentLoss** (`ContentLoss` class)
   - **Purpose:** Preserve original image content
   - **Mechanism:** 
     - Multi-scale VGG feature matching (4 encoder levels)
     - MSE between generated and original features at each level
     - Prevents excessive content destruction
   - **Formula:** `Σ ||VGG(generated) - VGG(original)||²`

3. **TotalVariationLoss** (`TotalVariationLoss` class)
   - **Purpose:** Reduce noise and artifacts
   - **Mechanism:** Penalizes high-frequency variations
   - **Formula:** `||∇ₓ image|| + ||∇ᵧ image||`

4. **Combined Loss** (`CLIPStylerLoss` class)
   ```
   Total Loss = λ_clip × clip_loss 
              + λ_patch × patch_loss 
              + λ_content × content_loss 
              + λ_tv × tv_loss
   ```
   **Tuning Parameters:**
   - `λ_clip` (500-2000): Style strength; higher = stronger style transfer
   - `λ_patch` (5000-20000): Spatial consistency; typically ~10× λ_clip
   - `λ_content` (50-300): Content preservation; higher = keep more original
   - `λ_tv` (0.001-0.1): Noise reduction; higher = smoother but less detail

**Architecture Diagram:**
```
Input Image (B, 3, H, W)
    ↓
[DownBlock 1] → 64 channels
    ↓
[DownBlock 2] → 128 channels
    ↓
[DownBlock 3] → 256 channels
    ↓
Bottleneck Features
    ↓
[UpBlock 3] → skip from DownBlock 3 → 128 channels
    ↓
[UpBlock 2] → skip from DownBlock 2 → 64 channels
    ↓
[UpBlock 1] → skip from DownBlock 1 → 3 channels (RGB)
    ↓
Output Image (B, 3, H, W) [0-255]
```
NOTE: you can adjust the number of blocks and channels as needed using depth paramter.
**Pre-trained Models Used:**
- CLIP ViT-Base-32 (frozen for feature extraction)
- VGG19 (frozen for content loss)
- Both models are non-trainable; only the style transfer network is optimized

---

### 5. Model Training

**Objective:** Optimize the style transfer network on provided images.

**Training Configuration:**

**File:** `train.ipynb` / `app.py` → `train()` function in `utils_t.py`

**Training Modes:**

1. **Mode 1: Train on Uploads** (Interactive mode)
   - Upload custom images
   - Specify style prompts (e.g., "watercolor", "sketch")
   - Real-time parameter tuning

2. **Mode 2: Full Training (Directory)**
   - Load entire directory of images
   - Batch processing across dataset
   - Useful for large-scale training

3. **Mode 3: Inference Only**
   - Load pre-trained checkpoint
   - Apply learned style to new images
   - No training, only inference

**Training Loop:**

```python
for epoch in range(n_epochs):
    for batch in data_loader:
        # 1. Forward pass through style transfer network
        styled_images = model(original_images)
        
        # 2. Compute style loss using CLIP
        clip_loss, patch_loss = directional_clip_loss(
            original_images, styled_images
        )
        
        # 3. Compute content loss using VGG
        content_loss = content_loss_fn(styled_images, original_images)
        
        # 4. Compute total variation loss
        tv_loss = tv_loss_fn(styled_images)
        
        # 5. Combine all losses
        total_loss = (λ_clip * clip_loss + 
                     λ_patch * patch_loss + 
                     λ_content * content_loss + 
                     λ_tv * tv_loss)
        
        # 6. Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

**Key Hyperparameters:**
- **Epochs:** 1-50 (default: 10)
- **Batch Size:** 1-8 (default: 1)
- **Learning Rate:** 1e-5 to 1e-2 (default: 5e-4)
- **Image Size:** 256-1024px (default: 512)
- **Dataset Scaling:** Repeat dataset N times (default: 1)

**Optimization Details:**
- **Optimizer:** Adam (typical, configured in training function)
- **Scheduler:** Cosine with warmup restarts learning rate scheduling
- **GPU Support:** CUDA-enabled for faster training
- **Mixed Precision:** for memory efficiency

**Checkpointing:**
- Saves model every N epochs (PyTorch Lightning integration)
- Allows resuming training from checkpoints
- Saves loss history for analysis

---

### 6. Model Testing and Inference

**Objective:** Evaluate trained models and apply style transfer to new images.

**Inference Modes:**

**Mode 1: Train on Uploads (Fastest)**
- train on user-uploaded images
- show real-time results

**Mode 2: Checkpoint**
- use pre-trained model checkpoints
- apply learned styles to new images


**Testing Metrics:**
- **Visual Quality:** Preservation of content while applying style
- **Style Consistency:** Uniformity of style across image regions
- **Artifact Detection:** Check for noise, distortions, color shifts
- **Perceptual Similarity:** CLIP similarity between styled image and target style

All metrics logged using tensorboard during training and inference for performance tracking.

**Output Generation:**
```
Input Image → Style Transfer Network → Styled Image (0-255)
                                          ↓
                                    Save as PNG/JPG
                                    Display in GUI
                                    Log results
```

**Performance Metrics Logged:**
- Inference time per image
- GPU memory usage
- Loss
- Real Image VS Generated

---

### 7. GUI Implementation and Application Running

**Objective:** Provide user-friendly interface for style transfer.

**Technology Stack:**
- **Framework:** Streamlit (Python web framework)
- **Backend:** PyTorch + Transformers
- **Frontend:** Streamlit widgets
- **File Management:** Temporary directories for uploads

**GUI Features:**

**File:** `app.py` (327 lines)

#### A. **Input Section**
- **Mode Selection:** Radio buttons for Train/Inference modes
- **Image Upload:** Drag-and-drop file uploader for JPG/PNG
- **Directory Input:** Text field for batch processing paths
- **Style Prompts:** Text area for multi-line style descriptions
- **Checkpoint Loading:** Upload pre-trained models (.ckpt, .pt)

#### B. **Parameter Configuration**
- **Training Parameters:**
  - Epochs (slider: 1-50)
  - Learning rate (selection: 1e-5 to 1e-2)
  - Batch size (slider: 1-8)
  - Image size (slider: 256-1024)
  - Dataset scaling factor (1-1000)

- **Loss Weights:**
  - λ CLIP: 10-5000 (style strength)
  - λ Content: 10-500 (content preservation)
  - λ Patch: 100-20000 (spatial smoothness)
  - λ TV: 1e-3 to 1e-1 (noise reduction)

#### C. **Output Section**
- **Real-time Progress:** Training loss visualization
- **Result Display:** Side-by-side original vs. styled images
- **Download:** Export styled images and checkpoints
- **Metrics:** Display training loss curves

#### D. **Session Management**
- Streamlit session state for tracking training status
- Temporary file handling for uploads
- Persistent checkpoint storage

**How to Run:**

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Open browser to http://localhost:8501
```

**Application Workflow:**

```
1. User opens app.py in browser (via Streamlit)
   ↓
2. Select mode (Train/Inference)
   ↓
3. Upload images and specify styles
   ↓
4. Adjust hyperparameters
   ↓
5. Click "Train" or "Apply Style"
   ↓
6. App processes and displays results
   ↓
7. Download styled images and checkpoints
   ↓
8. (Optional) Continue training with loaded checkpoint
```

---

## Project File Structure

```
CLIPStyler/
├── app.py                    # Streamlit GUI application
├── utils_t.py               # Core ML utilities and classes
├── train.ipynb              # Training pipeline notebook
├── cleaning.ipynb           # Data cleaning and exploration
├── README.md                # This file
│
├── data/                    # Training Image dataset directory

```

---

## Key Classes and Functions in `utils_t.py`

### Loss Functions
- **`DirectionalCLIPLoss`**: Enforces style alignment in CLIP embedding space
- **`ContentLoss`**: Preserves original image content using VGG features
- **`TotalVariationLoss`**: Reduces artifacts via regularization
- **`CLIPStylerLoss`**: Combines all losses with weighted parameters

### Network Architecture
- **`DownBlock`**: Downsampling with convolutions and instance normalization
- **`UpBlock`**: Upsampling with skip connections
- **`StyleTransferNetwork`**: Full encoder-decoder architecture

### Pre-trained Models
- **CLIP ViT-Base-32**: Image and text feature extraction
- **VGG19**: Perceptual content loss (frozen)
- **CLIPProcessor**: Input preprocessing

### Data Handling
- **`CustomDataset`**: PyTorch Dataset class for image loading
- **`DataLoader`**: Batch processing with shuffling

### Training Function
- **`train()`**: Main training loop with logging and checkpointing

---

## Dependencies

**Core Libraries:**
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `transformers` - Pre-trained models (HuggingFace)
- `lightning` - Training utilities and callbacks
- `kornia` - Advanced image augmentation
- `streamlit` - Web GUI framework
- `PIL` - Image processing
- `numpy` - Numerical computing
- `matplotlib` - Visualization (optional)

**Installation:**
```bash
pip install torch torchvision transformers lightning kornia streamlit pillow numpy
```

---

## Team Member Responsibilities

### Team Leader
- **Form Submission:** Enter team member names (in Arabic) with academic IDs
- **Session Instructor Selection:** Choose assigned instructor from dropdown
- **Project Organization:** Coordinate task distribution and deadlines
- **Final Submission:** Upload complete code and README to GitHub



---

## Usage Examples

### Example 1: Train on Uploaded Images
```python
# Launch app
streamlit run app.py

# In GUI:
1. Select "Train on Uploads"
2. Upload 5-10 photos
3. Enter styles: "oil painting", "impressionist", "watercolor"
4. Set epochs=20, λ_clip=1000, λ_content=150
5. Click Train
6. View results and download styled images
```

### Example 2: Batch Processing from Directory
```python
# In GUI:
1. Select "Full Training (Directory)"
2. Enter path: "C:\data\my_photos"
3. Specify style: "Van Gogh style"
4. Set scale_data=3 (repeat dataset 3 times)
5. Train for 15 epochs
```

### Example 3: Apply Trained Style to New Images
```python
# In GUI:
1. Select "Predict (Inference)"
2. Load checkpoint from "models/oil_painting.ckpt"
3. Upload new images
4. View real-time stylized output
```


---

## Performance Considerations

- **Training Time:** 5-30 minutes per style (depending on epochs, image size, GPU)
- **GPU Requirements:** NVIDIA GPU with ≥6GB VRAM recommended
- **Memory Usage:** ~4-8GB GPU memory for batch_size=4, img_size=512
- **Inference Speed:** ~0.5-2 seconds per image (depending on resolution)

---


### Saaid Ayad: Data Collection & Cleaning and Loss Functions
- **Tasks:**
  1. Collect DIV2K and Flickr2K datasets
  2. Implement CLIP-based filtering for relevant images
  3. Apply K-Means clustering for dataset balancing
  4. Document data cleaning process (see Data Cleaning section above)
  5. Implement DirectionalCLIPLoss, ContentLoss, TotalVariationLoss classes
  6. Helped in performance tuning efficiency increased performance by 700%
### Mohamed Diab: U-Net Architecture
- **Tasks:**
    1. Suggest U-Net style architecture for style transfer which was efficient and effective
    2. Design DownBlock and UpBlock modules
    3. Implement StyleTransferNetwork class
    4. Integrate skip connections
    5. Test architecture with dummy data
    6. Document architecture decisions (see Model Design section above)
### Khaled Eissa: Training Pipeline and Lighting Integration
- **Tasks:**
    1. Integrate PyTorch Lightning for training making it modular
    2. Implement training and validation loops with loss calculations
    3. Set up data loaders and batching
    4. Add checkpointing and logging functionality
    5. Monitor training experiments and document results (see `train.ipynb`)
    6. Debugging (With Tears)
### Abo Bakr & Mohamed Eldesouky: Streamlit GUI Development
- **Tasks:**
    1. Design user interface with Streamlit
    2. Implement file uploaders and parameter input fields
    3. Connect GUI inputs to training/inference functions
    4. Display real-time training progress and results
    5. Made the GUI user-friendly and intuitive
    6. Test GUI usability and document usage instructions (see GUI Implementation section above)