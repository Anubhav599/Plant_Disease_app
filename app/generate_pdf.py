from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from datetime import datetime

# Create PDF
pdf_file = "Plant_Disease_Classifier_Technical_Documentation.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)

# Container for the 'Flowable' objects
elements = []

# Define styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#1f4788'),
    spaceAfter=12,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=colors.HexColor('#2563eb'),
    spaceAfter=8,
    spaceBefore=8,
    fontName='Helvetica-Bold'
)

subheading_style = ParagraphStyle(
    'CustomSubHeading',
    parent=styles['Heading3'],
    fontSize=12,
    textColor=colors.HexColor('#1e40af'),
    spaceAfter=6,
    fontName='Helvetica-Bold'
)

body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['BodyText'],
    fontSize=10,
    alignment=TA_JUSTIFY,
    spaceAfter=8,
    leading=12
)

# Title Page
elements.append(Paragraph("PLANT DISEASE CLASSIFIER", title_style))
elements.append(Paragraph("AI/ML Model - Technical Documentation", heading_style))
elements.append(Spacer(1, 0.3*inch))
elements.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y')}", body_style))
elements.append(Paragraph("<b>Project Type:</b> Deep Learning Classification System", body_style))
elements.append(Spacer(1, 0.5*inch))

# Section 1: Project Overview
elements.append(Paragraph("1. PROJECT OVERVIEW", heading_style))
elements.append(Spacer(1, 0.1*inch))
overview_text = """
A <b>Plant Disease Classification System</b> using Deep Learning (Convolutional Neural Networks) to identify 
38 different plant diseases from leaf images. The system uses TensorFlow/Keras for model training and Streamlit 
for web-based deployment. The model processes leaf images and provides real-time disease classification with 
probability scores for accurate plant health diagnosis.
"""
elements.append(Paragraph(overview_text, body_style))
elements.append(Spacer(1, 0.2*inch))

# Key Statistics Table
stats_data = [
    ['Metric', 'Value'],
    ['Total Classes', '38 plant diseases'],
    ['Dataset Source', 'PlantVillage (Kaggle)'],
    ['Training Samples', '54,305 images'],
    ['Input Image Size', '224×224 pixels (RGB)'],
    ['Model Architecture', 'Convolutional Neural Network (CNN)'],
    ['Training Epochs', '5'],
    ['Batch Size', '32'],
    ['Train/Validation Split', '80%/20%'],
]

stats_table = Table(stats_data, colWidths=[2.5*inch, 3.5*inch])
stats_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 10),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f9ff')]),
]))
elements.append(stats_table)
elements.append(Spacer(1, 0.3*inch))

# Section 2: Machine Learning Concepts
elements.append(PageBreak())
elements.append(Paragraph("2. MACHINE LEARNING CONCEPTS", heading_style))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("2.1 Deep Learning Architecture - CNN (Convolutional Neural Network)", subheading_style))
cnn_text = """
A Convolutional Neural Network (CNN) is a specialized deep learning architecture designed for processing 
gridded data like images. It automatically learns spatial hierarchies of features through convolutional layers, 
making it ideal for image classification tasks. The model consists of:
"""
elements.append(Paragraph(cnn_text, body_style))

# CNN Layers Table
cnn_data = [
    ['Layer', 'Parameters', 'Purpose'],
    ['Conv2D (Layer 1)', '32 filters, 3×3 kernel', 'Extract low-level features (edges, textures)'],
    ['MaxPooling2D', '2×2 pool size', 'Reduce spatial dimensions, retain features'],
    ['Conv2D (Layer 2)', '64 filters, 3×3 kernel', 'Extract higher-level features (patterns, shapes)'],
    ['MaxPooling2D', '2×2 pool size', 'Further dimensionality reduction'],
    ['Flatten', 'N/A', 'Convert 2D feature maps to 1D vector'],
    ['Dense', '256 neurons, ReLU activation', 'Learn non-linear combinations of features'],
    ['Output Dense', '38 neurons, Softmax activation', '38-class probability distribution'],
]

cnn_table = Table(cnn_data, colWidths=[1.8*inch, 2*inch, 2.7*inch])
cnn_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#eff6ff')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f9ff')]),
]))
elements.append(cnn_table)
elements.append(Spacer(1, 0.2*inch))

why_cnn = """
<b>Why CNN?</b> CNNs are superior for image classification because they: (1) Automatically learn spatial 
hierarchies of features, (2) Exhibit shift and rotation invariance through convolution operations, 
(3) Share parameters across the image reducing model complexity, (4) Require fewer parameters than 
fully connected networks, (5) Perform exceptionally well on image recognition tasks.
"""
elements.append(Paragraph(why_cnn, body_style))
elements.append(Spacer(1, 0.2*inch))

# Key ML Concepts Table
elements.append(Paragraph("2.2 Key Machine Learning Concepts", subheading_style))
ml_concepts_data = [
    ['Concept', 'Explanation', 'Application'],
    ['Convolution', 'Applies filters across image to detect patterns', 'Feature extraction from leaf images'],
    ['Pooling', 'Reduces dimensionality while preserving features', 'Prevents overfitting, improves efficiency'],
    ['Activation Function (ReLU)', 'Introduces non-linearity (max(0, x))', 'Enables learning complex boundaries'],
    ['Softmax', 'Converts logits to probability distribution', 'Multi-class classification output'],
    ['Categorical Cross-Entropy', 'Loss function for multi-class problems', 'Measures prediction error'],
    ['Optimization (Adam)', 'Adaptive learning rate optimizer', 'Efficiently updates network weights'],
    ['Epochs', 'Complete passes through training data', '5 epochs used in this model'],
    ['Batch Size', 'Number of samples processed together', '32 samples per batch'],
    ['Validation Split', 'Percentage of data for validation', '20% used for testing generalization'],
]

ml_table = Table(ml_concepts_data, colWidths=[1.5*inch, 2.2*inch, 2.8*inch])
ml_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f9ff')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf7ff')]),
]))
elements.append(ml_table)
elements.append(Spacer(1, 0.3*inch))

# Section 3: Image Preprocessing
elements.append(PageBreak())
elements.append(Paragraph("3. IMAGE PREPROCESSING PIPELINE", heading_style))
elements.append(Spacer(1, 0.1*inch))

preprocess_text = """
Image preprocessing is a critical step that prepares raw input data for the neural network. The pipeline 
follows this sequence: <b>Input Image (JPG/PNG) → Resize (224×224) → Convert to Array → Add Batch Dimension 
→ Normalize (0-1) → CNN Model</b>
"""
elements.append(Paragraph(preprocess_text, body_style))
elements.append(Spacer(1, 0.15*inch))

elements.append(Paragraph("3.1 Preprocessing Steps (Detailed)", subheading_style))

preprocess_steps_data = [
    ['Step', 'Operation', 'Reason/Purpose'],
    ['1. Load Image', 'PIL.Image opens JPG/PNG file', 'Initial input from user or disk'],
    ['2. Resize', 'Scale to 224×224 pixels', 'Standard CNN input size; consistent dimensions'],
    ['3. Convert to Array', 'NumPy array format', 'Required for tensor operations in TensorFlow'],
    ['4. Add Batch Dimension', 'Shape: (224, 224, 3) → (1, 224, 224, 3)', 'Model expects batch of images'],
    ['5. Normalize', 'Divide pixel values by 255', 'Scale to [0, 1]; prevents gradient explosion'],
    ['6. Forward Pass', 'Feed through CNN layers', 'Extract features and classify'],
]

preprocess_table = Table(preprocess_steps_data, colWidths=[0.8*inch, 1.8*inch, 3.3*inch])
preprocess_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#059669')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecfdf5')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0fdf4')]),
]))
elements.append(preprocess_table)
elements.append(Spacer(1, 0.2*inch))

normalize_text = """
<b>Why Normalization?</b> Networks train significantly better with normalized inputs because: 
(1) Prevents vanishing/exploding gradients, (2) Accelerates training convergence, (3) Improves numerical stability, 
(4) Reduces internal covariate shift, (5) Standardizes input scale across all features.
"""
elements.append(Paragraph(normalize_text, body_style))
elements.append(Spacer(1, 0.3*inch))

# Section 4: Model Training Flow
elements.append(Paragraph("4. MODEL TRAINING FLOW (DETAILED)", heading_style))
elements.append(Spacer(1, 0.1*inch))

training_text = """
The training process involves multiple stages from data preparation through model optimization and evaluation. 
Each stage is critical for creating an accurate, generalizable model.
"""
elements.append(Paragraph(training_text, body_style))
elements.append(Spacer(1, 0.15*inch))

# Training Flow Stages
training_stages = [
    ('1. DATA CURATION', 
     'Download PlantVillage dataset from Kaggle (54,305 images)\nExtract 38 disease classes\nUse "color" variant images for better feature extraction'),
    
    ('2. DATA PREPARATION',
     'Create ImageDataGenerator for preprocessing\nSplit: 80% training, 20% validation\nBatch size: 32 samples\nTarget size: 224×224 pixels\nRescale: divide by 255 to normalize'),
    
    ('3. MODEL ARCHITECTURE',
     'Build Sequential model (linear layer stack)\nConv2D(32) + MaxPooling → Conv2D(64) + MaxPooling\nFlatten → Dense(256, ReLU) → Output Dense(38, Softmax)\nTotal parameters: ~1.3M'),
    
    ('4. COMPILATION',
     'Optimizer: Adam (adaptive moment estimation)\nLoss Function: Categorical cross-entropy\nMetrics: Accuracy for monitoring performance'),
    
    ('5. TRAINING',
     'Forward pass: Input → through all layers → output probabilities\nBackward propagation: Calculate gradients\nWeight updates: Adam optimizer adjusts all parameters\nValidation: Evaluate on held-out validation data\nRepeated for 5 epochs (5 complete passes through dataset)'),
    
    ('6. EVALUATION & STORAGE',
     'Calculate validation accuracy\nPlot training vs validation accuracy curves\nPlot training vs validation loss curves\nSave model as plant_disease_model.h5 (HDF5 format)')
]

for stage_title, stage_details in training_stages:
    elements.append(Paragraph(f"<b>{stage_title}</b>", subheading_style))
    elements.append(Paragraph(stage_details.replace('\n', '<br/>'), body_style))
    elements.append(Spacer(1, 0.1*inch))

elements.append(Spacer(1, 0.2*inch))

# Section 5: Inference Pipeline
elements.append(PageBreak())
elements.append(Paragraph("5. INFERENCE PIPELINE (DEPLOYMENT - main.py)", heading_style))
elements.append(Spacer(1, 0.1*inch))

inference_text = """
The inference pipeline is used when the model is deployed and makes predictions on new, unseen data. 
It follows this sequence: <b>User Upload Image → Load & Preprocess → Model Prediction → Class Mapping 
→ Display Result</b>
"""
elements.append(Paragraph(inference_text, body_style))
elements.append(Spacer(1, 0.15*inch))

# Detailed Inference Steps
inference_steps = [
    ('1. STREAMLIT UI INITIALIZATION',
     'Display title: "Plant Disease Classifier"\nCreate file uploader widget\nAccept: JPG, JPEG, PNG formats'),
    
    ('2. MODEL & CLASS LOADING',
     'Load pre-trained model: plant_disease_model.h5\nLoad class indices: class_indices.json (38 disease mappings)\nModel loaded once during app startup for efficiency'),
    
    ('3. IMAGE UPLOAD',
     'User selects leaf image through web interface\nImage stored in temporary memory\nNo direct file system access required'),
    
    ('4. PREPROCESSING',
     'Open image using PIL\nResize to 224×224 pixels\nConvert to NumPy array\nAdd batch dimension: (1, 224, 224, 3)\nNormalize: divide by 255 → [0, 1] range'),
    
    ('5. MODEL INFERENCE',
     'Forward pass through 7 layers\nOutput: 38 probability values (sum = 1.0)\nComputation time: ~100-500ms depending on hardware'),
    
    ('6. CLASS PREDICTION',
     'Use np.argmax() to find highest probability index\nMap index to disease name using class_indices.json\nExample: Index 0 → "Apple___Apple_scab"'),
    
    ('7. RESULT DISPLAY',
     'Display original image (150×150 px)\nShow prediction result using st.success()\nFormat: "Prediction: [Disease_Name]"\nProbability scores could be added for confidence level')
]

for step_title, step_details in inference_steps:
    elements.append(Paragraph(f"<b>{step_title}</b>", subheading_style))
    elements.append(Paragraph(step_details.replace('\n', '<br/>'), body_style))
    elements.append(Spacer(1, 0.08*inch))

elements.append(Spacer(1, 0.2*inch))

# Section 6: Dataset
elements.append(PageBreak())
elements.append(Paragraph("6. DATASET - 38 PLANT DISEASES", heading_style))
elements.append(Spacer(1, 0.1*inch))

dataset_text = """
The model is trained on the PlantVillage dataset containing 54,305 high-quality leaf images across 
14 crop types and 38 disease/health classes. Below is the complete classification scheme:
"""
elements.append(Paragraph(dataset_text, body_style))
elements.append(Spacer(1, 0.15*inch))

# Dataset Table
dataset_data = [
    ['Crop', 'Classes', 'Count'],
    ['Apple', 'Apple_scab, Black_rot, Cedar_apple_rust, healthy', '4'],
    ['Blueberry', 'healthy', '1'],
    ['Cherry', 'Powdery_mildew, healthy', '2'],
    ['Corn (Maize)', 'Cercospora, Common_rust, Northern_Leaf_Blight, healthy', '4'],
    ['Grape', 'Black_rot, Esca, Leaf_blight, healthy', '4'],
    ['Orange', 'Haunglongbing (Citrus_greening)', '1'],
    ['Peach', 'Bacterial_spot, healthy', '2'],
    ['Pepper, Bell', 'Bacterial_spot, healthy', '2'],
    ['Potato', 'Early_blight, Late_blight, healthy', '3'],
    ['Raspberry', 'healthy', '1'],
    ['Soybean', 'healthy', '1'],
    ['Squash', 'Powdery_mildew', '1'],
    ['Strawberry', 'Leaf_scorch, healthy', '2'],
    ['Tomato', 'Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot, Spider_mites, Target_Spot, YLCV, TMV, healthy', '10'],
    ['TOTAL', '', '38'],
]

dataset_table = Table(dataset_data, colWidths=[1.5*inch, 3.5*inch, 1*inch])
dataset_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7c2d12')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
    ('ALIGN', (1, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fef3c7')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef9e7')]),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
]))
elements.append(dataset_table)
elements.append(Spacer(1, 0.3*inch))

# Section 7: Technical Stack
elements.append(PageBreak())
elements.append(Paragraph("7. TECHNICAL STACK & DEPENDENCIES", heading_style))
elements.append(Spacer(1, 0.1*inch))

tech_data = [
    ['Component', 'Technology', 'Version', 'Purpose'],
    ['Deep Learning Framework', 'TensorFlow/Keras', 'Latest', 'Model training & inference, layer abstractions'],
    ['Image Processing', 'PIL (Pillow)', 'Latest', 'Load, resize, manipulate images'],
    ['Numerical Computing', 'NumPy', 'Latest', 'Array operations, matrix computations'],
    ['Web UI Framework', 'Streamlit', 'Latest', 'Interactive web interface, rapid prototyping'],
    ['Data Source', 'PlantVillage Dataset', 'Kaggle', '54,305 plant leaf images, 38 classes'],
    ['Containerization', 'Docker', '20.10+', 'Consistent deployment across environments'],
    ['Configuration', 'TOML', 'Python standard', 'Server settings, browser configuration'],
    ['Serialization', 'HDF5 (.h5)', 'TensorFlow standard', 'Model architecture & weights storage'],
]

tech_table = Table(tech_data, colWidths=[1.8*inch, 1.8*inch, 1.2*inch, 1.7*inch])
tech_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#eef2ff')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f3ff')]),
]))
elements.append(tech_table)
elements.append(Spacer(1, 0.3*inch))

# Section 8: Model Architecture Visualization
elements.append(Paragraph("8. MODEL ARCHITECTURE VISUALIZATION", heading_style))
elements.append(Spacer(1, 0.1*inch))

arch_text = """
<b>Architecture Flow:</b>

INPUT LAYER (224×224×3 RGB Image)
↓
Convolutional Layer 1 (Conv2D with 32 filters, 3×3 kernel) + ReLU Activation
↓
Max Pooling Layer 1 (2×2 pool size) [Reduces spatial dimensions to ~112×112]
↓
Convolutional Layer 2 (Conv2D with 64 filters, 3×3 kernel) + ReLU Activation
↓
Max Pooling Layer 2 (2×2 pool size) [Further reduces to ~56×56]
↓
Flatten Layer [Converts all feature maps to 1D vector]
↓
Fully Connected (Dense) Layer 1 (256 neurons, ReLU activation) [Non-linear feature combinations]
↓
Output Dense Layer (38 neurons, Softmax activation) [Probability distribution across diseases]
↓
OUTPUT [38 disease probability scores, sum = 1.0]
"""
elements.append(Paragraph(arch_text, body_style))
elements.append(Spacer(1, 0.3*inch))

# Section 9: Hyperparameters
elements.append(PageBreak())
elements.append(Paragraph("9. HYPERPARAMETERS & CONFIGURATION", heading_style))
elements.append(Spacer(1, 0.1*inch))

hyperparams_data = [
    ['Parameter', 'Value', 'Rationale'],
    ['Image Input Size', '224×224 pixels', 'Standard for modern CNNs; balance between detail and computation'],
    ['Batch Size', '32 samples', 'Trade-off between memory usage and training stability'],
    ['Training Epochs', '5', 'Limited epochs in demo; production uses 50-200 for better convergence'],
    ['Validation Split', '20%', 'Sufficient for monitoring generalization without reducing training data'],
    ['Conv2D Filters (Layer 1)', '32', 'Captures low-level features; reduced complexity'],
    ['Conv2D Filters (Layer 2)', '64', 'Captures higher-level patterns; doubles filter count'],
    ['Kernel Size', '3×3', 'Standard for feature extraction; balance between receptive field and parameters'],
    ['Pooling Size', '2×2', 'Reduces spatial dimensions by 50% per pooling layer'],
    ['Dense Layer Neurons', '256', 'Sufficient capacity for learning feature combinations'],
    ['Output Layer Neurons', '38', 'One neuron per disease class'],
    ['ReLU Activation', 'max(0, x)', 'Introduces non-linearity; standard for hidden layers'],
    ['Softmax Activation', 'e^x / Σ(e^x)', 'Normalizes outputs to probabilities for multi-class'],
    ['Optimizer', 'Adam', 'Adaptive learning rate; faster convergence than SGD'],
    ['Learning Rate (Adam)', '~0.001 (default)', 'Balanced convergence speed and stability'],
    ['Loss Function', 'Categorical Cross-Entropy', 'Standard for multi-class classification'],
]

hp_table = Table(hyperparams_data, colWidths=[1.8*inch, 1.8*inch, 2.3*inch])
hp_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c2410c')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 8),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fed7aa')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 7),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef3c7')]),
]))
elements.append(hp_table)
elements.append(Spacer(1, 0.3*inch))

# Section 10: Loss Functions & Optimization
elements.append(PageBreak())
elements.append(Paragraph("10. LOSS FUNCTIONS & OPTIMIZATION", heading_style))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("10.1 Categorical Cross-Entropy Loss", subheading_style))
loss_text = """
The loss function quantifies the difference between predicted and actual distributions. Categorical Cross-Entropy 
is the standard for multi-class classification:

<b>Formula: L = -Σ(y_i × log(ŷ_i))</b>

Where:
• y_i = true label (one-hot encoded, 0 or 1)
• ŷ_i = predicted probability for class i
• Σ = sum across all 38 classes

<b>Why Cross-Entropy?</b> It heavily penalizes confident wrong predictions, encouraging the model to learn 
discriminative features. It has nice mathematical properties for backpropagation and gradient descent optimization.
"""
elements.append(Paragraph(loss_text, body_style))
elements.append(Spacer(1, 0.2*inch))

elements.append(Paragraph("10.2 Adam Optimizer", subheading_style))
adam_text = """
Adam (Adaptive Moment Estimation) is an advanced optimization algorithm that adapts learning rates for each parameter:

<b>Key Features:</b>
• Maintains exponential moving average of gradients (momentum term)
• Maintains exponential moving average of squared gradients (RMSprop term)
• Adapts learning rate per parameter based on these moving averages
• Default learning rate: 0.001 (typically requires no tuning)
• Computationally efficient and memory-friendly

<b>Advantages over Standard SGD:</b>
• Faster convergence with fewer iterations
• Handles sparse gradients well
• Robust to different hyperparameter choices
• Works well with mini-batches (batch size: 32)
"""
elements.append(Paragraph(adam_text, body_style))
elements.append(Spacer(1, 0.3*inch))

# Section 11: Deployment Architecture
elements.append(Paragraph("11. DEPLOYMENT ARCHITECTURE (DOCKER)", heading_style))
elements.append(Spacer(1, 0.1*inch))

docker_text = """
The application is containerized using Docker for consistent deployment across different environments. 
The container includes all dependencies, the trained model, and configuration files.
"""
elements.append(Paragraph(docker_text, body_style))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("Docker Container Structure:", subheading_style))

docker_struct = """
<b>Base Image:</b> python:3.10-slim (lightweight Python runtime)

<b>Container Setup:</b>
• COPY: Copy all application files to /app directory
• WORKDIR: Set /app as working directory
• RUN: Execute pip install for dependencies
• EXPOSE: Port 80 (HTTP traffic)

<b>Configuration:</b>
• Create ~/.streamlit directory
• Copy config.toml (server settings)
• Copy credentials.toml (authentication)

<b>Entrypoint:</b> streamlit run main.py
• Starts Streamlit server on 0.0.0.0:80
• Accessible from any network interface
• Port 80: Standard HTTP port (no port forwarding needed)

<b>Advantages:</b>
• Reproducible deployments across servers
• Isolation from host system dependencies
• Easy scaling and orchestration
• Version control for entire environment
"""
elements.append(Paragraph(docker_struct, body_style))
elements.append(Spacer(1, 0.3*inch))

# Section 12: Critical ML Concepts
elements.append(PageBreak())
elements.append(Paragraph("12. CRITICAL ML CONCEPTS FOR DEVELOPMENT", heading_style))
elements.append(Spacer(1, 0.1*inch))

concepts = [
    ('Overfitting Prevention',
     'Problem: Model memorizes training data instead of learning generalizable patterns.\nSolution: Use validation set (20%) to monitor generalization. If val_loss increases while train_loss decreases = overfitting.\nTechniques: Early stopping, dropout layers, regularization, data augmentation.'),
    
    ('Data Augmentation',
     'Creates variations of training images (rotations, flips, zooms, color changes).\nBenefits: Increases effective training data size, improves model robustness, prevents overfitting.\nImplementation: ImageDataGenerator in current code supports multiple augmentation techniques.'),
    
    ('Feature Extraction',
     'CNN layers learn hierarchical features: Layer 1 (edges) → Layer 2 (textures) → Dense layers (object parts).\nTransfer Learning: Use pre-trained weights (ImageNet) instead of training from scratch for faster convergence.'),
    
    ('Vanishing Gradients',
     'Problem: Gradients become too small during backpropagation in deep networks.\nSolution: ReLU activation, batch normalization, proper weight initialization.\nImpact: Without mitigation, network cannot learn effectively.'),
    
    ('Model Serialization',
     'Save/load models for reuse without retraining. HDF5 format (.h5) stores:\n• Model architecture (layer definitions)\n• Weights (learned parameters)\n• Training configuration\nBenefit: Inference uses pre-trained weights, ~1000x faster than training.'),
    
    ('Gradient Descent & Backpropagation',
     'Forward Pass: Input → layers → output predictions\nBackward Pass: Calculate ∂L/∂w (gradient of loss w.r.t. weights)\nUpdate: w_new = w_old - α × ∂L/∂w (where α = learning rate)\nRepeated for all layers in reverse order (hence "back" propagation).'),
]

for concept_title, concept_details in concepts:
    elements.append(Paragraph(f"<b>{concept_title}</b>", subheading_style))
    elements.append(Paragraph(concept_details.replace('\n', '<br/>'), body_style))
    elements.append(Spacer(1, 0.12*inch))

elements.append(Spacer(1, 0.3*inch))

# Section 13: Performance Metrics
elements.append(PageBreak())
elements.append(Paragraph("13. PERFORMANCE METRICS & EVALUATION", heading_style))
elements.append(Spacer(1, 0.1*inch))

metrics_text = """
The model's performance is evaluated using multiple metrics to ensure accuracy and generalization:
"""
elements.append(Paragraph(metrics_text, body_style))
elements.append(Spacer(1, 0.1*inch))

metrics = [
    ('Accuracy',
     'Percentage of correct predictions: (True Positives + True Negatives) / Total\nRange: 0-100% (higher is better)\nUsage: Overall performance metric across all classes.'),
    
    ('Loss',
     'Categorical Cross-Entropy loss value\nRange: 0 to ∞ (lower is better)\nTraining Loss: Monitors learning on training data\nValidation Loss: Monitors generalization on unseen data\nIf val_loss > train_loss consistently → overfitting.'),
    
    ('Training vs Validation Curves',
     'Plotted against epochs to visualize learning\nIdeal: Both curves decrease smoothly and converge\nWarning Signs: Validation plateau while training continues = overfitting\nMetric in Code: Used to detect model performance trends.'),
    
    ('Confusion Matrix (Optional)',
     'Shows true/false positives/negatives per class\nIdentifies which diseases are confused with others\nUseful for understanding model weaknesses.\nCould enhance current implementation.'),
    
    ('Per-Class Metrics (Optional)',
     'Precision: TP / (TP + FP) - accuracy for each disease\nRecall: TP / (TP + FN) - disease detection rate\nF1-Score: Harmonic mean of precision & recall\nUseful for imbalanced datasets.'),
]

for metric_title, metric_details in metrics:
    elements.append(Paragraph(f"<b>{metric_title}</b>", subheading_style))
    elements.append(Paragraph(metric_details.replace('\n', '<br/>'), body_style))
    elements.append(Spacer(1, 0.1*inch))

elements.append(Spacer(1, 0.3*inch))

# Section 14: Complete Workflow
elements.append(PageBreak())
elements.append(Paragraph("14. COMPLETE WORKFLOW SEQUENCE", heading_style))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("Training Phase (app.py):", subheading_style))
training_workflow = """
① Download PlantVillage Dataset from Kaggle (54,305 images)
② Extract and organize 38 disease classes
③ Create ImageDataGenerator with rescaling (÷255)
④ Split data: 80% training, 20% validation
⑤ Build Sequential CNN model (7 layers total)
⑥ Compile with Adam optimizer & cross-entropy loss
⑦ Train for 5 epochs with batch size 32
⑧ Monitor training/validation accuracy and loss
⑨ Evaluate on validation set
⑩ Plot performance curves (accuracy & loss)
⑪ Save model as plant_disease_model.h5 (HDF5 format)
⑫ Save class indices mapping as class_indices.json
"""
elements.append(Paragraph(training_workflow, body_style))
elements.append(Spacer(1, 0.15*inch))

elements.append(Paragraph("Inference Phase (main.py):", subheading_style))
inference_workflow = """
① Load pre-trained model from plant_disease_model.h5
② Load class indices mapping from class_indices.json
③ Initialize Streamlit web interface
④ Display title and file uploader widget
⑤ User uploads leaf image (JPG/JPEG/PNG)
⑥ Display uploaded image (150×150 px preview)
⑦ User clicks "Classify" button
⑧ Preprocess image: resize to 224×224, normalize to [0,1]
⑨ Run forward pass through CNN layers (~100-500ms)
⑩ Model outputs 38 probability values
⑪ Extract class with highest probability using argmax()
⑫ Map class index to disease name
⑬ Display prediction result to user
⑭ Ready for next image classification
"""
elements.append(Paragraph(inference_workflow, body_style))
elements.append(Spacer(1, 0.3*inch))

# Section 15: Future Enhancements
elements.append(PageBreak())
elements.append(Paragraph("15. FUTURE ENHANCEMENTS & IMPROVEMENTS", heading_style))
elements.append(Spacer(1, 0.1*inch))

enhancements = [
    ('Transfer Learning',
     'Use pre-trained ImageNet weights as starting point instead of random initialization.\nBenefit: Faster training, better accuracy with fewer data, leverages general image features.\nExample: Use MobileNetV2 or ResNet50 as backbone.'),
    
    ('Data Augmentation',
     'Apply random rotations, flips, zoom, brightness changes to training images.\nBenefit: Simulates different lighting/angles, improves robustness.\nCurrent Status: Infrastructure exists (ImageDataGenerator) but not fully utilized.'),
    
    ('Confidence Scores',
     'Display probability for predicted class and top-3 alternatives.\nBenefit: Users understand model confidence, easier to validate predictions.\nImplementation: Extract top_3 indices from prediction array.'),
    
    ('Model Validation Metrics',
     'Add confusion matrix, precision/recall per class, ROC curves.\nBenefit: Detailed performance analysis, identify problematic classes.\nTools: scikit-learn confusion_matrix, classification_report.'),
    
    ('Batch Prediction',
     'Allow uploading multiple images at once.\nBenefit: Process folders of farm images, generate reports.\nImplementation: Loop through multiple files, collect predictions.'),
    
    ('Mobile Deployment',
     'Convert to TensorFlow Lite for mobile apps (Android/iOS).\nBenefit: On-device inference, no internet required, faster response.\nTools: tf.lite.TFLiteConverter.'),
    
    ('API Development',
     'Create REST API (FastAPI/Flask) for programmatic access.\nBenefit: Integration with other systems, batch processing workflows.\nEndpoint: POST /predict with image → returns JSON with classification.'),
    
    ('Explainability',
     'Visualize which image regions most influence predictions (Grad-CAM, LIME).\nBenefit: Build trust, understand model reasoning, debugging.\nTools: tf-explain, grad-cam libraries.'),
]

for enh_title, enh_details in enhancements:
    elements.append(Paragraph(f"<b>{enh_title}</b>", subheading_style))
    elements.append(Paragraph(enh_details.replace('\n', '<br/>'), body_style))
    elements.append(Spacer(1, 0.1*inch))

elements.append(Spacer(1, 0.3*inch))

# Section 16: Common Issues & Troubleshooting
elements.append(PageBreak())
elements.append(Paragraph("16. COMMON ISSUES & TROUBLESHOOTING", heading_style))
elements.append(Spacer(1, 0.1*inch))

troubleshooting = [
    ('OutOfMemory Error',
     'Symptom: "CUDA out of memory" or "Cannot allocate memory"\nCauses: Batch size too large, model too large for GPU/RAM\nSolutions: Reduce batch size (try 16 or 8), use smaller model, enable gradient checkpointing.'),
    
    ('Model Accuracy Low (<70%)',
     'Symptom: Validation accuracy plateau at low value\nCauses: Insufficient training epochs, learning rate too high/low, bad data quality\nSolutions: Train longer (50+ epochs), adjust learning rate, verify image quality.'),
    
    ('Severe Overfitting',
     'Symptom: Train accuracy 95%+ but validation accuracy 50%\nCauses: Model too complex, insufficient data, no augmentation\nSolutions: Add dropout layers, use data augmentation, collect more data, reduce model size.'),
    
    ('Slow Inference',
     'Symptom: Predictions take 5+ seconds\nCauses: CPU inference (no GPU), model too large, disk I/O delays\nSolutions: Use GPU (CUDA), quantize model, optimize preprocessing.'),
    
    ('Image Preprocessing Errors',
     'Symptom: "Image size mismatch" or dimension errors\nCauses: Inconsistent image sizes, wrong color channels\nSolutions: Always resize to 224×224, ensure RGB format (not RGBA or grayscale).'),
    
    ('Class Index Mismatch',
     'Symptom: Predictions show wrong disease names\nCauses: class_indices.json doesn\'t match model training\nSolutions: Regenerate class_indices.json from train_generator.class_indices.'),
    
    ('Streamlit Not Loading',
     'Symptom: Page keeps loading, no error message\nCauses: Model loading takes too long, file path incorrect\nSolutions: Add caching (@st.cache), verify file paths, check console logs.'),
]

for issue_title, issue_details in troubleshooting:
    elements.append(Paragraph(f"<b>{issue_title}</b>", subheading_style))
    elements.append(Paragraph(issue_details.replace('\n', '<br/>'), body_style))
    elements.append(Spacer(1, 0.1*inch))

elements.append(Spacer(1, 0.3*inch))

# Conclusion
elements.append(PageBreak())
elements.append(Paragraph("17. CONCLUSION & KEY TAKEAWAYS", heading_style))
elements.append(Spacer(1, 0.1*inch))

conclusion = """
<b>Summary</b>

This Plant Disease Classifier demonstrates a complete end-to-end deep learning pipeline for practical 
agricultural applications. The system combines CNNs' powerful feature extraction capabilities with Streamlit's 
user-friendly interface for accessible plant disease diagnosis.

<b>Key Technical Achievements:</b>

✓ Implemented 7-layer CNN architecture optimized for 38-class disease classification
✓ Preprocesses images to consistent format ([0,1] normalized, 224×224 pixels)
✓ Uses Adam optimizer with categorical cross-entropy loss for stable training
✓ Achieves reasonable accuracy within 5 training epochs
✓ Deployed via Docker for reproducible, scalable deployment
✓ Provides web interface for non-technical users
✓ Leverages Kaggle's PlantVillage dataset with 54,305 high-quality images

<b>ML Concepts Demonstrated:</b>

• Convolutional Neural Networks and feature hierarchies
• Data preprocessing and normalization techniques
• Train/validation split and overfitting prevention
• Backpropagation and gradient descent optimization
• Model serialization and inference pipelines
• Web interface development for ML models
• Containerization for consistent deployment

<b>Real-World Applications:</b>

This system can be deployed on farms for:
✓ Early disease detection for crop management
✓ Reduce pesticide waste through targeted treatment
✓ Prevent large-scale crop failures
✓ Enable data-driven agricultural decisions
✓ Lower farming costs and increase yields

<b>Next Steps for Production Deployment:</b>

1. Collect more diverse training data from different farms/regions
2. Implement data augmentation for robustness
3. Use transfer learning to improve accuracy
4. Add confidence scores for predictions
5. Create mobile app for field use
6. Integrate with farm management systems
7. Implement user feedback loop for continuous improvement
8. Deploy on edge devices for offline operation

Understanding these technical details is crucial for maintaining, improving, and extending this 
system for practical agricultural applications.
"""
elements.append(Paragraph(conclusion, body_style))
elements.append(Spacer(1, 0.5*inch))

# Footer
footer_text = f"<i>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}<br/>Plant Disease Classifier - Technical Documentation</i>"
elements.append(Paragraph(footer_text, body_style))

# Build PDF
doc.build(elements)
print(f"PDF created successfully: {pdf_file}")
print(f"File location: {__file__.replace('generate_pdf.py', '')}Plant_Disease_Classifier_Technical_Documentation.pdf")
