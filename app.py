import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ============================================================
# LOAD AI MODEL
# ============================================================

MODEL_ID = "kimcomehome/plantvillage-vit-leaf-disease"

print("Loading Plant Disease Detection AI model...")
print("Please wait...")

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

model.eval()

print("AI model loaded successfully!")


# ============================================================
# DISEASE SOLUTIONS
# ============================================================

SOLUTIONS = {

    "Early blight": [
        "Remove infected leaves from the plant.",
        "Avoid watering the leaves directly.",
        "Keep the leaves dry and improve air circulation.",
        "Remove infected plant debris.",
        "Use an appropriate fungicide according to the product label and local agricultural guidance."
    ],

    "Late blight": [
        "Remove severely infected leaves and plant parts.",
        "Avoid overhead watering.",
        "Improve air circulation around the plant.",
        "Remove infected plant debris.",
        "Seek local agricultural advice because late blight can spread rapidly."
    ],

    "Bacterial spot": [
        "Remove severely infected leaves and fruits.",
        "Avoid overhead watering.",
        "Avoid handling plants when they are wet.",
        "Keep the growing area clean.",
        "Follow local agricultural recommendations for bacterial disease management."
    ],

    "Powdery mildew": [
        "Remove severely infected leaves.",
        "Improve air circulation around the plant.",
        "Avoid excessive humidity.",
        "Avoid excessive nitrogen fertilizer.",
        "Use an appropriate fungicide according to the product label."
    ],

    "Leaf Mold": [
        "Improve ventilation around the plant.",
        "Reduce humidity.",
        "Avoid overhead watering.",
        "Remove severely infected leaves.",
        "Use an appropriate fungicide according to local recommendations."
    ],

    "Septoria leaf spot": [
        "Remove infected lower leaves.",
        "Avoid splashing water onto the leaves.",
        "Improve air circulation.",
        "Remove infected plant debris.",
        "Use an appropriate fungicide according to the product label."
    ],

    "Target Spot": [
        "Remove severely infected leaves.",
        "Avoid overhead watering.",
        "Improve air circulation.",
        "Remove infected plant debris.",
        "Use an appropriate fungicide according to local recommendations."
    ],

    "Black rot": [
        "Remove infected leaves and fruits.",
        "Remove dead and infected plant material.",
        "Improve air circulation through pruning.",
        "Keep the growing area clean.",
        "Use an appropriate fungicide according to local recommendations."
    ],

    "Common rust": [
        "Monitor the plant regularly.",
        "Remove severely infected leaves where practical.",
        "Maintain good field sanitation.",
        "Improve air circulation.",
        "Use resistant varieties or recommended fungicides when appropriate."
    ],

    "Northern Leaf Blight": [
        "Remove infected crop debris.",
        "Practice crop rotation where possible.",
        "Use resistant varieties.",
        "Avoid prolonged leaf wetness.",
        "Use appropriate fungicide programs when recommended."
    ],

    "Cercospora leaf spot": [
        "Remove heavily infected leaves.",
        "Improve air circulation.",
        "Avoid excessive leaf wetness.",
        "Maintain good field sanitation.",
        "Use appropriate fungicides according to local recommendations."
    ],

    "Spider mites": [
        "Check the underside of leaves for mites.",
        "Wash foliage gently with water where appropriate.",
        "Reduce plant stress.",
        "Encourage beneficial insects where practical.",
        "Use an approved miticide only when necessary and according to its label."
    ],

    "Tomato Yellow Leaf Curl Virus": [
        "Remove severely infected plants.",
        "Control whiteflies using integrated pest management.",
        "Use healthy planting material.",
        "Remove weeds that may harbor whiteflies.",
        "Use resistant varieties where available."
    ],

    "Tomato mosaic virus": [
        "Remove severely infected plants.",
        "Disinfect gardening tools.",
        "Wash hands after handling infected plants.",
        "Remove weeds that may act as alternate hosts.",
        "Use resistant varieties where available."
    ],

    "Apple scab": [
        "Remove infected leaves and fallen debris.",
        "Improve air circulation by pruning.",
        "Avoid unnecessary overhead watering.",
        "Keep the area around the tree clean.",
        "Use an appropriate fungicide according to local guidance."
    ],

    "Cedar apple rust": [
        "Remove severely infected leaves.",
        "Improve air circulation.",
        "Remove infected plant debris.",
        "Monitor the plant regularly.",
        "Use a recommended fungicide according to the product label."
    ],

    "Bacterial spot": [
        "Remove severely infected leaves and fruits.",
        "Avoid overhead irrigation.",
        "Keep foliage dry.",
        "Use healthy planting material.",
        "Follow local agricultural recommendations."
    ]
}


# ============================================================
# FIND SOLUTION
# ============================================================

def get_solution(disease):

    disease_lower = disease.lower()

    for name, solutions in SOLUTIONS.items():

        if name.lower() in disease_lower:
            return solutions

    # General solution if exact disease is not listed

    return [
        "Remove severely infected plant material where practical.",
        "Avoid overhead watering.",
        "Improve air circulation around the plant.",
        "Keep the growing area clean.",
        "Consult a local agriculture expert for confirmation and treatment."
    ]


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image(image):

    if image is None:

        return """
# ❌ No Image Uploaded

Please upload a clear image of a plant leaf.
"""

    try:

        # Prepare image

        image = image.convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        # AI prediction

        with torch.no_grad():

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

        # Get highest prediction

        confidence, predicted_class = torch.max(
            probabilities,
            dim=0
        )

        disease = model.config.id2label[
            int(predicted_class)
        ]

        # Clean disease name

        disease = disease.replace("___", " - ")
        disease = disease.replace("_", " ")

        confidence = float(confidence) * 100


        # ====================================================
        # HEALTHY PLANT
        # ====================================================

        if "healthy" in disease.lower():

            return f"""
# 🌿 Healthy Plant

### 🌱 Prediction

**{disease}**

### 📊 AI Confidence

**{confidence:.2f}%**

### ✅ Result

The AI model considers this plant leaf **healthy**.

### 🌿 Recommended Care

- Continue regular monitoring.
- Maintain proper watering.
- Provide adequate sunlight.
- Maintain good air circulation.
- Remove damaged or dead leaves.

---

⚠️ This is an AI-based screening result and should not
replace professional agricultural diagnosis.
"""


        # ====================================================
        # DISEASE
        # ====================================================

        solutions = get_solution(disease)

        solution_text = ""

        for number, solution in enumerate(solutions, 1):

            solution_text += f"{number}. {solution}\n\n"


        return f"""
# ⚠️ Possible Disease Detected

### 🌱 Plant / Disease

**{disease}**

### 📊 AI Confidence

**{confidence:.2f}%**

### 🦠 Result

The AI model detected a **possible plant disease**.

### 🌿 Recommended Solution

{solution_text}


"""


    except Exception as error:

        return f"""
# ❌ Analysis Error

The image could not be analyzed.

Error:

{error}
"""


# ============================================================
# WEBSITE
# ============================================================

with gr.Blocks(
    title="Plant Disease Detection"
) as demo:

    gr.Markdown(
        """
# 🌱 Hii! Welcome to Plant Disease Detection

## AI-Based Plant Disease Detection & Solution System

Upload a clear image of a plant leaf.


"""
    )

    image = gr.Image(
        type="pil",
        label="📷 Upload Plant Leaf Image"
    )

    analyze_button = gr.Button(
        "🔍 Analyze Image",
        variant="primary"
    )

    result = gr.Markdown()

    analyze_button.click(
        fn=analyze_image,
        inputs=image,
        outputs=result
    )

    gr.Markdown(
        """
---


"""
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    demo.launch(
        share=True
    )
    