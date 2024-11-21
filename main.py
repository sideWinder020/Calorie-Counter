import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np
import io
import json
import requests
from icecream import ic
import PIL
import re
from typing import Any
import traceback
from fastapi.responses import JSONResponse





# Initialize FastAPI
app = FastAPI()


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAPIKEY")

# Serve static files
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Define model paths and URLs
MODEL_DIR = "Model"
os.makedirs(MODEL_DIR, exist_ok=True)

INCEPTION_MODEL_PATH = os.path.join(MODEL_DIR, "fine_tuned_inceptionV3_final_96.keras")
VGG_MODEL_PATH = os.path.join(MODEL_DIR, "complete_vgg16_model.h5")

INCEPTION_MODEL_URL = os.getenv("INCEPTION_MODEL_URL")
VGG_MODEL_URL = os.getenv("VGG_MODEL_URL")


import re
import json

def to_camel_case(snake_str):
    """
    Converts a string to camelCase.
    """
    words = re.split(r'[ _]', snake_str)
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

def convert_keys_to_camel_case(data):
    """
    Recursively converts all dictionary keys to camelCase.
    """
    if isinstance(data, dict):
        return {to_camel_case(key): convert_keys_to_camel_case(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_keys_to_camel_case(item) for item in data]
    else:
        return data




@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    """
    Serve the chat homepage.
    """
    try:
       
        return FileResponse("static/calorie.html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Homepage not found.")
    

img_width, img_height = 299, 299


with open("class_indices.json","r") as handle:
    class_indices = json.load(handle)


with open("class_indices_english.json","r") as handle:
    class_indices_english = json.load(handle)



def download_model(url, destination):
    """
    Downloads a model file from a given URL if it does not already exist locally.
    """
    if not os.path.exists(destination):
        print(f"Downloading model from {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Model downloaded and saved to {destination}")
        else:
            raise Exception(f"Failed to download model from {url}, status code: {response.status_code}")
    else:
        print(f"Model already exists at {destination}")

# Download models at startup
try:
    download_model(INCEPTION_MODEL_URL, INCEPTION_MODEL_PATH)
    download_model(VGG_MODEL_URL, VGG_MODEL_PATH)
except Exception as e:
    print(f"Error during model download: {e}")
    raise

# Load Models
inception_model = load_model(INCEPTION_MODEL_PATH, compile=False)
vgg_model = load_model(VGG_MODEL_PATH, compile=False)

def predict_image(img_data, model, target_size=(224, 224), class_indices=None):
    """
    Predicts the class label of an image using the provided model.
    """
    img = load_img(io.BytesIO(img_data), target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    if class_indices:
        label_map = {v: k for k, v in class_indices.items()}
        predicted_label = label_map.get(predicted_index, "Unknown")
    else:
        predicted_label = f"Class {predicted_index}"

    return predicted_label, confidence



def get_nutritional_info(food_item):
    """
    Calls OpenAI API to retrieve nutritional information for the predicted food item.
    """
    prompt = f"Provide the nutritional information for a typical serving of {food_item} in JSON format. Include only Calories, Total Fat, Cholesterol, and Protein."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing nutritional information in JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.2
        )
        
     
        nutritional_info_text = response['choices'][0]['message']['content'].strip()
        # print(f"Raw nutritional info response: {nutritional_info_text}")
        
        try:
            nutritional_info = json.loads(nutritional_info_text)
        except json.JSONDecodeError as json_error:
            print("Error parsing JSON:", json_error)
            return {"error": "Failed to parse nutritional information as JSON."}


        nutritional_info = convert_keys_to_camel_case(nutritional_info)

        if nutritional_info:
            key = list(nutritional_info.keys())[-1]
            if key in nutritional_info:
                return nutritional_info[key]
            else:
                print("Key not found in nutritional info:", key)
                return {"error": "Key not found in nutritional information."}
        else:
            return {"error": "Empty nutritional information received."}

    except openai.error.OpenAIError as api_error:
        print("Error with OpenAI API request:", api_error)
        return {"error": "Failed to retrieve data from OpenAI API."}

    except Exception as general_error:
        print("Unexpected error:", general_error)
        return {"error": "An unexpected error occurred."}


@app.post("/predict-food-nutrition")
async def predict_food_nutrition(file: UploadFile = File(...)) -> Any:
    """
    Endpoint to predict food item and get nutritional information.
    """
    try:
        # Read the uploaded image
        img_data = await file.read()
        if not img_data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty or invalid.")

        # Perform predictions using the models
        inception_predicted_label, inception_confidence = predict_image(
            img_data, inception_model, target_size=(img_width, img_height), class_indices=class_indices
        )
        vgg_predicted_label, vgg_confidence = predict_image(
            img_data, vgg_model, target_size=(224, 224), class_indices=class_indices_english
        )

        # Convert numpy.float32 to native Python float for serialization
        inception_confidence = float(inception_confidence)
        vgg_confidence = float(vgg_confidence)

        # Log predictions for debugging
        ic(inception_predicted_label, inception_confidence)
        ic(vgg_predicted_label, vgg_confidence)

        # Choose the final prediction based on confidence
        if inception_confidence > vgg_confidence:
            final_predicted_label = inception_predicted_label
            final_confidence = inception_confidence
        else:
            final_predicted_label = vgg_predicted_label
            final_confidence = vgg_confidence

        # Handle low-confidence predictions
        if final_confidence * 100 <= 25:
            return JSONResponse(
                status_code=200,
                content={
                    "predictedFoodItem": "Probably not a food item, try again.",
                    "confidence": final_confidence,
                    "nutritionalInfo": {"none": None},
                },
            )

        # Get nutritional information using OpenAI API
        nutritional_info = get_nutritional_info(final_predicted_label)
        # final_predicted_label = to_camel_case(final_predicted_label)


        final_predicted_label = " ".join(final_predicted_label.split("_")).title()

        # Prepare and return the response
        response_output = {
            "predictedFoodItem": final_predicted_label,
            "confidence": final_confidence,
            "nutritionalInfo": nutritional_info,
        }
        ic(response_output)
        return JSONResponse(status_code=200, content=response_output)

    except HTTPException as http_err:
        # Handle HTTP exceptions explicitly
        raise http_err

    except Exception as err:
        # Log unexpected errors and return a generic error response
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred while processing the request."},
        )



class FoodItemRequest(BaseModel):
    foodItem: str


@app.post("/get-nutritional-info-user-prompt")
async def get_nutritional_info_endpoint(request: FoodItemRequest):
    """
    Endpoint to fetch nutritional information based on user input.
    """


    food_item = request.foodItem

    # print(f"Received food item: {food_item}")
    if not food_item:
        raise HTTPException(status_code=400, detail="Food item is required.")

    nutritional_info = get_nutritional_info(food_item)
    ordered_nutritional_info = {"Food Name": food_item}
    ordered_nutritional_info.update(nutritional_info)
    ordered_nutritional_info = convert_keys_to_camel_case(ordered_nutritional_info)
    ordered_nutritional_info["foodName"] = " ".join(ordered_nutritional_info["foodName"].split(" ")).title()
    ic(ordered_nutritional_info)

    if "error" in ordered_nutritional_info:
        raise HTTPException(status_code=500, detail=ordered_nutritional_info["error"])

    return {"nutritionalInfo": ordered_nutritional_info}