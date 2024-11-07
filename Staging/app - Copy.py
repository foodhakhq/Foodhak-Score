import vertexai
import requests
import json
import time
import os
from flask import Flask, jsonify, request
from flask import Flask, request, Response
from dotenv import load_dotenv
from vertexai.preview.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
from google.api_core.exceptions import ServiceUnavailable, BadGateway
from collections import OrderedDict
from werkzeug.exceptions import HTTPException  
# Load environment variables from the .env file
load_dotenv()
app = Flask(__name__)

# Get the API key from the environment variables
STAGING_API_KEY = os.getenv('STAGING_API_KEY')

def validate_api_key(auth_header):
    token = auth_header.split(" ")[1]  # Split "Bearer <token>"
    if token != STAGING_API_KEY:
        return False
    return True

# Initialize the Gemini model
def initialize_gemini_model():
    vertexai.init(project="central-muse-388319", location="us-central1")
    return GenerativeModel("gemini-1.5-pro-001")

# Fetch product information from OpenFoodFacts API
def fetch_product_info(barcode):
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('product', {})
    else:
        return None

# Use LLM to grade the product based on the grading criteria
# Use LLM to grade the product based on the grading criteria
def llm_grade_product(model, product_info, retries=3, delay=10):
    product_name = product_info.get('product_name', '').strip()
    
    fallback_product_names = [
        product_info.get('product_name_en', '').strip(),
        product_info.get('product_name_es', '').strip(),
        product_info.get('product_name_fr', '').strip(),
        product_info.get('product_name_pl', '').strip(),
        product_info.get('product_name_pt', '').strip(),
        product_info.get('product_name_zh', '').strip(),
        product_info.get('product_name_de', '').strip(),
        product_info.get('product_name_it', '').strip(),
        product_info.get('product_name_ru', '').strip(),
        product_info.get('product_name_no', '').strip(),
        product_info.get('product_name_fi', '').strip()
    ]
    
    product_name = product_name if product_name else next((name for name in fallback_product_names if name), "Unnamed")
    
    product_image_url = product_info.get('image_url', 'Image URL not available')

    prompt = f"""
    Grade the healthiness of the following product based on the specific grading criteria provided below. Follow the criteria exactly and calculate a score on a scale of 1 to 10, with a higher score indicating a healthier product. Provide the score and a brief explanation of how each criterion was applied.

    Grading Criteria:
    1. Nutrient Quality: 3 points

    This component evaluates the overall nutritional profile of the product by considering both positive and negative factors.

    Positive Factors:
    Fiber: A high fiber content is beneficial for digestion and can help maintain a healthy weight.
    Protein: Important for building and repairing tissues, protein is essential for muscle health and overall bodily functions.
    Vitamins and Minerals: The presence of essential vitamins and minerals (like vitamins A, C, D, calcium, iron, etc.) contributes to overall health and well-being.
    
    Negative Factors:
    Added Sugars: High amounts of added sugars can lead to various health issues, including obesity, diabetes, and heart disease.
    Saturated Fat: High saturated fat content is linked to heart disease and should be limited in a healthy diet.
    Medium Salt: Excess sodium intake is associated with high blood pressure and cardiovascular problems.

    Scoring:

    3 points: If the product has a balanced profile with high positive nutrients and minimal negatives.
    2 points: If the product has some negative factors but is still somewhat nutritious.
    1 point: For products that have poor nutritional value with high negatives or very low positives.


    2. Ingredient Quality: 3 points

    This component assesses the quality of the ingredients used in the product.

    Clean Ingredients: This refers to whole, recognizable ingredients without artificial additives or preservatives. Products with few ingredients that are easily identifiable (like whole grains, fruits, vegetables, etc.) score higher.
    Natural Additives: Some products may contain natural preservatives (like salt or vinegar) that are acceptable, while others might have synthetic preservatives that are less desirable.

    Scoring:

    3 points: If the product contains no artificial ingredients or preservatives.
    2 points: If the product has minimal additives or preservatives but is mostly clean.
    1 point: For products with multiple artificial ingredients or preservatives.

    3. Whole Foods Content: 3 points

    This component measures how much of the product is made from whole, unprocessed foods.

    Whole Foods: These are foods that are minimally processed and remain close to their natural state, such as fruits, vegetables, whole grains, nuts, and seeds.
    Processed Foods: Foods that have been significantly altered from their natural state, often containing added sugars, preservatives, and other artificial ingredients.

    Scoring:

    3 points: For products that contain 75% whole food ingredients.
    2 points: If the product has a mix of 50% whole and 50% processed ingredients.
    1 point: If the product has a mix of <50% whole and >50% processed ingredients..

    4. Processing Level (NOVA Score): 1 point

    This component measures how processed the product is, using the NOVA classification system.

    NOVA 1: Minimally processed foods (e.g., fresh fruits and vegetables).
    NOVA 2: Processed culinary ingredients (e.g., oils, butter).
    NOVA 3: Processed foods (e.g., canned vegetables with salt).
    NOVA 4: Ultra-processed foods (e.g., sugary drinks, snacks with artificial ingredients).

    Scoring:

    1 point: For minimally processed foods (NOVA 1).
    0.75 points: For NOVA 2.
    0.5 points: For NOVA 3.
    0.25 points: For NOVA 4.

    Product Data:
    - Product Name: {product_info.get('product_name', 'Unnamed')}
    - Ingredients: {product_info.get('ingredients_text_en', 'Not provided')}
    - Additives: {product_info.get('additives_tags', [])}
    - NOVA Group: {product_info.get('nova_group', 'Unknown')}
    - Nutrients: {product_info.get('nutriments', {})}
    - Number of Additives: {product_info.get('additives_n', 0)}
    - Ingredients from Palm Oil: {product_info.get('ingredients_from_palm_oil_n', 0)}
    - Ingredient Origins: {product_info.get('origins', 'Unknown')}
    - Nutrient Levels: {product_info.get('nutrient_levels_tags', [])}

    Calculate the score based on the criteria and provide a brief explanation. Make sure to refer to the product by its name.
    
    ### Response Format:
    {{
      "Nutrient Quality": {{
        "Score": [Your score here],
        "Explanation": "[Your explanation here]"
      }},
      "Processing Level": {{
        "Score": [Your score here],
        "Explanation": "[Your explanation here]"
      }},
      "Ingredient Quality": {{
        "Score": [Your score here],
        "Explanation": "[Your explanation here]"
      }},
      "Whole Foods Content": {{
        "Score": [Your score here],
        "Explanation": "[Your explanation here]"
      }},
      "Total Score": {{
        "Score": [Your score here],
        "Explanation": "[Your explanation with reasoning here]"
      }},
      "Product Name": "{product_name}",
      "Product Image URL": "{product_image_url}"
    }}
    """
    
    for attempt in range(retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 1056,
                    "temperature": 0,
                    "top_p": 1
                },
                stream=False
            )
            if response.candidates and response.candidates[0].content.parts:
                grading_output = response.candidates[0].content.parts[0].text
                grading_output = grading_output.replace('*', '').replace("```", "").replace("json", "")
                try:
                    parsed_response = json.loads(grading_output)
                    # Maintain order using OrderedDict
                    ordered_response = OrderedDict([
                        ("Nutrient Quality", parsed_response.get("Nutrient Quality", {})),
                        ("Processing Level", parsed_response.get("Processing Level", {})),
                        ("Ingredient Quality", parsed_response.get("Ingredient Quality", {})),
                        ("Whole Foods Content", parsed_response.get("Whole Foods Content", {})),
                        ("Total Score", parsed_response.get("Total Score", {})),
                        ("Product Name", product_name),
                        ("Product Image URL", product_image_url)
                    ])
                    
                    return ordered_response
                except json.JSONDecodeError:
                    time.sleep(delay)
                    continue  # Retry if JSON parsing fails
            else:
                time.sleep(delay)
                continue
        except (ServiceUnavailable, BadGateway):
            time.sleep(delay)

    return {"error": "Exceeded retry limit, could not generate response."}

# API endpoint for grading a product
@app.route('/foodhak-score', methods=['POST'])
def grade_product():
    max_attempts = 3  # Number of times to retry the entire operation
    for attempt in range(max_attempts):
        try:
            auth_header = request.headers.get('Authorization')
            # Validate Authorization header
            if not auth_header or not auth_header.startswith("Bearer ") or not validate_api_key(auth_header):
                # Return immediately for client errors
                return jsonify({"error": "Invalid or missing API key."}), 401

            data = request.json
            if not data:
                # Return immediately for client errors
                return jsonify({"error": "Invalid JSON payload."}), 400

            barcode = data.get('barcode')
            if not barcode:
                # Return immediately for client errors
                return jsonify({"error": "barcode is required"}), 400

            model = initialize_gemini_model()
            product_info = fetch_product_info(barcode)
            # Handle case where product is not found (FS-404)
            if not product_info:
                # Return immediately for client errors
                return jsonify({
                    "error": "FS-404",
                    "message": "Product Not Found",
                    "description": "The product with the provided barcode could not be found in our database. Please ensure the barcode is correct or try scanning another product."
                }), 404

            # Required fields for reliable scoring, but no longer a strict requirement
            required_fields = ['ingredients_text_en', 'nova_group', 'nutriments']
            missing_fields = [field for field in required_fields if not product_info.get(field)]

            # Grade the product using the LLM
            grading = llm_grade_product(model, product_info)

            # Check if grading failed
            if not grading or "error" in grading:
                error_message = grading.get("error", "Unknown error") if grading else "No response from grading function"
                print(f"Grading failed on attempt {attempt + 1}: {error_message}")
                if attempt < max_attempts - 1:
                    time.sleep(2)  # Wait before retrying
                    continue
                else:
                    return jsonify({
                        "error": "FS-500",
                        "message": "Internal Server Error",
                        "description": "An unexpected error occurred while trying to calculate the score. Please try again later."
                    }), 500

            # Organize and return the successful grading response
            ordered_response = OrderedDict([
                ("Nutrient Quality", grading.get("Nutrient Quality", {})),
                ("Processing Level", grading.get("Processing Level", {})),
                ("Ingredient Quality", grading.get("Ingredient Quality", {})),
                ("Whole Foods Content", grading.get("Whole Foods Content", {})),
                ("Total Score", grading.get("Total Score", {})),
                ("Product Name", grading.get("Product Name", "Unnamed")),
                ("Product Image URL", grading.get("Product Image URL", "Image URL not available"))
            ])

            if missing_fields:
                ordered_response["error"] = {
                    "code": "FS-204",
                    "message": "Insufficient Data for Scoring",
                    "description": f"Not enough nutritional data is available to calculate a reliable health score for this product. Please check the explanation to see what information is missing."
                }
                # Return 206 status code with the response body
                return Response(json.dumps(ordered_response, separators=(',', ':'), ensure_ascii=False), status=206, mimetype='application/json')

            # Return the ordered response with json.dumps to preserve order
            return Response(json.dumps(ordered_response, separators=(',', ':'), ensure_ascii=False), mimetype='application/json')
            
        except HTTPException as http_exc:
            # For HTTP exceptions, return the response immediately
            return jsonify({"error": str(http_exc), "message": http_exc.name}), http_exc.code

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with exception: {e}")
            if attempt < max_attempts - 1:
                time.sleep(1)  # Wait before retrying
                continue
            else:
                return jsonify({
                    "error": "FS-500",
                    "message": "Internal Server Error",
                    "description": "An unexpected error occurred while trying to process the request. Please try again later."
                }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)