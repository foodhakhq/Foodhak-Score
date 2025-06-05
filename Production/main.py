import os
import json
import logging
import aiohttp
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Depends
from pydantic import BaseModel
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi import status
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global Variables
claude_client = None

session = None

# Environment Variables
API_KEY = os.getenv("API_KEY")
PRODUCTION_OPENAI_API_KEY = os.getenv("PRODUCTION_OPENAI_API_KEY")


async def safe_send(websocket: WebSocket, message: dict) -> bool:
    try:
        await websocket.send_json(message)
    except WebSocketDisconnect:
        logging.warning("WebSocket disconnected while sending message.")
        return False
    except RuntimeError as e:
        logging.warning(f"Runtime error while sending message: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error while sending message: {e}")
        return False
    return True


def get_openai_client():
    api_key = os.getenv("PRODUCTION_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("PRODUCTION_OPENAI_API_KEY not found in environment variables")
    client = OpenAI(api_key=api_key)
    return client


_sentinel = object()  # Unique sentinel value


def safe_next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _sentinel


class AsyncIteratorWrapper:
    """Wrap a synchronous iterator in an asynchronous one using a sentinel to avoid StopIteration issues."""

    def __init__(self, iterator):
        self._iterator = iterator

    def __aiter__(self):
        return self

    async def __anext__(self):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, safe_next, self._iterator)
        if result is _sentinel:
            raise StopAsyncIteration
        return result


# Verify API Key
async def verify_token(authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    token = authorization.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")


# Pydantic Schema for API Request
class ProductRequest(BaseModel):
    barcode: str


# Get or Initialize aiohttp Session
async def get_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()
    return session


# Fetch Product Info from OpenFoodFacts
async def fetch_product_info(barcode):
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    session = await get_session()
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            return data.get('product', {})
    return None


# Stream Product Grading using Claude AI
async def llm_grade_product(client, product_info, websocket: WebSocket):
    """Use LLM to grade the product based on the grading criteria and stream the response."""

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
    product_name_fallback = product_name if product_name else next((name for name in fallback_product_names if name),
                                                                   "Unnamed")
    product_image_url = product_info.get('image_url', 'Image URL not available')
    # Check if there's enough data to analyze the product
    missing_data = []
    essential_keys = {
        'ingredients_text_en': 'Ingredients',
        'additives_tags': 'Additives',
        'nova_group': 'NOVA Group',
        'nutriments': 'Nutrients',
    }

    for key, desc in essential_keys.items():
        value = product_info.get(key)
        if value is None or (isinstance(value, (list, dict)) and not value):
            missing_data.append(desc)

    # If too many essential data points are missing, return an error
    if len(missing_data) >= 3:  # If 3 or more essential data points are missing
        await websocket.send_json({
            "error": {
                "code": 422,
                "message": "Insufficient product data for analysis",
                "description": f"Cannot analyze product due to missing critical data: {', '.join(missing_data)}",
                "data": {
                    "product_name": product_name_fallback,
                    "image_url": product_image_url,
                    "barcode": product_info.get('code', 'Unknown')
                }
            }
        })
        await websocket.close(code=1003, reason="Insufficient product data")
        return
    system_prompt = [{
        "type": "text",
        "text": f"""
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

                            Calculate the score based on the criteria and provide a brief explanation. Make sure to refer to the product by its name.
                """,
        "cache_control": {"type": "ephemeral"}}]
    user_prompt = [{"role": "user", "content":
        f"""
                            Product Data:
                            - Product Name: {product_name_fallback}
                            - product_image_url: {product_info.get('image_url', 'Image URL not available')}
                            - Ingredients: {product_info.get('ingredients_text_en', 'Not provided')}
                            - Additives: {product_info.get('additives_tags', [])}
                            - NOVA Group: {product_info.get('nova_group', 'Unknown')}
                            - Nutrients: {product_info.get('nutriments', {})}
                            - Number of Additives: {product_info.get('additives_n', 0)}
                            - Ingredients from Palm Oil: {product_info.get('ingredients_from_palm_oil_n', 0)}
                            - Ingredient Origins: {product_info.get('origins', 'Unknown')}
                            - Nutrient Levels: {product_info.get('nutrient_levels_tags', [])}

                            Calculate the score based on the criteria and provide a brief explanation. Make sure to refer to the product by its name.

                            Please provide SEVEN separate JSON blocks (each on its own line followed by a blank line).

                            1) "Nutrient Quality"  
                            2) "Processing Level"  
                            3) "Ingredient Quality"  
                            4) "Whole Foods Content"  
                            5) "Total Score"  
                            6) "Product Name"  
                            7) "Product Image URL"

                            **Do not** wrap them all in a single JSON object—**each key** should be in its **own** JSON object.  
                            No extra text, commentary, or explanation outside these JSON blocks.

                            ### Example Format

                            1) For Nutrient Quality:
                            ```
                            {{
                              "Nutrient Quality": {{
                                "Explanation": "[Your explanation here]",
                                "Score": [Your score here],
                                "title": Nutrient Quality
                              }}
                            }}

                            ```

                            2) For Processing Level:
                            ```
                            {{
                              "Processing Level": {{
                                "Explanation": "[Your explanation here]",
                                "Score": [Your score here],
                                "title": Processing Level
                              }}
                            }}

                            ```

                            3) For Ingredient Quality:
                            ```
                            {{
                              "Ingredient Quality": {{
                                "Explanation": "[Your explanation here]",
                                "Score": [Your score here],
                                "title": Ingredient Quality
                              }}
                            }}

                            ```

                            4) For Whole Foods Content:
                            ```
                            {{
                              "Whole Foods Content": {{
                                "Explanation": "[Your explanation here]",
                                "Score": [Your score here],
                                "title": Whole Foods Content
                              }}
                            }}

                            ```

                            5) For Total Score:
                            ```
                            {{
                              "Total Score": {{
                                "Explanation": "[Based on the data you computed previously, determine whether the product can be considered nutritious or healthy.
Explain how each factor (Nutrient Quality, Processing Level, Ingredient Quality, Whole Foods Content) influenced your decision.
Provide a concise, user-friendly summary that someone could share with friends or on social media.]",
                                "Score": [Your score here],
                                "title": Total Score
                              }}
                            }}

                            ```

                            6) For Product Name:
                            ```
                            {{
                              "Product Name": "{product_name_fallback}"
                            }}

                            ```

                            7) For Product Image URL:
                            ```
                            {{
                              "Product Image URL": "{product_info.get('image_url', 'Image URL not available')}"
                            }}

                            ```

                            **Important**:  
                            1. Output **exactly** these seven **separate** JSON blocks, **one per line** or separated by blank lines.  
                            2. **No extra text** before, between, or after them.  
                            3. **Do not** nest them in a parent JSON array or object.  
                            4. **Do not** provide any additional commentary or keys.

                            ---
                            """}]
    system_prompt_openai = f"""
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

                            Calculate the score based on the criteria and provide a brief explanation. Make sure to refer to the product by its name.
                """
    user_prompt_openai = f"""
                            Product Data:
                            - Product Name: {product_name_fallback}
                            - product_image_url: {product_info.get('image_url', 'Image URL not available')}
                            - Ingredients: {product_info.get('ingredients_text_en', 'Not provided')}
                            - Additives: {product_info.get('additives_tags', [])}
                            - NOVA Group: {product_info.get('nova_group', 'Unknown')}
                            - Nutrients: {product_info.get('nutriments', {})}
                            - Number of Additives: {product_info.get('additives_n', 0)}
                            - Ingredients from Palm Oil: {product_info.get('ingredients_from_palm_oil_n', 0)}
                            - Ingredient Origins: {product_info.get('origins', 'Unknown')}
                            - Nutrient Levels: {product_info.get('nutrient_levels_tags', [])}

                            Calculate the score based on the criteria and provide a brief explanation. Make sure to refer to the product by its name.

                            Please provide SEVEN separate JSON blocks (each on its own line followed by a blank line).

                            1) "Nutrient Quality"  
                            2) "Processing Level"  
                            3) "Ingredient Quality"  
                            4) "Whole Foods Content"  
                            5) "Total Score"  
                            6) "Product Name"  
                            7) "Product Image URL"

                            **Do not** wrap them all in a single JSON object—**each key** should be in its **own** JSON object.  
                            No extra text, commentary, or explanation outside these JSON blocks.

                            ### Example Format

                            1) For Nutrient Quality:
                            ```
                            {{
                              "Nutrient Quality": {{
                                "Explanation": "[Your explanation here]",
                                "Score": [Your score here],
                                "title": Nutrient Quality
                              }}
                            }}

                            ```

                            2) For Processing Level:
                            ```
                            {{
                              "Processing Level": {{
                                "Explanation": "[Your explanation here]",
                                "Score": [Your score here],
                                "title": Processing Level
                              }}
                            }}

                            ```

                            3) For Ingredient Quality:
                            ```
                            {{
                              "Ingredient Quality": {{
                                "Explanation": "[Your explanation here]",
                                "Score": [Your score here],
                                "title": Ingredient Quality
                              }}
                            }}

                            ```

                            4) For Whole Foods Content:
                            ```
                            {{
                              "Whole Foods Content": {{
                                "Explanation": "[Your explanation here]",
                                "Score": [Your score here],
                                "title": Whole Foods Content
                              }}
                            }}

                            ```

                            5) For Total Score:
                            ```
                            {{
                              "Total Score": {{
                                "Explanation": "[Based on the data you computed previously, determine whether the product can be considered nutritious or healthy.
Explain how each factor (Nutrient Quality, Processing Level, Ingredient Quality, Whole Foods Content) influenced your decision.
Provide a concise, user-friendly summary that someone could share with friends or on social media.]",
                                "Score": [Your score here],
                                "title": Total Score
                              }}
                            }}

                            ```

                            6) For Product Name:
                            ```
                            {{
                              "Product Name": "{product_name_fallback}"
                            }}

                            ```

                            7) For Product Image URL:
                            ```
                            {{
                              "Product Image URL": "{product_info.get('image_url', 'Image URL not available')}"
                            }}

                            ```

                            **Important**:  
                            1. Output **exactly** these seven **separate** JSON blocks, **one per line** or separated by blank lines.  
                            2. **No extra text** before, between, or after them.  
                            3. **Do not** nest them in a parent JSON array or object.  
                            4. **Do not** provide any additional commentary or keys.

                            ---
    """
    try:
        current_json = ""  # Buffer for building current JSON object
        brace_count = 0  # Track nested braces
        message_id = None

        async with client.beta.prompt_caching.messages.stream(
                model="claude-3-7-sonnet-20250219",
                max_tokens=5012,
                temperature=0,
                system=system_prompt,
                messages=user_prompt,
        ) as stream:
            async for chunk in stream:
                if hasattr(chunk, "type"):
                    if chunk.type == "message_start":
                        message_id = chunk.message.id

                    elif chunk.type == "content_block_delta" and hasattr(chunk.delta, "text"):
                        text = chunk.delta.text

                        # Process the text character by character
                        for char in text:
                            if char == '{':
                                brace_count += 1
                                current_json += char
                            elif char == '}':
                                brace_count -= 1
                                current_json += char

                                # If we've completed a JSON object (brace_count back to 0)
                                if brace_count == 0 and current_json:
                                    try:
                                        # Parse and validate the JSON object
                                        json_obj = json.loads(current_json)

                                        # Send the complete JSON object to the client
                                        await websocket.send_json({
                                            "message_id": message_id,
                                            "type": "json_object",
                                            "data": json_obj
                                        })

                                        # Reset the current JSON buffer
                                        current_json = ""
                                    except json.JSONDecodeError:
                                        # If JSON is invalid, log it and continue
                                        logging.warning(f"Invalid JSON object: {current_json}")
                                        current_json = ""
                            else:
                                if brace_count > 0:
                                    current_json += char

                    elif chunk.type == "message_stop":
                        # Send any remaining valid JSON
                        if current_json:
                            try:
                                json_obj = json.loads(current_json)
                                await websocket.send_json({
                                    "type": "json_object",
                                    "data": json_obj,
                                    "message_id": message_id
                                })
                            except json.JSONDecodeError:
                                logging.warning(f"Invalid final JSON object: {current_json}")

                        # Send completion message
                        await websocket.send_json({
                            "type": "message_stop",
                            "message_id": message_id,
                            "data": "Streaming complete"
                        })

                        await websocket.close(code=1000, reason="Streaming completed successfully")
                        break

    except Exception as anthropic_error:
        # Log the error to see what args look like
        logging.error(f"Anthropic error args: {anthropic_error.args}")
        error_data = anthropic_error.args[0] if anthropic_error.args else {}

        # If error_data is not a dict, try to parse it as JSON
        if not isinstance(error_data, dict):
            try:
                error_data = json.loads(error_data)
            except Exception:
                error_data = {}

        # Also check the error message string as a fallback
        error_str = str(anthropic_error)

        if error_data.get("error", {}).get("type") == "overloaded_error" or "overloaded_error" in error_str:
            logging.warning("Anthropic Claude is overloaded. Falling back to OpenAI...")

            openai_client = get_openai_client()

            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt_openai},
                        {"role": "user", "content": user_prompt_openai}
                    ],
                    stream=True,
                    response_format={"type": "json_object"}
                )

                buffer_fallback = ""
                async for event in AsyncIteratorWrapper(response):
                    delta = event.choices[0].delta

                    if message_id is None and hasattr(event, 'id'):
                        message_id = event.id
                        if not await safe_send(websocket, {
                            "message_id": message_id,
                            "type": "message_start",
                            "data": "Streaming started."
                        }):
                            break

                    if hasattr(delta, 'content') and delta.content:
                        buffer_fallback += delta.content

                        while buffer_fallback:
                            try:
                                json_obj, index = json.JSONDecoder().raw_decode(buffer_fallback)
                                if not await safe_send(websocket, {
                                    "message_id": message_id,
                                    "type": "json_object",
                                    "data": json_obj
                                }):
                                    break
                                buffer_fallback = buffer_fallback[index:].lstrip()
                            except json.JSONDecodeError:
                                break
                        if not await safe_send(websocket, {
                            "message_id": message_id,
                            "type": "content_block_delta",
                            "data": delta.content
                        }):
                            break

                    if event.choices[0].finish_reason is not None:
                        if not await safe_send(websocket, {
                            "message_id": message_id,
                            "type": "message_stop",
                            "data": "Streaming complete."
                        }):
                            break
                        await websocket.close(code=1000, reason="Streaming completed successfully.")
                        logging.info(f"WebSocket connection closed after streaming for message_id: {message_id}")
                        break

            except Exception as openai_error:
                logging.error(f"GPT fallback error: {openai_error}")
                await safe_send(websocket, {
                    "message_id": message_id,
                    "type": "error",
                    "data": str(openai_error)
                })

        else:
            await safe_send(websocket, {
                "error": "Claude and GPT processing failed.",
                "description": error_str
            })


# WebSocket Endpoint for Real-Time Grading with barcode
@app.websocket("/ws/grade/{barcode:path}")
async def websocket_grade(websocket: WebSocket, barcode: str):
    await websocket.accept()

    try:
        # Check if a barcode was provided (barcode is a path parameter so it should always be present,
        # but we handle the case where it might be an empty string)
        if not barcode.strip():
            await websocket.send_json({
                "error": {
                    "code": 400,
                    "message": "Barcode is required.",
                    "description": "Please provide a barcode."
                }
            })
            await websocket.close(code=1008, reason="Missing barcode.")
            return

        # Fetch product details using the barcode
        product_details = await fetch_product_info(barcode)
        if not product_details:
            await websocket.send_json({
                "error": {
                    "code": 404,
                    "message": f"No product details found for the barcode: {barcode}.",
                    "description": "Please verify the barcode and try again."
                }
            })
            await websocket.close(code=1003, reason="Product not found.")
            return

        # Stream response with Claude AI (now includes message_id)
        await llm_grade_product(claude_client, product_details, websocket)

    except WebSocketDisconnect:
        logging.info(f"WebSocket client disconnected for barcode: {barcode}")

    except Exception as e:
        logging.error(f"Unexpected error in WebSocket: {e}")
        await websocket.send_json({
            "error": {
                "code": 500,
                "message": "Unexpected error occurred during WebSocket communication.",
                "description": str(e)
            }
        })
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Unexpected server error.")


# REST API Endpoint to Trigger WebSocket
@app.post("/grade-product")
async def grade_product(request: ProductRequest, token_valid: str = Depends(verify_token)):
    barcode = request.barcode

    # Check for empty or whitespace-only barcode
    if not barcode:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": 400,
                    "message": "Barcode is required and cannot be empty.",
                    "description": "Please provide a valid barcode."
                }
            }
        )

    # Fetch product details using the barcode
    product_details = await fetch_product_info(barcode)
    if not product_details:
        # Return a structured error response
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": 404,
                    "message": f"No product details found for the barcode: {barcode}.",
                    "description": "Please verify the barcode and try again."
                }
            }
        )

    # Check if there's enough data to analyze the product
    missing_data = []
    essential_keys = {
        'ingredients_text_en': 'Ingredients',
        'additives_tags': 'Additives',
        'nova_group': 'NOVA Group',
        'nutriments': 'Nutrients',
    }

    for key, desc in essential_keys.items():
        value = product_details.get(key)
        if value is None or (isinstance(value, (list, dict)) and not value):
            missing_data.append(desc)

    # If too many essential data points are missing, return an error
    if len(missing_data) >= 3:  # If 3 or more essential data points are missing
        product_name = product_details.get('product_name', '').strip()
        fallback_product_names = [
            product_details.get('product_name_en', '').strip(),
            product_details.get('product_name_es', '').strip(),
            product_details.get('product_name_fr', '').strip(),
            product_details.get('product_name_pl', '').strip(),
            product_details.get('product_name_pt', '').strip(),
            product_details.get('product_name_zh', '').strip(),
            product_details.get('product_name_de', '').strip(),
            product_details.get('product_name_it', '').strip(),
            product_details.get('product_name_ru', '').strip(),
            product_details.get('product_name_no', '').strip(),
            product_details.get('product_name_fi', '').strip()
        ]
        product_name_fallback = product_name if product_name else next(
            (name for name in fallback_product_names if name), "Unnamed")
        product_image_url = product_details.get('image_url', 'Image URL not available')

        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": 422,
                    "message": "Insufficient product data for analysis",
                    "description": f"Cannot analyze product due to missing critical data: {', '.join(missing_data)}",
                    "data": {
                        "product_name": product_name_fallback,
                        "image_url": product_image_url,
                        "barcode": product_details.get('code', barcode)
                    }
                }
            }
        )

    # If we have enough data, continue with normal processing
    product_name = product_details.get('product_name', '').strip()
    fallback_product_names = [
        product_details.get('product_name_en', '').strip(),
        product_details.get('product_name_es', '').strip(),
        product_details.get('product_name_fr', '').strip(),
        product_details.get('product_name_pl', '').strip(),
        product_details.get('product_name_pt', '').strip(),
        product_details.get('product_name_zh', '').strip(),
        product_details.get('product_name_de', '').strip(),
        product_details.get('product_name_it', '').strip(),
        product_details.get('product_name_ru', '').strip(),
        product_details.get('product_name_no', '').strip(),
        product_details.get('product_name_fi', '').strip()
    ]
    product_name_fallback = product_name if product_name else next((name for name in fallback_product_names if name),
                                                                   "Unnamed")

    product_image_url = product_details.get('image_url', 'Image URL not available')

    websocket_url = f"wss://staging.ai-foodhak.com/ws/grade/{barcode}"

    return {
        "websocket_url": websocket_url,
        "product_name": product_name_fallback,
        "product_image_url": product_image_url,
        "message": f"Please use this WebSocket URL to connect for real-time grading of barcode {barcode}."
    }


# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    global claude_client
    claude_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_PRODUCTION_API_KEY"))


@app.on_event("shutdown")
async def shutdown_event():
    global session
    if session and not session.closed:
        await session.close()


@app.get("/")
async def root():
    return {"message": "Server is running. Use WebSocket for real-time grading."}


@app.get("/health")
async def health_check():
    print("Production: Foodhakscore is up and running")
    return {"status": "healthy", "message": "Service is up and running."}


# Run the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

