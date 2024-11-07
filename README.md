This repo contains scripts for Foodhak-Score staging and production environments.

The **Foodhak-score** API is a structured and insightful tool for evaluating food products based on health criteria. Here's a quick summary of its key components and the overall process:

### Key Features
1. **Integration with OpenFoodFacts API**: Retrieves food product data using barcodes, enabling comprehensive information for health analysis.
2. **AI-Driven Scoring System**: Uses a Large Language Model (LLM) to generate a health score based on:
   - Nutrient Quality (e.g., proteins, fibers, added sugars)
   - Processing Level (NOVA score indicating processing level)
   - Ingredient Quality (natural vs. artificial, preservatives)
   - Whole Foods Content (degree of whole, unprocessed ingredients)

3. **Detailed Scoring Categories**:
   - **Nutrient Quality**: Starts at 30 points, deductions for high sugars, fats, etc.
   - **Processing Level**: Up to 20 points based on NOVA group.
   - **Ingredient Quality**: Starts at 30, deductions for additives and artificial ingredients.
   - **Whole Foods Content**: Out of 20 points, based on whole ingredient presence.
   
4. **Authorization & Error Handling**:
   - Requires a Bearer token for authentication.
   - Includes retries for API calls and response generation in case of network or server issues.

### Example cURL Requests
- **Production**:
    ```bash
    curl -X POST https://www.foodhakai.com/foodhak-score \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer viJ8u142.NaQl7JEW5u8bEJpqnnRuvilTfDbHyWty" \
    -d '{"barcode": "80177173"}'
    ```
- **Staging**:
    ```bash
    curl -X POST https://www.staging-foodhakai.com/foodhak-score \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer mS6WabEO.1Qj6ONyvNvHXkWdbWLFi9mLMgHFVV4m7" \
    -d '{"barcode": "80177173"}'
    ```

### Response Structure
The response includes a health score breakdown for each category, an explanation, the product name, and an image URL, providing clients with an informed and detailed overview of the product's healthiness. 

### Usage Scenario
Ideal for applications focused on nutrition transparency, this API equips users with health scores and in-depth explanations, supporting healthier food choices by analyzing nutrient, processing, and ingredient quality.
