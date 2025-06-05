````markdown
# Foodhak-Score AI Service

**Real-time health grading for food products by barcode.  
Streams granular, evidence-based nutrition scores using Claude 3 and OpenAI fallback.  
Designed to work seamlessly alongside [Barcode-Recommend](../barcode-recommend).**

---

## 🌐 Environments

- **Production:** `https://ai-foodhak.com`
- **Staging:**    `https://staging.ai-foodhak.com`

---

## 🧠 What does it do?

- Accepts a barcode (EAN/UPC/etc) for a food product.
- Looks up OpenFoodFacts for nutrition, ingredients, NOVA score, etc.
- Streams back **seven granular JSON blocks**—covering nutrient quality, ingredient quality, processing, whole foods %, total score, name, and image—via WebSocket.
- Uses Claude 3 for primary grading, with OpenAI fallback for reliability.
- Strict JSON output for easy UI or API consumption.
- REST endpoint also provided for barcode lookup and WebSocket URL retrieval.

---

## 🚦 Endpoints

| Method | Endpoint                | Description                                         |
|--------|-------------------------|-----------------------------------------------------|
| POST   | `/grade-product`        | Get WebSocket URL & preview info for grading        |
| WS     | `/ws/grade/{barcode}`   | Real-time, streamed JSON grading for a product      |
| GET    | `/health`               | Health check                                        |
| GET    | `/`                     | Welcome/status message                              |

> **All endpoints require:**  
> `Authorization: Bearer <API_KEY>`

---

## 🛠️ Usage

### 1. Grade a Product — REST: `/grade-product`

#### Production

```bash
curl -X POST https://ai-foodhak.com/grade-product \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"barcode": "3046920029754"}'
````

#### Staging

```bash
curl -X POST https://staging.ai-foodhak.com/grade-product \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"barcode": "3046920029754"}'
```

**Success response:**

```json
{
  "websocket_url": "wss://ai-foodhak.com/ws/grade/3046920029754",
  "product_name": "Evian Natural Mineral Water",
  "product_image_url": "https://...",
  "message": "Please use this WebSocket URL to connect for real-time grading of barcode 3046920029754."
}
```

**Failure (invalid or missing product):**

```json
{
  "error": {
    "code": 404,
    "message": "No product details found for the barcode: 0000000000000.",
    "description": "Please verify the barcode and try again."
  }
}
```

---

### 2. Real-Time Grading — WebSocket: `/ws/grade/{barcode}`

* Connect to:

  * Production: `wss://ai-foodhak.com/ws/grade/{barcode}`
  * Staging:    `wss://staging.ai-foodhak.com/ws/grade/{barcode}`
* **No Bearer header required for WS connection (security is enforced via the REST API).**
* **Send:** Nothing (server streams results immediately after connection).
* **Receive:** 7 JSON objects, each streamed individually.

#### Example JSON blocks (in order, each as a standalone JSON object):

```json
{
  "Nutrient Quality": {
    "Explanation": "...",
    "Score": 3,
    "title": "Nutrient Quality"
  }
}

{
  "Processing Level": {
    "Explanation": "...",
    "Score": 1,
    "title": "Processing Level"
  }
}
```

*(Continues through all 7 blocks: Nutrient Quality, Processing Level, Ingredient Quality, Whole Foods Content, Total Score, Product Name, Product Image URL)*

* After all blocks, you'll receive:

```json
{
  "type": "message_stop",
  "message_id": "<id>",
  "data": "Streaming complete"
}
```

* If the product can't be graded (insufficient data), you’ll receive:

```json
{
  "error": {
    "code": 422,
    "message": "Insufficient product data for analysis",
    "description": "Cannot analyze product due to missing critical data: Ingredients, Additives, NOVA Group",
    "data": {
      "product_name": "Unknown",
      "image_url": "Image URL not available",
      "barcode": "0000000000000"
    }
  }
}
```

---

### 3. Health Check

```bash
curl https://ai-foodhak.com/health
```

**Result:**

```json
{
  "status": "healthy",
  "message": "Service is up and running."
}
```

---

## ⚡ Features

* **Claude 3 (Anthropic) for primary grading** with robust OpenAI fallback.
* **7-part JSON output** — ready for UI and analytics.
* **Streams as results are generated** (real-time feedback).
* **Automatic error handling** for bad/missing barcodes or product data.
* **Strict formatting**: Each response is a *standalone* JSON object, never wrapped in arrays or extra commentary.
* **Transparent criteria**: The grading system is fully described in the prompt (nutrients, processing, whole foods, etc.).

---

## 📝 Developer Notes

* All API keys and Claude/OpenAI secrets must be set as environment variables.
* Supports both production and staging with identical API structure.
* *No profile required*: Only the barcode is needed.
* Designed to be chained with [Barcode-Recommend](../barcode-recommend) for full user-facing product/scan flows.

---