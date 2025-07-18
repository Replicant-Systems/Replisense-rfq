# Replisense Request For Quotation Microservice

A Python-based containerised microservice for parsing and extracting RFQ (Request for Quotation) data from email text and attachments (PDF, Excel). This service powers the RFQ intelligence module in the [Replisense](https://www.replicantsys.com/) platform by providing fast, reliable LLM-driven extraction of line items and metadata.

---

## 🚀 Features

- Extract RFQ line items from raw email body or file attachments
- Integrates with LLMs (e.g., Gemini, GPT) for intelligent parsing
- Supports `.pdf`, `.xlsx`, `.csv`, and raw email text
- FastAPI-based for low-latency JSON API
- Docker-ready for EC2 or container orchestration deployment
- Designed to run behind internal auth or proxy (no exposed secrets)

---

## 🛠️ API Endpoints

### `POST /upload/`
Upload an RFQ file (PDF or Excel) and get structured line items.

**Request:**
- `multipart/form-data` with key `file`

**Response:**
```json
{
  "success": true,
  "line_items": [
    {
      "part_number": "PRT-001-A",
      "quantity": 500,
      "target_price": 1020
    }
  ]
}

---

### `POST /parse-text/`

Send raw email body text and get parsed RFQ line items.

**Request Body:**

```json
{
  "text": "Hello team, please send a quote for the following items..."
}
```

**Response:**
Same format as `/upload/`.

---

## 🐳 Docker Usage

### 1. Build the Docker image

```bash
docker build -t replisense-rfq .
```

### 2. Run the container

```bash
docker run -d -p 8000:8000 --name rfq-service replisense-rfq
```

Now the API will be available at: `http://localhost:8000`

---

## 📦 Project Structure

```
replisense-rfq/
├── main.py                 # FastAPI entrypoint
├── requirements.txt
├── Dockerfile
└── README.md
└── file_parser.py
└── main.py
└── rfq_agent.py

```

---

## 🔧 Environment Variables (Optional)

| Name          | Description                          |
| ------------- | ------------------------------------ |
| `LLM_API_KEY` | API key for Gemini / OpenAI backend  |

---

## Integration with Replisense Node.js Backend

Our Node.js backend can talk to this microservice via:

```ts
axios.post("http://localhost:8000/parse-text/", { text: emailBody });
```

Or for files:

```ts
axios.post("http://localhost:8000/upload/", formData);
```

> Make sure the RFQ service is running and accessible from your Node backend.

---

## 📜 License

This microservice is part of the proprietary Replisense platform. Not for public use without permission.

---

