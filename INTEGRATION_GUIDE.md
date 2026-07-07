# MMRAG Healthcare Frontend Integration Guide

This guide details how to connect the MMRAG Healthcare frontend to any compatible custom backend server. By adhering to the contract defined below, you can replace the client-side Gemini simulation with a live, server-side multimodal RAG pipeline without changing any frontend code.

---

## 1. Backend Endpoint Configuration

The frontend uses a single environment variable to toggle between mock/client-side mode and your custom live backend server:

Create or edit your `.env.local` file in the root directory:
```bash
# Set this to your unified backend endpoint
VITE_API_URL="http://localhost:8000/api/query"
```

*   **Mock Fallback**: If `VITE_API_URL` is empty, commented out, or not set, the frontend will automatically fallback to local mock simulators or direct client-side Gemini calls (if `VITE_GEMINI_API_KEY` is provided).
*   **Production Live**: If `VITE_API_URL` is set, all frontend analysis queries (both Baseline and Enhanced RAG) will be routed to your server.

---

## 2. API Contract (Request Format)

When a query is initiated, the frontend sends a **POST** request to the configured `VITE_API_URL` with a JSON body matching the following structure:

```json
{
  "query": "Evaluate right and left lower lobes for signs of acute pneumonia.",
  "domain": "healthcare",
  "top_k": 3,
  "include_images": true,
  "image_b64": "iVBORw0KGgoAAAANS...",
  "is_baseline": false
}
```

### Request Schema Definitions

| Field | Type | Description |
| :--- | :--- | :--- |
| `query` | `string` | The clinical question or suggestion selected by the user. |
| `domain` | `string` | `"healthcare"` (radiology) or `"scientific"` (research documents). |
| `top_k` | `number` | Number of relevant evidence documents to retrieve. |
| `include_images` | `boolean` | Flag indicating whether the visual radiographic image is attached. |
| `image_b64` | `string` (optional) | Base64-encoded JPEG image data (if `include_images` is true). |
| `is_baseline` | `boolean` | **Key Route Selector**: `true` if the UI is querying the Baseline/Basic pipeline; `false` if it is querying the Enhanced RAG pipeline. |

---

## 3. Response Contracts

Depending on the `is_baseline` flag, your backend must return the corresponding JSON response structure:

### Case A: Enhanced RAG Response (`is_baseline: false` or omitted)

Must return a full diagnostic package containing answers, source attributions, verification scores, and an optional knowledge graph.

```json
{
  "answer": "There is a patchy opacity in the right lower lung zone [1] indicating possible infection. Mild blunting of the right costophrenic angle is noted [2].",
  "confidence": 0.87,
  "latency_ms": 1200,
  "clinical_note": "Correlate with acute patient inflammatory symptoms and follow up in 2 weeks.",
  "insights": [
    "Right lower-lobe opacity detected with possible consolidation.",
    "Mild right costophrenic angle blunting observed."
  ],
  "verification": {
    "attribution": true,
    "faithfulness": true,
    "confidence_pass": true
  },
  "sources": [
    {
      "doc_id": "Radiopaedia: Lower Lobe Pneumonia",
      "page": 1,
      "title": "Airspace Opacification Guidelines",
      "relevance_score": 0.94,
      "snippet": "Patient presents with patchy basal airspace opacity and right costophrenic blunting patterns consistent with infection."
    }
  ],
  "graph": {
    "nodes": [
      { "id": "opacity", "label": "Patchy Opacity", "type": "Finding" },
      { "id": "pneumonia", "label": "Pneumonia", "type": "Condition" }
    ],
    "edges": [
      { "source": "opacity", "target": "pneumonia", "relation": "indicates" }
    ]
  }
}
```

### Case B: Baseline Response (`is_baseline: true`)

Must return a simpler, direct generation containing only the raw text response and timing latency.

```json
{
  "answer": "Patchy opacification is noted in the right lower lung zone. This finding is non-specific and may represent infectious/inflammatory consolidation.",
  "latency_ms": 350
}
```

---

## 4. Reference Implementation Examples

### Python (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI()

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    domain: str
    top_k: int
    include_images: bool
    image_b64: Optional[str] = None
    is_baseline: Optional[bool] = False

@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    try:
        if req.is_baseline:
            # Simple direct model generation (no RAG)
            return {
                "answer": "Direct model generation answer matching query: " + req.query,
                "latency_ms": 250
            }
        else:
            # Full Multimodal RAG pipeline execution
            return {
                "answer": "Enhanced grounded answer with citations [1].",
                "confidence": 0.92,
                "latency_ms": 1100,
                "clinical_note": "Clinical warning details go here.",
                "insights": ["Insight point 1", "Insight point 2"],
                "verification": {
                    "attribution": True,
                    "faithfulness": True,
                    "confidence_pass": True
                },
                "sources": [
                    {
                        "doc_id": "doc_001",
                        "page": 2,
                        "title": "Reference Article 1",
                        "relevance_score": 0.95,
                        "snippet": "Text snippet retrieved that contains relevant details."
                    }
                ],
                "graph": {
                    "nodes": [{"id": "a", "label": "Node A", "type": "Topic"}],
                    "edges": [{"source": "a", "target": "a", "relation": "self"}]
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 5. UI Error & Timeout Handlers

The frontend implements a **15-second client-side timeout** using `AbortController`. 

If your backend is slow (e.g. cold-start or API rate-limit), the client will automatically cancel the fetch request and show a red alert card. Ensure your backend returns responses under 15 seconds, or implement streaming/chunking endpoints if long delays are unavoidable.
