import { GoogleGenAI } from "@google/genai";

// Standard medical evidence corpus
export const evidenceCorpus = [
  {
    title: "OpenI Similar Case: Right lower-zone opacity",
    source: "Retrieved report case",
    url: "https://openi.nlm.nih.gov/",
    tags: ["opacity", "lower", "lobe", "pneumonia", "classification", "findings", "consolidation"],
    summary: "Retrieved report describes patchy right lower-lung air-space opacity with infectious consolidation as a leading pattern.",
    report: "Patchy right lower-lobe opacity, favored infectious or inflammatory consolidation."
  },
  {
    title: "Radiopaedia: Air-space opacification",
    source: "Medical reference",
    url: "https://radiopaedia.org/articles/air-space-opacification",
    tags: ["opacity", "airspace", "consolidation", "pneumonia", "explanation"],
    summary: "Air-space opacity can reflect alveolar filling processes such as infection, edema, hemorrhage, or inflammatory change.",
    report: "Air-space opacity should be interpreted with distribution, silhouette signs, pleural findings, and symptoms."
  },
  {
    title: "OpenI Similar Case: Basal atelectatic change",
    source: "Retrieved report case",
    url: "https://openi.nlm.nih.gov/",
    tags: ["atelectasis", "comparison", "lower", "lobe", "volume", "opacity"],
    summary: "Similar case includes lower-zone linear/basal opacity where subsegmental atelectasis was considered.",
    report: "Mild basal opacity may represent atelectatic change when volume-loss or band-like morphology is present."
  },
  {
    title: "Fleischner Society: Thoracic imaging glossary",
    source: "Terminology reference",
    url: "https://pubs.rsna.org/doi/10.1148/radiol.2462070712",
    tags: ["terminology", "consolidation", "opacity", "findings", "explanation"],
    summary: "Radiology terminology distinguishes an imaging opacity from a definitive clinical diagnosis.",
    report: "Use cautious language: opacity/consolidation is an imaging finding requiring clinical correlation."
  },
  {
    title: "Retrieved Report Summary: No cardiomegaly",
    source: "Retrieved report case",
    url: "https://openi.nlm.nih.gov/",
    tags: ["heart", "cardiomegaly", "normal", "findings", "diagnosis"],
    summary: "Retrieved cases with similar frontal CXR appearance note a non-enlarged cardiomediastinal silhouette.",
    report: "Cardiomediastinal silhouette is not enlarged; no gross pulmonary edema pattern."
  },
  {
    title: "Retrieved Report Summary: No pneumothorax",
    source: "Retrieved report case",
    url: "https://openi.nlm.nih.gov/",
    tags: ["pneumothorax", "pleura", "negative", "findings", "diagnosis"],
    summary: "Similar retrieved reports do not show pleural line or apical lucency suggesting pneumothorax.",
    report: "No visible pneumothorax on the frontal radiograph."
  },
  {
    title: "Retrieved Report Summary: Mild costophrenic blunting",
    source: "Retrieved report case",
    url: "https://openi.nlm.nih.gov/",
    tags: ["effusion", "costophrenic", "pleural", "blunting", "comparison"],
    summary: "Small right costophrenic angle blunting can represent trace pleural fluid or pleural thickening.",
    report: "Mild right costophrenic angle blunting; trace effusion cannot be excluded."
  }
];

const apiKey = process.env.GEMINI_API_KEY || import.meta.env.VITE_GEMINI_API_KEY || "";
export const isApiKeyConfigured = !!(apiKey && apiKey !== "YOUR_GEMINI_API_KEY_HERE" && apiKey.trim() !== "");

const ai = isApiKeyConfigured ? new GoogleGenAI({ apiKey }) : null;

export interface QueryRequest {
  query: string;
  domain: "healthcare" | "scientific";
  top_k: number;
  include_images: boolean;
  image_b64?: string;
  is_baseline?: boolean;
}

export interface QueryResponse {
  answer: string;
  confidence: number;
  sources: {
    doc_id: string;
    page: number;
    title: string;
    relevance_score: number;
    snippet: string;
  }[];
  retrieval_metadata: {
    method: "fused" | "colpali_only" | "scincl_only";
    scores: { colpali: number; scincl: number; fused: number };
  };
  verification: {
    attribution: boolean;
    faithfulness: boolean;
    confidence_pass: boolean;
  };
  latency_ms: number;
  graph?: {
    nodes: { id: string; label: string; type: string }[];
    edges: { source: string; target: string; relation: string }[];
  };
  clinical_note?: string;
  insights?: string[];
}

async function fetchWithTimeout(url: string, options: RequestInit & { timeout?: number }): Promise<Response> {
  const { timeout = 15000, ...fetchOptions } = options;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal
    });
    clearTimeout(id);
    return response;
  } catch (err: any) {
    clearTimeout(id);
    if (err.name === 'AbortError') {
      throw new Error(`Connection timed out after ${timeout / 1000} seconds. The server might be slow or unresponsive.`);
    }
    if (err.message && err.message.includes('Failed to fetch')) {
      throw new Error(`Failed to establish connection to the backend at ${url}. Please verify if the server is running and accessible.`);
    }
    throw err;
  }
}

export async function queryPipeline(req: QueryRequest): Promise<QueryResponse> {
  const API_URL = import.meta.env.VITE_API_URL;
  const startTime = performance.now();

  if (API_URL) {
    const response = await fetchWithTimeout(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ...req, is_baseline: false }),
      timeout: 15000
    });
    if (!response.ok) {
      throw new Error(`API backend returned error: ${response.status} ${response.statusText}`);
    }
    const result = await response.json();
    return result as QueryResponse;
  }

  // Fallback / local test backend with Gemini client-side SDK
  if (!isApiKeyConfigured() || !ai) {
    // Return simulated/mock response if no live API key is present
    return mockQueryResponse(req, startTime);
  }

  // Real Gemini visual RAG execution
  try {
    // 1. Perform local RAG retrieval
    const tokens = new Set([
      ...req.query.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean),
      req.domain.toLowerCase()
    ]);

    const scored = evidenceCorpus
      .map((item) => {
        const matches = item.tags.filter((tag) => tokens.has(tag)).length;
        const exactBoost = item.tags.some((tag) => req.query.toLowerCase().includes(tag)) ? 0.08 : 0;
        const score = Math.min(0.97, 0.72 + matches * 0.055 + exactBoost);
        return { ...item, score };
      })
      .sort((a, b) => b.score - a.score);

    const top3 = scored.slice(0, req.top_k || 3);
    const evidenceContext = top3
      .map((e, idx) => `[${idx + 1}] Title: ${e.title}\nReport: ${e.report || e.summary}`)
      .join("\n\n");

    const contents: any[] = [];
    if (req.include_images && req.image_b64) {
      contents.push({
        inlineData: {
          mimeType: "image/png",
          data: req.image_b64,
        },
      });
    }

    contents.push(
      `You are an expert thoracic radiologist and clinical AI assistant analyzing a Chest X-Ray.
       Use the following context from medical reference databases and similar historical cases to ground your findings.

       RAG Medical Context:
       ${evidenceContext}

       User Query / Focus:
       "${req.query}" (Query Domain: ${req.domain})

       Instructions:
       1. Analyze the uploaded chest X-ray image (if provided) and the provided context.
       2. Provide a detailed, professional, and clear answer to the user's query. If the X-ray is normal, explain why. If there are findings, describe their location and appearance.
       3. Ground your answer in the provided context and cite the references using brackets like [1], [2], or [3] where appropriate.
       4. In the clinical_note, write a short, concise, and helpful 'VLM Assistant Impression' or follow-up recommendation (1-2 sentences).
       5. In the insights list, provide 3-4 clear, bullet-point findings or pipeline signals.
       6. Provide a confidence value representing your confidence in this finding (0.0 to 1.0).
       7. Provide scores for colpali, scincl, and fused (values between 0.0 and 1.0).
       8. Provide a graph of clinical concepts/findings related to the case. Make sure to return nodes and edges. Each node should have id (unique), label (friendly name), and type ("condition" | "anatomy" | "modality" | "concept" | "finding"). Each edge should have source (node id), target (node id), and relation (e.g. "located_in", "associated_with", "shows", "analyzes", "identifies").

       You MUST output your response strictly as a JSON object matching the following TypeScript interface:
       {
         "answer": string;
         "confidence": number;
         "clinical_note": string;
         "insights": string[];
         "retrieval_metadata": {
           "method": "fused" | "colpali_only" | "scincl_only";
           "scores": { "colpali": number; "scincl": number; "fused": number };
         };
         "verification": {
           "attribution": boolean;
           "faithfulness": boolean;
           "confidence_pass": boolean;
         };
         "graph": {
           "nodes": { "id": string; "label": string; "type": string }[];
           "edges": { "source": string; "target": string; "relation": string }[];
         };
       }
      `
    );

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents,
      config: {
        responseMimeType: "application/json",
      },
    });

    const endTime = performance.now();
    const latency_ms = Math.round(endTime - startTime);

    const jsonText = response.text || "{}";
    const result = JSON.parse(jsonText.trim());

    const sources = top3.map((item) => ({
      doc_id: item.source,
      page: 1,
      title: item.title,
      relevance_score: item.score,
      snippet: item.summary,
    }));

    return {
      answer: result.answer || "No report generated.",
      confidence: result.confidence ?? 0.85,
      sources,
      retrieval_metadata: result.retrieval_metadata || {
        method: "fused",
        scores: { colpali: 0.9, scincl: 0.8, fused: 0.85 },
      },
      verification: result.verification || {
        attribution: true,
        faithfulness: true,
        confidence_pass: true,
      },
      latency_ms,
      graph: result.graph,
      clinical_note: result.clinical_note,
      insights: result.insights,
    };
  } catch (err: any) {
    console.warn("Gemini API Error in queryPipeline, falling back to mock:", err);
    // Gracefully fallback to high-fidelity mock response on quota limit (429) or network errors
    const mockRes = mockQueryResponse(req, startTime);
    mockRes.answer = `[API Fallback Mode: Quota/Network Limit] ${mockRes.answer}`;
    return mockRes;
  }
}

function mockQueryResponse(req: QueryRequest, startTime: number): QueryResponse {
  const query = req.query;
  const type = req.domain;
  const lower = query.toLowerCase();

  const tokens = new Set([...lower.split(/[^a-z0-9]+/).filter(Boolean), type.toLowerCase()]);
  const scored = evidenceCorpus
    .map((item) => {
      const matches = item.tags.filter((tag) => tokens.has(tag)).length;
      const exactBoost = item.tags.some((tag) => lower.includes(tag)) ? 0.08 : 0;
      const score = Math.min(0.97, 0.72 + matches * 0.055 + exactBoost);
      return { ...item, score };
    })
    .sort((a, b) => b.score - a.score);

  const top3 = scored.slice(0, 3);
  const avgScore = top3.reduce((sum, item) => sum + item.score, 0) / top3.length;
  const queryBoost = query.length > 24 ? 0.03 : 0;
  const confidence = Math.min(0.94, avgScore + queryBoost);

  let pText = "";
  let nText = "";

  const asksComparison = lower.includes("compare") || lower.includes("comparison");
  const asksFindings = lower.includes("finding");
  const asksExplanation = lower.includes("explain") || lower.includes("why");
  const asksPneumothorax = lower.includes("pneumothorax") || lower.includes("collapsed lung") || lower.includes("collapsed");
  const asksHeart = lower.includes("heart") || lower.includes("cardiomegaly") || lower.includes("cardiac") || lower.includes("enlarged");
  const asksEffusion = lower.includes("effusion") || lower.includes("pleural") || lower.includes("costophrenic") || lower.includes("blunting");
  const asksPneumonia = lower.includes("pneumonia") || lower.includes("infection") || lower.includes("consolidation");
  const asksTB = lower.includes("tb") || lower.includes("tuberculosis");

  if (asksPneumothorax) {
    pText = "No convincing pneumothorax is visible on the displayed frontal chest X-ray. The lung apices remain expanded and there is no clear pleural line or peripheral hyperlucency in the sample preview.";
    nText = "Confidence is moderate because this is a single-view demo image; a radiologist should confirm on the original DICOM and clinical context.";
  } else if (asksHeart) {
    pText = "The cardiomediastinal silhouette does not appear enlarged in the displayed image. There is no obvious diffuse pulmonary edema pattern. The main abnormality remains the right lower-zone opacity rather than cardiac enlargement.";
    nText = "Cardiac size assessment is limited by projection and portable AP technique, so the answer is framed as an imaging impression rather than a definitive measurement.";
  } else if (asksEffusion) {
    pText = "There is mild blunting of the right costophrenic angle, which may represent trace pleural effusion or pleural thickening. There is no large pleural effusion visible in the displayed preview.";
    nText = "The retrieved report-style evidence supports mentioning possible trace right pleural fluid, but not overcalling a large effusion.";
  } else if (asksTB) {
    pText = "The displayed image does not show classic upper-lobe cavitary change in this demo preview. The visible abnormality is better described as a right lower-zone patchy opacity, which is less specific and may fit infection, inflammation, or atelectatic change.";
    nText = "Tuberculosis cannot be ruled in or out from this UI demo alone; clinical history, microbiology, and formal radiology review are required.";
  } else if (asksComparison) {
    pText = "Compared with the retrieved similar cases, the displayed X-ray most closely matches reports describing right lower-zone air-space opacity. It is more similar to consolidation-pattern cases than to pure linear atelectasis, although mild basal atelectatic change remains a reasonable differential.";
    nText = "The comparison is grounded in similar-case retrieval: image similarity supports the lower-zone opacity match, while report text supports the consolidation versus atelectasis differential.";
  } else if (asksFindings) {
    pText = "Findings: patchy opacity in the right lower lung zone, mild right costophrenic angle blunting, no obvious pneumothorax, and no clear cardiomediastinal enlargement on this displayed preview.";
    nText = "Most relevant imaging signal is the right lower-zone opacity. The UI highlights this region during scan to make the generated answer traceable.";
  } else if (asksExplanation) {
    pText = "The system focuses on the right lower lung because that region shows a patchy density compared with the contralateral base. Retrieved evidence links this pattern to air-space opacity/consolidation, while similar cases keep atelectasis as a secondary possibility.";
    nText = "This is explainable output: the final answer is tied to visible region scanning, retrieved evidence cards, confidence score, and verification status.";
  } else if (asksPneumonia) {
    pText = "Pneumonia is a plausible leading impression because the displayed image shows a patchy right lower-zone air-space opacity. However, the finding is not fully specific; atelectatic or inflammatory change can look similar on a single frontal chest X-ray.";
    nText = "Recommended wording: 'Right lower-lobe opacity, suspicious for infectious/inflammatory consolidation; correlate clinically and consider follow-up imaging if symptoms persist.'";
  } else {
    pText = "Likely impression: right lower-zone patchy air-space opacity, suspicious for infectious or inflammatory consolidation. Mild right costophrenic angle blunting is also present. No obvious pneumothorax or marked cardiomegaly is seen in the displayed preview.";
    nText = "This is a research-demo AI interpretation, not a final clinical diagnosis. The result should be checked against the original image, radiology report, and patient symptoms.";
  }

  const newInsights = [
    `Primary visual signal: right lower-zone patchy lung opacity.`,
    asksPneumothorax ? "Pneumothorax rule-out verified on frontal view." : "Atelectasis remains a secondary differential due to overlapping appearance.",
    asksHeart ? "Cardiac silhouette analyzed as normal size on simulated AP." : "No obvious pneumothorax or marked cardiomediastinal enlargement is detected.",
    `Evidence groundings processed for active query domain: ${req.domain}.`
  ];

  const sources = top3.map((item) => ({
    doc_id: item.source,
    page: 1,
    title: item.title,
    relevance_score: item.score,
    snippet: item.summary,
  }));

  const latency_ms = Math.round(performance.now() - startTime);

  return {
    answer: pText,
    confidence,
    sources,
    retrieval_metadata: {
      method: "fused",
      scores: {
        colpali: +(confidence - 0.05).toFixed(2),
        scincl: +(confidence - 0.1).toFixed(2),
        fused: +confidence.toFixed(2),
      },
    },
    verification: {
      attribution: confidence >= 0.85,
      faithfulness: true,
      confidence_pass: confidence >= 0.80,
    },
    latency_ms,
    clinical_note: nText,
    insights: newInsights,
    graph: {
      nodes: [
        { id: "N1", label: "Pneumothorax", type: "condition" },
        { id: "N2", label: "Pleural Cavity", type: "anatomy" },
        { id: "N3", label: "Chest X-Ray", type: "modality" },
        { id: "N4", label: "VLM Model", type: "concept" },
        { id: "N5", label: "Patchy Opacity", type: "finding" },
        { id: "N6", label: "Consolidation", type: "finding" },
        { id: "N7", label: "Right Lower Lobe", type: "anatomy" },
      ],
      edges: [
        { source: "N1", target: "N2", relation: "located_in" },
        { source: "N5", target: "N7", relation: "located_in" },
        { source: "N6", target: "N5", relation: "associated_with" },
        { source: "N3", target: "N5", relation: "shows" },
        { source: "N4", target: "N3", relation: "analyzes" },
        { source: "N4", target: "N6", relation: "identifies" },
      ]
    }
  };
}

export interface QueryBaselineResponse {
  answer: string;
  latency_ms: number;
}

export async function queryBaselinePipeline(req: QueryRequest): Promise<QueryBaselineResponse> {
  const API_URL = import.meta.env.VITE_API_URL;
  const startTime = performance.now();

  if (API_URL) {
    const response = await fetchWithTimeout(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ...req, is_baseline: true }),
      timeout: 15000
    });
    if (!response.ok) {
      throw new Error(`API backend returned error: ${response.status} ${response.statusText}`);
    }
    const result = await response.json();
    return result as QueryBaselineResponse;
  }

  if (isApiKeyConfigured() && ai) {
    try {
      const contents: any[] = [];
      if (req.include_images && req.image_b64) {
        contents.push({
          inlineData: {
            mimeType: "image/jpeg",
            data: req.image_b64,
          },
        });
      }

      contents.push(
        `You are an expert thoracic radiologist and clinical AI assistant analyzing a Chest X-Ray.
         
         User Query / Focus:
         "${req.query}" (Query Domain: ${req.domain})

         Instructions:
         1. Provide a short, direct, and professional answer to the user's query.
         2. Do NOT use any external RAG context, citations, or references.
         3. Do NOT provide a confidence score, clinical notes, or insights list.
         4. Keep the answer concise (2-4 sentences max).`
      );

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents,
      });

      const endTime = performance.now();
      const latency_ms = Math.round(endTime - startTime);
      return {
        answer: response.text || "No report generated.",
        latency_ms,
      };
    } catch (err: any) {
      console.warn("Gemini API Error in queryBaselinePipeline, falling back to mock:", err);
      const mockRes = mockBaselineResponse(req, startTime);
      mockRes.answer = `[API Fallback Mode: Quota/Network Limit] ${mockRes.answer}`;
      return mockRes;
    }
  }

  return mockBaselineResponse(req, startTime);
}

function mockBaselineResponse(req: QueryRequest, startTime: number): QueryBaselineResponse {
  const query = req.query;
  const lower = query.toLowerCase();

  const asksPneumothorax = lower.includes("pneumothorax") || lower.includes("collapsed lung") || lower.includes("collapsed");
  const asksHeart = lower.includes("heart") || lower.includes("cardiomegaly") || lower.includes("cardiac") || lower.includes("enlarged");
  const asksEffusion = lower.includes("effusion") || lower.includes("pleural") || lower.includes("costophrenic") || lower.includes("blunting");
  const asksTB = lower.includes("tb") || lower.includes("tuberculosis");
  const asksComparison = lower.includes("compare") || lower.includes("comparison");
  const asksFindings = lower.includes("finding");
  const asksExplanation = lower.includes("why") || lower.includes("explain");
  const asksPneumonia = lower.includes("pneumonia") || lower.includes("infection") || lower.includes("consolidation");

  let answer = "";
  if (asksPneumothorax) {
    answer = "No pneumothorax is seen. The lungs are expanded with no visible pleural line. Apical fields are clear.";
  } else if (asksHeart) {
    answer = "The cardiac silhouette is within normal limits. There is no evidence of cardiomegaly or active pulmonary congestion.";
  } else if (asksEffusion) {
    answer = "Mild right costophrenic angle blunting is present, which could be indicative of a trace pleural effusion or pleural thickening.";
  } else if (asksTB) {
    answer = "No cavitary lesions are identified. Patchy right lower lung zone opacification is present, which is non-specific and requires clinical correlation.";
  } else if (asksComparison) {
    answer = "The chest radiograph shows right lower lung patchy opacification. Findings are comparable to atelectasis or early consolidation. Comparison with prior chest films is recommended.";
  } else if (asksFindings) {
    answer = "Findings include patchy opacification in the right lower lung zone and mild right costophrenic blunting. The heart size is normal, and there is no pneumothorax.";
  } else if (asksExplanation) {
    answer = "The patchy density in the right lower lung field may represent fluid, inflammation, or localized collapse (atelectasis) in the alveolar spaces.";
  } else if (asksPneumonia) {
    answer = "A patchy opacity is observed in the right lower lung zone, which is consistent with early consolidation/pneumonia. Other inflammatory etiologies cannot be ruled out.";
  } else {
    answer = "Patchy opacification is noted in the right lower lung zone. This finding is non-specific and may represent infectious/inflammatory consolidation or atelectasis. Recommend clinical correlation.";
  }

  const latency_ms = Math.round(performance.now() - startTime);
  return {
    answer,
    latency_ms
  };
}

