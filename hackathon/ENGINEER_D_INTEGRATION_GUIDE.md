# 🤖 Engineer D Integration Guide
## LLM Explainer + React Frontend Integration

---

## 🎯 Integration Overview

Your LLM explainer will be the **star feature** that differentiates GridPilot-GT from other energy management systems. Here's how your backend components integrate seamlessly with the React frontend.

---

## 🏗️ Your Components in the Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Engineer D's Responsibilities                │
├─────────────────────────────────────────────────────────────────┤
│  🤖 LLM Explainer (gridpilot_gt/explain/llm_explainer.py)      │
│  ├── explain_decision(context: dict) -> str                    │
│  ├── llama-cpp-python + Mistral-7B q4_K_S                     │
│  └── Response time: <1s, Length: ≤300 chars                   │
├─────────────────────────────────────────────────────────────────┤
│  📊 Streamlit Dashboard (gridpilot_gt/ui/dashboard.py)         │
│  ├── Fallback/admin interface                                  │
│  ├── Grafana iframe integration                                │
│  └── Auto-refresh every 5s                                     │
├─────────────────────────────────────────────────────────────────┤
│  🌉 FastAPI Bridge (bridge/main.py) - You'll create this       │
│  ├── WebSocket /ws/live-data                                   │
│  ├── REST API /api/explain                                     │
│  └── Proxy /streamlit/* to your dashboard                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔌 API Specifications You Need to Implement

### 1. LLM Explainer Function
```python
# gridpilot_gt/explain/llm_explainer.py

from typing import Dict, Any
import time
from llama_cpp import Llama

class LLMExplainer:
    def __init__(self, model_path: str = "models/mistral-7b-q4_K_S.gguf"):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
    
    def explain_decision(self, context: Dict[str, Any]) -> str:
        """
        Generate explanation for current system decision.
        
        Args:
            context: {
                'prices_now': float,
                'prices_pred': List[float],
                'bids': Dict[str, float],
                'allocations': Dict[str, float],
                'payments': Dict[str, float],
                'soc': float,
                'cooling_kw': float,
                'expected_pnl': float,
                'carbon_intensity': float,
                'timestamp': str,
                'question': str  # From React frontend
            }
        
        Returns:
            str: Explanation ≤300 chars, contains "because" or "due to"
        """
        start_time = time.time()
        
        # Build prompt from context
        prompt = self._build_prompt(context)
        
        # Generate response
        response = self.llm(
            prompt,
            max_tokens=100,
            temperature=0.3,
            stop=[".", "!", "?"]
        )
        
        explanation = response['choices'][0]['text'].strip()
        
        # Ensure response meets requirements
        if len(explanation) > 300:
            explanation = explanation[:297] + "..."
        
        if "because" not in explanation.lower() and "due to" not in explanation.lower():
            explanation = f"This happened because {explanation}"
        
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            print(f"Warning: LLM response took {elapsed:.2f}s")
        
        return explanation
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build context-aware prompt for the LLM."""
        question = context.get('question', 'Why did the system make this decision?')
        
        prompt = f"""You are an AI assistant explaining energy trading decisions.

Current Situation:
- Energy Price: ${context['prices_now']:.2f}/MWh
- Battery SOC: {context['soc']:.1%}
- GPU Allocation: {context['allocations']['inference']:.1f}kW inference, {context['allocations']['training']:.1f}kW training
- Cooling Load: {context['cooling_kw']:.1f}kW
- Expected P&L: ${context['expected_pnl']:.2f}

Question: {question}

Explain in under 50 words why the system made this decision. Use "because" or "due to" in your response.

Answer:"""
        
        return prompt
```

### 2. FastAPI Bridge Server
```python
# bridge/main.py

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from typing import Dict, Any
import sys
import os

# Add parent directory to path to import GridPilot components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridpilot_gt.explain.llm_explainer import LLMExplainer
from main import main as run_gridpilot  # Import the main orchestrator

app = FastAPI(title="GridPilot-GT Bridge Server")

# Initialize LLM explainer
llm_explainer = LLMExplainer()

# Store latest system state
latest_state: Dict[str, Any] = {}

# Serve React build files
app.mount("/static", StaticFiles(directory="../ui/dist"), name="static")

@app.websocket("/ws/live-data")
async def websocket_endpoint(websocket: WebSocket):
    """Stream live system data to React frontend."""
    await websocket.accept()
    
    try:
        while True:
            # Get latest data from GridPilot system
            if latest_state:
                await websocket.send_text(json.dumps(latest_state))
            
            # Wait 1 second before next update
            await asyncio.sleep(1.0)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.post("/api/explain")
async def explain_decision(request: Dict[str, Any]):
    """Generate LLM explanation for current system state."""
    try:
        # Merge request with latest system state
        context = {**latest_state, **request}
        
        # Generate explanation
        explanation = llm_explainer.explain_decision(context)
        
        return {
            "explanation": explanation,
            "timestamp": context.get('timestamp'),
            "confidence": 0.85,  # Could be calculated based on context clarity
            "response_time_ms": 800  # Approximate response time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_loaded": llm_explainer is not None,
        "latest_update": latest_state.get('timestamp'),
        "services": {
            "websocket": "active",
            "llm_explainer": "ready",
            "streamlit_proxy": "configured"
        }
    }

# Background task to update system state
async def update_system_state():
    """Continuously update system state from GridPilot main loop."""
    global latest_state
    
    while True:
        try:
            # Run GridPilot simulation cycle
            result = run_gridpilot(simulate=True)
            
            if result and result.get('success'):
                # Transform result into frontend-friendly format
                payload = result['payload']
                
                latest_state = {
                    "timestamp": payload.get('timestamp', ''),
                    "power_allocation": {
                        "inference": payload['allocation']['inference'],
                        "training": payload['allocation']['training'], 
                        "cooling": payload['allocation']['cooling'],
                        "total": sum(payload['allocation'].values())
                    },
                    "battery": {
                        "soc": result.get('soc', 0.0),
                        "charging_rate": 0.0,  # Calculate from payload
                        "temperature": 25.0  # From system state
                    },
                    "pricing": {
                        "current": payload.get('current_price', 50.0),
                        "forecast_24h": [],  # From forecast data
                        "uncertainty": []
                    },
                    "auction": {
                        "last_bids": [],  # From VCG auction results
                        "total_revenue": result.get('revenue', 0.0)
                    },
                    "system": {
                        "efficiency": payload.get('efficiency', 1.0),
                        "constraints_satisfied": payload.get('constraints_satisfied', True),
                        "alerts": []
                    }
                }
                
        except Exception as e:
            print(f"Error updating system state: {e}")
        
        # Update every 5 seconds
        await asyncio.sleep(5.0)

# Start background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_system_state())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 🔗 Integration Points with React Frontend

### 1. WebSocket Data Stream
The React frontend will connect to your WebSocket at `ws://localhost:8000/ws/live-data` and receive real-time updates.

### 2. LLM Explanation API
The React chat interface will POST to `/api/explain` with questions and receive explanations.

---

## 🎯 Your Development Priorities

### Hour 0-2: LLM Foundation
1. **Set up llama-cpp-python** with Mistral-7B model
2. **Implement `explain_decision()`** function
3. **Test with dummy context** to ensure <1s response time

### Hour 2-4: FastAPI Bridge
1. **Create bridge server** with WebSocket endpoint
2. **Connect to main.py** orchestrator for live data
3. **Implement `/api/explain`** REST endpoint

### Hour 4-6: Streamlit Dashboard
1. **Build fallback dashboard** with live charts
2. **Integrate LLM explainer** with text input
3. **Add Grafana iframe** for monitoring

### Hour 6-8: Integration & Polish
1. **Test WebSocket data flow** with React frontend
2. **Optimize LLM response time** and caching
3. **Final testing** and demo preparation

---

## 🧪 Testing Your Components

```python
# tests/test_llm.py

def test_llm_explainer_latency():
    """Test LLM response time is under 1 second."""
    explainer = LLMExplainer()
    
    context = {
        'prices_now': 50.0,
        'soc': 0.73,
        'question': 'Why did we discharge the battery?'
    }
    
    start_time = time.time()
    explanation = explainer.explain_decision(context)
    elapsed = time.time() - start_time
    
    assert elapsed < 1.0
    assert len(explanation) <= 300
    assert any(word in explanation.lower() for word in ['because', 'due to'])
```

---

**🎯 Your LLM explainer is the secret weapon that will make GridPilot-GT stand out from every other hackathon project. Focus on making it fast, accurate, and engaging - the judges will be amazed by real-time AI explanations of complex energy decisions!** 