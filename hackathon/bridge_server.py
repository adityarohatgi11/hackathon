#!/usr/bin/env python3
"""
FastAPI Bridge Server for GridPilot-GT Frontend Integration
Serves as the WebSocket + REST API bridge between React frontend and core engine.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time
from typing import Dict, Any, List
import logging
from datetime import datetime
import traceback

# Import core GridPilot components
from main import main as run_gridpilot
from api_client.client import MARAClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GridPilot-GT Bridge Server",
    description="WebSocket + REST API bridge for React frontend",
    version="1.0.0"
)

# Enable CORS for React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
class GridPilotState:
    def __init__(self):
        self.latest_data: Dict[str, Any] = {}
        self.auction_history: List[Dict[str, Any]] = []
        self.connected_clients: List[WebSocket] = []
        self.is_running = False
        
    async def update_data(self, new_data: Dict[str, Any]):
        """Update system state and broadcast to all connected clients."""
        self.latest_data = new_data
        
        # Add to auction history if this is an auction result
        if 'auction_results' in new_data:
            auction_entry = {
                'timestamp': new_data.get('timestamp', datetime.now().isoformat()),
                'bids': new_data['auction_results'].get('bids', {}),
                'allocations': new_data['auction_results'].get('allocations', {}),
                'payments': new_data['auction_results'].get('payments', {}),
                'clearing_price': new_data['auction_results'].get('clearing_price', 0)
            }
            self.auction_history.append(auction_entry)
            
            # Keep only last 1000 entries
            if len(self.auction_history) > 1000:
                self.auction_history = self.auction_history[-1000:]
        
        # Broadcast to all connected WebSocket clients
        if self.connected_clients:
            message = json.dumps({
                'type': 'system_update',
                'data': new_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Send to all clients (remove disconnected ones)
            disconnected = []
            for client in self.connected_clients:
                try:
                    await client.send_text(message)
                except:
                    disconnected.append(client)
            
            # Remove disconnected clients
            for client in disconnected:
                self.connected_clients.remove(client)

# Global state instance
state = GridPilotState()

# Background task to run GridPilot engine
async def gridpilot_engine_task():
    """Background task that runs the GridPilot engine and updates state."""
    logger.info("Starting GridPilot engine background task")
    
    while True:
        try:
            if state.is_running:
                # Run one iteration of the GridPilot engine
                result = run_gridpilot()
                
                if result and result.get('success'):
                    # Extract relevant data for frontend
                    frontend_data = {
                        'timestamp': datetime.now().isoformat(),
                        'success': True,
                        'prices_now': result.get('prices_now', 0),
                        'prices_pred': result.get('prices_pred', []),
                        'auction_results': result.get('auction_results', {}),
                        'battery_soc': result.get('battery_soc', 0.5),
                        'cooling_kw': result.get('cooling_kw', 0),
                        'expected_pnl': result.get('expected_pnl', 0),
                        'carbon_intensity': result.get('carbon_intensity', 0.4),
                        'system_status': 'operational'
                    }
                    
                    await state.update_data(frontend_data)
                else:
                    # Handle error case
                    error_data = {
                        'timestamp': datetime.now().isoformat(),
                        'success': False,
                        'error': result.get('error', 'Unknown error'),
                        'system_status': 'error'
                    }
                    await state.update_data(error_data)
            
            # Wait before next iteration (adjust based on your needs)
            await asyncio.sleep(5.0)  # 5-second intervals
            
        except Exception as e:
            logger.error(f"Error in GridPilot engine task: {e}")
            logger.error(traceback.format_exc())
            
            # Send error to frontend
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'system_status': 'error'
            }
            await state.update_data(error_data)
            
            # Wait before retrying
            await asyncio.sleep(10.0)

# Start background task when server starts
@app.on_event("startup")
async def startup_event():
    """Start background tasks when server starts."""
    logger.info("Starting GridPilot Bridge Server")
    
    # Start the GridPilot engine task
    asyncio.create_task(gridpilot_engine_task())

# WebSocket endpoint for real-time data
@app.websocket("/ws/live-data")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming live system data to React frontend."""
    await websocket.accept()
    state.connected_clients.append(websocket)
    
    logger.info(f"WebSocket client connected. Total clients: {len(state.connected_clients)}")
    
    try:
        # Send current state immediately upon connection
        if state.latest_data:
            await websocket.send_text(json.dumps({
                'type': 'initial_state',
                'data': state.latest_data,
                'auction_history': state.auction_history[-100:],  # Last 100 entries
                'timestamp': datetime.now().isoformat()
            }))
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client (e.g., commands, settings)
                message = await websocket.receive_text()
                data = json.loads(message)
                
                # Handle different message types
                if data.get('type') == 'start_system':
                    state.is_running = True
                    await websocket.send_text(json.dumps({
                        'type': 'system_started',
                        'message': 'GridPilot engine started'
                    }))
                    
                elif data.get('type') == 'stop_system':
                    state.is_running = False
                    await websocket.send_text(json.dumps({
                        'type': 'system_stopped',
                        'message': 'GridPilot engine stopped'
                    }))
                    
                elif data.get('type') == 'ping':
                    await websocket.send_text(json.dumps({
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }))
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(state.connected_clients)}")

# REST API endpoints

@app.post("/api/explain")
async def explain_decision(request: Dict[str, Any]):
    """Generate LLM explanation for current system state."""
    try:
        question = request.get('question', 'Why did the system make this decision?')
        
        # For now, return a mock explanation
        # TODO: Integrate with Engineer D's LLM explainer
        explanation = f"The system allocated power based on current energy prices of ${state.latest_data.get('prices_now', 45.50):.2f}/MWh. High-value inference workloads received priority allocation because they can pay premium rates during peak demand periods."
        
        return {
            "explanation": explanation,
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85,
            "response_time_ms": 250,
            "context": {
                "question": question,
                "current_price": state.latest_data.get('prices_now', 0),
                "battery_soc": state.latest_data.get('battery_soc', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "engine_running": state.is_running,
        "connected_clients": len(state.connected_clients),
        "last_update": state.latest_data.get('timestamp', 'Never'),
        "auction_history_size": len(state.auction_history)
    }

@app.get("/api/auction-history")
async def get_auction_history(limit: int = 100):
    """Get recent auction history."""
    return {
        "auction_history": state.auction_history[-limit:],
        "total_count": len(state.auction_history),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/system-data")
async def get_system_data():
    """Get current system state (REST endpoint)."""
    return {
        "data": state.latest_data,
        "timestamp": datetime.now().isoformat(),
        "engine_running": state.is_running
    }

@app.post("/api/system/start")
async def start_system():
    """Start the GridPilot engine."""
    state.is_running = True
    return {
        "message": "GridPilot engine started",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/system/stop")
async def stop_system():
    """Stop the GridPilot engine."""
    state.is_running = False
    return {
        "message": "GridPilot engine stopped",
        "timestamp": datetime.now().isoformat()
    }

# Serve React build files (when ready)
try:
    app.mount("/static", StaticFiles(directory="ui/gridpilot-dashboard/dist"), name="static")
    logger.info("Serving React build files from ui/gridpilot-dashboard/dist")
except:
    logger.warning("React build files not found. Frontend will need to run on separate port.")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting GridPilot-GT Bridge Server on http://localhost:8000")
    logger.info("WebSocket endpoint: ws://localhost:8000/ws/live-data")
    logger.info("API documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "bridge_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 