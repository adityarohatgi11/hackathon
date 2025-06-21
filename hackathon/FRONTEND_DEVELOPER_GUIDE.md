# Frontend Developer Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Start the Backend Server
```bash
cd /path/to/hackathon
./start_api_server.sh
```

The server will start at `http://localhost:8000` with:
- **Swagger UI**: http://localhost:8000/docs
- **API Base**: http://localhost:8000/api/v1/
- **WebSocket**: ws://localhost:8000/ws/live-data

### 2. Test the API
Open http://localhost:8000/docs in your browser and try:
- `GET /api/v1/system/status` - Check if everything is working
- `GET /api/v1/data/prices` - Get live energy price data
- `POST /api/v1/ml/forecast` - Generate ML price forecasts

### 3. Connect via WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live-data');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Live data:', data);
};
```

---

## 🎯 What You Get

### Real Live Data
- **119 real price records** from MARA Hackathon API
- **999.97 MW power capacity** available
- **68.7% battery SOC** 
- **$3.00/MWh current energy price**

### ML & Quantitative Features
- **Prophet + Random Forest** forecasting (MAE=0.22)
- **104 engineered features** (technical indicators, seasonality, etc.)
- **VCG auction optimization** with 493 kW allocations
- **<1.5s end-to-end latency** for complete trading cycle

### Ready-to-Use Components
- **React hooks** for WebSocket data
- **API client functions** for all endpoints
- **Dashboard component examples**
- **Error handling** and loading states

---

## 📊 Key Data Structures

### Live System Data (WebSocket)
```javascript
{
  "timestamp": "2025-06-21T15:30:00Z",
  "prices": { "current": 3.00, "trend": "stable" },
  "inventory": { "power_available": 999968, "battery_soc": 0.687 },
  "allocation": { "inference": 197.2, "training": 147.9, "cooling": 147.9 }
}
```

### Forecast Data
```javascript
{
  "forecast": [
    { "timestamp": "...", "predicted_price": 2.95, "lower_bound": 2.80, "upper_bound": 3.10 }
  ],
  "metadata": { "avg_predicted_price": 2.89, "features_used": 104 },
  "performance_metrics": { "processing_time_ms": 850 }
}
```

### Optimization Results
```javascript
{
  "allocation": { "inference": 197.2, "training": 147.9, "cooling": 147.9 },
  "optimization_metrics": { "total_allocation_kw": 493.0, "utilization": 0.000493 },
  "constraints_satisfied": true
}
```

---

## 🛠️ Development Workflow

### 1. Build Your Frontend
```bash
# In your frontend directory
npx create-react-app gridpilot-dashboard
cd gridpilot-dashboard
npm install recharts axios  # or your preferred chart/HTTP library
```

### 2. Copy API Client Code
Use the examples from `API_DOCUMENTATION.md`:
- React hooks for WebSocket
- API client functions
- Dashboard component examples

### 3. Key Features to Implement
- **Real-time price chart** (WebSocket data)
- **Battery SOC gauge** (circular progress)
- **Power allocation display** (inference/training/cooling)
- **Forecast visualization** (line chart with uncertainty bands)
- **System status indicators** (health, alerts, uptime)

### 4. Recommended UI Libraries
- **Charts**: Recharts, Chart.js, or D3.js
- **Components**: Material-UI, Ant Design, or Chakra UI
- **Styling**: Tailwind CSS or styled-components
- **Icons**: React Icons or Heroicons

---

## 🎨 UI/UX Recommendations

### Dashboard Layout
```
┌─────────────────────────────────────────────────────────┐
│ Header: GridPilot-GT | Status: 🟢 | SOC: 68.7%         │
├─────────────────┬───────────────────┬───────────────────┤
│ Live Price      │ 24h Forecast      │ Power Allocation  │
│ $3.00/MWh      │ Chart with        │ Inference: 197kW  │
│ Trend: Stable   │ uncertainty bands │ Training: 148kW   │
│                 │                   │ Cooling: 148kW    │
├─────────────────┼───────────────────┼───────────────────┤
│ System Metrics  │ ML Performance    │ Controls          │
│ Uptime: 1h      │ MAE: 0.22        │ [Forecast]        │
│ Latency: 1.4s   │ Features: 104     │ [Optimize]        │
│ Tests: 92.8%    │ Confidence: High  │ [Full Cycle]      │
└─────────────────┴───────────────────┴───────────────────┘
```

### Color Scheme (Energy Industry)
- **Primary**: Deep blue (#1e3a8a)
- **Success**: Green (#10b981) 
- **Warning**: Amber (#f59e0b)
- **Error**: Red (#ef4444)
- **Background**: Light gray (#f8fafc)

---

## 🔧 Troubleshooting

### Backend Not Starting?
```bash
# Check if port 8000 is available
lsof -i :8000

# Install missing dependencies
pip install fastapi uvicorn pandas numpy
```

### WebSocket Connection Issues?
```javascript
// Add reconnection logic
const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8000/ws/live-data');
    ws.onclose = () => {
        setTimeout(connectWebSocket, 5000); // Reconnect after 5s
    };
    return ws;
};
```

### CORS Issues?
The backend has CORS enabled for all origins. If you still have issues:
```javascript
// Make sure you're using the correct base URL
const API_BASE = 'http://localhost:8000';
```

---

## 📞 Support

### Documentation
- **Full API Docs**: `API_DOCUMENTATION.md`
- **ML Strategy**: `ML_QUANTITATIVE_STRATEGY_OVERVIEW.md`
- **Interactive Docs**: http://localhost:8000/docs

### Quick Test Commands
```bash
# Test API connectivity
curl http://localhost:8000/health

# Get system status
curl http://localhost:8000/api/v1/system/status

# Get live prices
curl http://localhost:8000/api/v1/data/prices
```

### Performance Expectations
- **System Status**: <100ms response
- **Price Data**: <200ms response  
- **ML Forecast**: <2000ms response
- **Optimization**: <500ms response
- **WebSocket Updates**: Every 5 seconds

---

## 🎉 You're Ready!

Your backend is now running with:
- ✅ **Real MARA API data** (119 price records)
- ✅ **ML forecasting** (Prophet + Random Forest)
- ✅ **VCG optimization** (493 kW allocation)
- ✅ **WebSocket streaming** (live updates)
- ✅ **Complete documentation** (examples + API reference)

Start building your frontend and you'll have live energy trading data flowing immediately! 