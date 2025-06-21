# GridPilot-GT FastAPI Backend Documentation

## Quick Start

### Starting the Server
```bash
# Option 1: Use the startup script (recommended)
./start_api_server.sh

# Option 2: Direct uvicorn command
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### Server URLs
- **Base URL**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`
- **WebSocket**: `ws://localhost:8000/ws/live-data`

---

## API Endpoints Overview

### System Status & Health
- `GET /` - Root endpoint with service info
- `GET /health` - Simple health check
- `GET /api/v1/system/status` - Comprehensive system status

### Data Endpoints
- `GET /api/v1/data/prices` - Market price data
- `GET /api/v1/data/inventory` - System inventory & operational status

### ML & Forecasting
- `POST /api/v1/ml/forecast` - Generate price forecasts

### Optimization & Trading
- `POST /api/v1/optimization/allocate` - Optimize resource allocation
- `POST /api/v1/dispatch/execute` - Execute dispatch orders
- `POST /api/v1/system/run-full-cycle` - Run complete optimization cycle

### Real-time Data
- `WebSocket /ws/live-data` - Real-time data streaming

---

## Detailed API Reference

### 1. System Status

#### `GET /api/v1/system/status`
Get comprehensive system health and performance metrics.

**Response:**
```json
{
  "status": "operational",
  "timestamp": "2025-06-21T15:30:00Z",
  "api_connectivity": {
    "overall_status": "operational",
    "prices_available": true,
    "api_key_configured": true
  },
  "system_metrics": {
    "last_optimization_time": 0.45,
    "total_allocation_kw": 493.0,
    "constraints_satisfied": true,
    "uptime": 3600
  },
  "performance_summary": {
    "forecaster_ready": true,
    "last_data_update": true,
    "websocket_clients": 2,
    "uptime_seconds": 3600,
    "last_allocation_kw": 493.0
  }
}
```

### 2. Market Data

#### `GET /api/v1/data/prices?hours_back=24`
Get current and historical energy prices.

**Parameters:**
- `hours_back` (optional): Number of hours of historical data (default: 24)

**Response:**
```json
{
  "data": [
    {
      "timestamp": "2025-06-21T15:00:00Z",
      "energy_price": 3.00,
      "hash_price": 8.0,
      "token_price": 3.0,
      "volume": 1000,
      "volatility_24h": 0.39
    }
  ],
  "metadata": {
    "total_records": 119,
    "latest_price": 3.00,
    "price_range": {
      "min": 2.95,
      "max": 3.05
    },
    "timestamp": "2025-06-21T15:30:00Z"
  }
}
```

#### `GET /api/v1/data/inventory`
Get current system inventory and operational status.

**Response:**
```json
{
  "power": {
    "total_kw": 1000000,
    "available_kw": 999968,
    "used_kw": 32,
    "utilization": 0.000032
  },
  "battery": {
    "soc": 0.687,
    "capacity_mwh": 1.0,
    "max_power_kw": 250.0
  },
  "gpu": {
    "utilization": 0.8,
    "cooling_load_kw": 15.2
  },
  "operational": {
    "temperature": 68.5,
    "efficiency": 0.92,
    "status": "operational",
    "alerts": []
  },
  "timestamp": "2025-06-21T15:30:00Z"
}
```

### 3. ML Forecasting

#### `POST /api/v1/ml/forecast`
Generate ML-based price forecasts using Prophet + ensemble models.

**Request Body:**
```json
{
  "periods": 24,
  "use_advanced": false
}
```

**Response:**
```json
{
  "forecast": [
    {
      "timestamp": "2025-06-21T16:00:00Z",
      "predicted_price": 2.95,
      "lower_bound": 2.80,
      "upper_bound": 3.10,
      "uncertainty_energy": 0.15,
      "method": "combined"
    }
  ],
  "metadata": {
    "periods": 24,
    "model_type": "standard",
    "features_used": 104,
    "avg_predicted_price": 2.89,
    "price_volatility": 0.12
  },
  "performance_metrics": {
    "processing_time_ms": 850,
    "forecast_accuracy": "N/A",
    "model_confidence": "high"
  }
}
```

### 4. Optimization & Allocation

#### `POST /api/v1/optimization/allocate`
Optimize resource allocation using VCG auction and MPC.

**Request Body:**
```json
{
  "forecast_periods": 24,
  "risk_aversion": 0.5,
  "lambda_deg": 0.0002
}
```

**Response:**
```json
{
  "allocation": {
    "inference": 197.2,
    "training": 147.9,
    "cooling": 147.9
  },
  "payments": {
    "inference": 0.0,
    "training": 0.0,
    "cooling": 0.0
  },
  "optimization_metrics": {
    "processing_time_ms": 45,
    "social_welfare": 1250.5,
    "allocation_efficiency": 0.493,
    "total_allocation_kw": 493.0,
    "utilization": 0.000493
  },
  "constraints_satisfied": true
}
```

#### `POST /api/v1/dispatch/execute`
Execute dispatch based on current allocation.

**Response:**
```json
{
  "dispatch_result": {
    "status": "success",
    "payload": {
      "allocation": {
        "air_miners": 0,
        "inference": 197.2,
        "training": 147.9,
        "cooling": 147.9
      },
      "power_requirements": {
        "total_power_kw": 493.0,
        "cooling_power_kw": 8.4,
        "battery_power_kw": 0
      },
      "constraints_satisfied": true,
      "system_state": {
        "soc": 0.687,
        "utilization": 0.000493,
        "temperature": 68.5,
        "efficiency": 0.92
      }
    },
    "cooling_requirements": {
      "cooling_kw": 8.4,
      "cop": 3.5,
      "efficiency": 0.85
    }
  },
  "performance_metrics": {
    "processing_time_ms": 12,
    "constraints_satisfied": true,
    "total_power_kw": 493.0
  },
  "timestamp": "2025-06-21T15:30:00Z"
}
```

#### `POST /api/v1/system/run-full-cycle`
Run complete GridPilot-GT optimization cycle (data → forecast → optimize → dispatch).

**Request Body:**
```json
{
  "simulate": true
}
```

**Response:**
```json
{
  "cycle_result": {
    "success": true,
    "elapsed_time": 1.403,
    "soc": 0.687,
    "total_power": 493.0,
    "revenue": {
      "inference": 0.0,
      "training": 0.0,
      "cooling": 0.0
    },
    "power_requirements": {
      "total_power_kw": 493.0,
      "cooling_power_kw": 8.4
    },
    "constraints_satisfied": true,
    "performance_metrics": {
      "build_time_ms": 140.3,
      "total_time_ms": 1403.0
    }
  },
  "performance_metrics": {
    "total_time_ms": 1403.0,
    "success": true
  },
  "timestamp": "2025-06-21T15:30:00Z"
}
```

### 5. WebSocket Real-time Data

#### `WebSocket /ws/live-data`
Real-time data streaming for dashboard updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live-data');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Real-time data:', data);
};
```

**Message Format:**
```json
{
  "timestamp": "2025-06-21T15:30:00Z",
  "prices": {
    "current": 3.00,
    "trend": "stable"
  },
  "inventory": {
    "power_available": 999968,
    "battery_soc": 0.687
  },
  "allocation": {
    "inference": 197.2,
    "training": 147.9,
    "cooling": 147.9
  },
  "system_metrics": {
    "last_optimization_time": 0.45,
    "total_allocation_kw": 493.0,
    "constraints_satisfied": true,
    "uptime": 3600
  }
}
```

---

## Frontend Integration Examples

### React Hook for Real-time Data
```javascript
import { useState, useEffect } from 'react';

export function useGridPilotData() {
  const [data, setData] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/live-data');
    
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      setData(JSON.parse(event.data));
    };

    return () => ws.close();
  }, []);

  return { data, connected };
}
```

### API Client Functions
```javascript
const API_BASE = 'http://localhost:8000';

export const gridPilotAPI = {
  // Get system status
  async getStatus() {
    const response = await fetch(`${API_BASE}/api/v1/system/status`);
    return response.json();
  },

  // Get market prices
  async getPrices(hoursBack = 24) {
    const response = await fetch(`${API_BASE}/api/v1/data/prices?hours_back=${hoursBack}`);
    return response.json();
  },

  // Get inventory
  async getInventory() {
    const response = await fetch(`${API_BASE}/api/v1/data/inventory`);
    return response.json();
  },

  // Generate forecast
  async generateForecast(periods = 24, useAdvanced = false) {
    const response = await fetch(`${API_BASE}/api/v1/ml/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ periods, use_advanced: useAdvanced })
    });
    return response.json();
  },

  // Optimize allocation
  async optimizeAllocation(forecastPeriods = 24, riskAversion = 0.5) {
    const response = await fetch(`${API_BASE}/api/v1/optimization/allocate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        forecast_periods: forecastPeriods, 
        risk_aversion: riskAversion 
      })
    });
    return response.json();
  },

  // Run full cycle
  async runFullCycle(simulate = true) {
    const response = await fetch(`${API_BASE}/api/v1/system/run-full-cycle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ simulate })
    });
    return response.json();
  }
};
```

### Dashboard Component Example
```javascript
import React, { useState, useEffect } from 'react';
import { gridPilotAPI, useGridPilotData } from './api';

export function GridPilotDashboard() {
  const { data: liveData, connected } = useGridPilotData();
  const [forecast, setForecast] = useState(null);
  const [allocation, setAllocation] = useState(null);

  // Generate forecast
  const handleForecast = async () => {
    const result = await gridPilotAPI.generateForecast(24);
    setForecast(result);
  };

  // Optimize allocation
  const handleOptimize = async () => {
    const result = await gridPilotAPI.optimizeAllocation();
    setAllocation(result);
  };

  return (
    <div className="dashboard">
      <div className="status-bar">
        <span>WebSocket: {connected ? '🟢 Connected' : '🔴 Disconnected'}</span>
        <span>Current Price: ${liveData?.prices?.current}/MWh</span>
        <span>Battery SOC: {(liveData?.inventory?.battery_soc * 100).toFixed(1)}%</span>
      </div>

      <div className="controls">
        <button onClick={handleForecast}>Generate Forecast</button>
        <button onClick={handleOptimize}>Optimize Allocation</button>
      </div>

      {forecast && (
        <div className="forecast-section">
          <h3>24-Hour Price Forecast</h3>
          <p>Avg Price: ${forecast.metadata.avg_predicted_price.toFixed(2)}/MWh</p>
          <p>Processing Time: {forecast.performance_metrics.processing_time_ms.toFixed(0)}ms</p>
        </div>
      )}

      {allocation && (
        <div className="allocation-section">
          <h3>Optimal Allocation</h3>
          <p>Inference: {allocation.allocation.inference.toFixed(1)} kW</p>
          <p>Training: {allocation.allocation.training.toFixed(1)} kW</p>
          <p>Cooling: {allocation.allocation.cooling.toFixed(1)} kW</p>
          <p>Total: {allocation.optimization_metrics.total_allocation_kw.toFixed(1)} kW</p>
        </div>
      )}
    </div>
  );
}
```

---

## Error Handling

All endpoints return standard HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **500**: Internal Server Error

Error response format:
```json
{
  "detail": "Error description"
}
```

---

## Performance Considerations

### Response Times (Target)
- System status: < 100ms
- Price data: < 200ms
- Forecasting: < 2000ms
- Optimization: < 500ms
- Dispatch: < 100ms
- Full cycle: < 2000ms

### Rate Limiting
- No rate limiting currently implemented
- Consider implementing for production use

### Caching
- System automatically caches recent data
- Background refresh every 2 minutes
- WebSocket updates every 5 seconds

---

## Development Tips

1. **Use the interactive documentation** at `http://localhost:8000/docs` to test endpoints
2. **Monitor server logs** for debugging information
3. **WebSocket connection** handles reconnection automatically
4. **All timestamps** are in ISO 8601 format (UTC)
5. **Power values** are in kW, energy in MWh
6. **Prices** are in $/MWh
7. **Battery SOC** is a decimal (0.0 - 1.0)

---

## Production Deployment

For production deployment:

1. **Set proper CORS origins** in `api_server.py`
2. **Add authentication/authorization** 
3. **Implement rate limiting**
4. **Add request logging**
5. **Use a production ASGI server** (e.g., Gunicorn + Uvicorn)
6. **Set up monitoring and alerting**

Example production command:
```bash
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
``` 