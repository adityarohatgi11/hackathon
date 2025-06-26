# GridPilot-GT: Engineering Delegation Instructions

## 🎯 Project Status: READY FOR PARALLEL DEVELOPMENT

All foundational work is complete! The project scaffold is set up and the integration is tested. You can now start parallel development immediately.

## 📋 Pre-Delegation Checklist ✅

- ✅ Repository initialized and synced
- ✅ Dependencies locked and installed (`requirements.txt`)
- ✅ Complete module scaffolding with working stubs
- ✅ CI pipeline configured (GitHub Actions)
- ✅ Feature branches created for parallel development
- ✅ Integration tested end-to-end
- ✅ All modules pass basic smoke tests

## 🚀 Quick Start for Each Engineer

### For Engineer B (Lane B: Bidding & MPC)
```bash
git clone https://github.com/adityarohatgi11/hackathon.git
cd hackathon
git checkout lane-b-bidding-mpc
pip install -r requirements.txt
python main.py --simulate 1  # Verify system works
```

**Your Mission**: Implement sophisticated bidding strategies and Model Predictive Control
- **Primary Files**: `game_theory/bid_generators.py`, `game_theory/mpc_controller.py`
- **Contract**: See `README_CURSOR.md` → Lane B section
- **Dependencies**: CVXPY for optimization, comprehensive backtesting

### For Engineer C (Lane C: Auction & Dispatch)
```bash
git clone https://github.com/adityarohatgi11/hackathon.git
cd hackathon
git checkout lane-c-auction-dispatch
pip install -r requirements.txt
python main.py --simulate 1  # Verify system works
```

**Your Mission**: Implement VCG auction mechanisms and real-time dispatch
- **Primary Files**: `game_theory/vcg_auction.py`, `dispatch/dispatch_agent.py`
- **Contract**: See `README_CURSOR.md` → Lane C section
- **Focus**: Real-time performance (<100ms), safety protocols

### For Engineer D (Lane D: UI & LLM)
```bash
git clone https://github.com/adityarohatgi11/hackathon.git
cd hackathon
git checkout lane-d-ui-llm
pip install -r requirements.txt
python main.py --simulate 1  # Verify system works
```

**Your Mission**: Build beautiful dashboard and LLM integration
- **Primary Files**: `ui/dashboard.py`, `llm_integration/chat_interface.py`
- **Contract**: See `README_CURSOR.md` → Lane D section
- **Goal**: Beautiful Streamlit UI + local LLM for insights

## 🔄 Development Workflow

### 1. Daily Standup (Recommended)
- **When**: Start of each development session
- **What**: Quick sync on progress, blockers, integration points
- **Duration**: 5-10 minutes

### 2. Development Cycle
1. **Pull latest**: `git pull origin main` (check for any integration updates)
2. **Work on your branch**: Stay in your lane branch
3. **Test frequently**: Use `python main.py --simulate 1` to test integration
4. **Commit often**: Small, focused commits with clear messages
5. **Push regularly**: `git push` to backup your work

### 3. Integration Points
- **Every 2-4 hours**: Test your changes with main integration
- **Before major changes**: Coordinate with other engineers
- **End of session**: Ensure your code integrates cleanly

### 4. Code Review & Merge
1. **When ready**: Open PR from your lane branch to `main`
2. **Requirements**: All tests pass, lint clean, integration works
3. **Review**: Other engineers review for conflicts
4. **Merge**: Merge to main when approved

## 🔗 Integration Interface Points

### Data Flow Between Lanes
```
Lane A (Data) → Lane B (Bidding) → Lane C (Auction) → Lane D (UI)
      ↓              ↓                ↓              ↑
    API Data    Optimized Bids    Allocations    Insights
```

### Critical Dependencies
- **Lane B depends on**: Lane A forecasting output format
- **Lane C depends on**: Lane B bid vector format
- **Lane D depends on**: All lanes for dashboard data
- **All lanes depend on**: `config.toml` structure

### Shared Data Formats (DO NOT CHANGE)
- **Price DataFrame**: columns `['timestamp', 'price', 'volume']`
- **Forecast DataFrame**: columns `['timestamp', 'predicted_price', 'σ_energy', 'σ_hash', 'σ_token']`
- **Allocation Dict**: keys `['inference', 'training', 'cooling']`
- **Inventory Dict**: keys `['power_total', 'power_available', 'battery_soc', 'gpu_utilization']`

## 🧪 Testing Strategy

### Unit Tests (Your Responsibility)
- **Minimum**: 80% coverage for your modules
- **Run**: `pytest tests/test_your_module.py -v`
- **Add**: Tests for all major functions

### Integration Tests (Shared Responsibility)
- **Run**: `python main.py --simulate 1` (full integration)
- **Run**: `pytest tests/test_basic.py -v` (basic integration)
- **Add**: Tests for cross-module interactions

### Performance Tests
- **Lane B**: Bid optimization < 1 second
- **Lane C**: Dispatch response < 100ms
- **Lane D**: Dashboard load < 3 seconds

## 🚨 Critical Success Factors

### 1. Stick to Contracts
- **Interface Signatures**: DO NOT change function signatures in `README_CURSOR.md`
- **Data Formats**: DO NOT change shared data structures
- **Return Types**: Match exactly what other modules expect

### 2. Test Integration Early & Often
- **Every commit**: Test that main.py still works
- **Every feature**: Test your module with real data flow
- **Before PR**: Full integration test passes

### 3. Communication Protocol
- **Breaking Changes**: Notify all engineers immediately
- **API Changes**: Must be agreed upon by all affected parties
- **Blockers**: Escalate within 30 minutes

## 📊 Success Metrics

### Technical KPIs
- [ ] All modules integrate seamlessly
- [ ] System processes real market data
- [ ] Performance meets real-time requirements
- [ ] 80%+ test coverage across all modules
- [ ] CI pipeline always green

### Business KPIs
- [ ] UI provides clear, actionable insights
- [ ] Bidding strategy shows profit optimization
- [ ] Auction mechanism ensures fairness
- [ ] System handles edge cases gracefully

## 🆘 Troubleshooting

### Common Issues & Solutions

**Import Errors**:
```bash
# Solution: Ensure you're in the right directory and have dependencies
cd hackathon
pip install -r requirements.txt
```

**Integration Breaks**:
```bash
# Solution: Test against main branch
git checkout main
git pull
python main.py --simulate 1
git checkout your-lane-branch
# Fix your code to match main interfaces
```

**Merge Conflicts**:
```bash
# Solution: Rebase onto main
git fetch origin
git rebase origin/main
# Resolve conflicts, then continue
```

**Performance Issues**:
- Check if you're following the optimization requirements
- Profile your code with `python -m cProfile main.py --simulate 1`
- Focus on the critical path timing requirements

### Getting Help
1. **Technical Issues**: Check `README_CURSOR.md` contracts
2. **Integration Problems**: Test with minimal changes first
3. **Merge Conflicts**: Coordinate with other engineers
4. **Performance**: Profile and optimize critical sections

## 🎁 Bonus Features (If Time Permits)

### Lane B Enhancements
- Machine learning for bid optimization
- Multi-objective optimization (profit + reliability)
- Advanced risk management models

### Lane C Enhancements
- Real-time market monitoring
- Advanced emergency protocols
- Auction mechanism variations

### Lane D Enhancements
- Mobile-responsive design
- Advanced LLM prompting
- Real-time alerting system

## 🏁 Final Integration

### Hour 5-6: Integration Sprint
- All engineers work together on final integration
- Fix any remaining interface issues
- Performance optimization
- End-to-end testing
- Deployment preparation

### Demo Preparation
- Prepare 5-minute demo of your module
- Show key features and innovations
- Highlight technical achievements
- Practice the full system demo

---

## 💪 You've Got This!

The foundation is solid, the contracts are clear, and the system is tested. Focus on implementing your module's core functionality, test integration frequently, and coordinate with your team. 

**Remember**: Perfect is the enemy of good. Aim for working, tested, integrated code over complex features that break the system.

**Good luck building GridPilot-GT! 🚀⚡🎯**

# 🚀 Frontend Developer Delegation Guide
## GridPilot-GT Hackathon Frontend Development

---

## 🎯 **IMMEDIATE ACTION ITEMS FOR FRONTEND DEVELOPER**

### **Setup (15 minutes)**
```bash
# 1. Navigate to the hackathon directory
cd /path/to/hackathon

# 2. Create frontend workspace
mkdir -p ui/gridpilot-dashboard
cd ui/gridpilot-dashboard

# 3. Initialize Next.js project
npx create-next-app@latest . --typescript --tailwind --app --src-dir --import-alias "@/*"

# 4. Install required dependencies
npm install @tanstack/react-table echarts echarts-for-react @deck.gl/react @deck.gl/core mapbox-gl zustand framer-motion react-use-websocket ws @types/ws
npm install -D @types/mapbox-gl

# 5. Start development server
npm run dev
```

### **Backend Mock Server (20 minutes)**
Create this file to start developing immediately without waiting for backend:

```typescript
// ui/gridpilot-dashboard/src/lib/mock-server.ts
export const mockData = {
  systemState: {
    timestamp: new Date().toISOString(),
    prices_now: 45.50,
    prices_pred: [46.2, 47.1, 45.8, 44.3, 43.9],
    bids: {
      inference: 45.50,
      training: 38.20,
      cooling: 12.10
    },
    allocations: {
      inference: 125.3,
      training: 89.7,
      cooling: 45.2
    },
    payments: {
      inference: 42.10,
      training: 35.80,
      cooling: 12.10
    },
    soc: 0.734,
    cooling_kw: 45.2,
    expected_pnl: 156.80,
    carbon_intensity: 0.45
  },
  auctionHistory: [
    {
      time: "13:45:23",
      bidder: "Inference",
      bid: 45.50,
      allocation: 125.3,
      payment: 42.10
    },
    // Add more mock auction entries...
  ]
};

// Mock WebSocket for development
export class MockWebSocket {
  onmessage: ((event: MessageEvent) => void) | null = null;
  
  constructor(url: string) {
    // Simulate real-time data updates
    setInterval(() => {
      if (this.onmessage) {
        const updatedData = {
          ...mockData.systemState,
          timestamp: new Date().toISOString(),
          prices_now: 45.50 + (Math.random() - 0.5) * 10,
          soc: 0.734 + (Math.random() - 0.5) * 0.1
        };
        
        this.onmessage({
          data: JSON.stringify(updatedData)
        } as MessageEvent);
      }
    }, 1000);
  }
  
  close() {}
}
```

---

## 🏗️ **ARCHITECTURE OVERVIEW**

### **Your Responsibilities**
- **React Dashboard**: Modern, responsive UI with real-time charts
- **WebSocket Integration**: Live data streaming from backend
- **LLM Chat Interface**: Streaming AI explanations
- **3D Visualizations**: Globe view with site locations
- **Auction Tables**: Real-time VCG auction results

### **Backend Team Responsibilities** (In Progress)
- **FastAPI Bridge Server**: WebSocket + REST API endpoints
- **LLM Explainer**: AI explanations for system decisions
- **Core Engine**: Forecasting, optimization, dispatch
- **Data Pipeline**: Real-time MARA API integration

### **Integration Points**
```
React Frontend (Port 3000) ↔ FastAPI Bridge (Port 8000) ↔ Core Engine
                                     ↓
                            LLM Explainer Backend
```

---

## 📱 **COMPONENT DEVELOPMENT PRIORITY**

### **Phase 1: Foundation (Hours 0-2)**
1. **Layout & Navigation**
   ```typescript
   // src/components/layout/MainLayout.tsx
   // - Sidebar navigation
   // - Header with system status
   // - Dark theme setup
   ```

2. **Real-time Data Hook**
   ```typescript
   // src/hooks/useGridPilotData.ts
   // - WebSocket connection management
   // - State management with Zustand
   // - Error handling & reconnection
   ```

3. **Basic Dashboard Grid**
   ```typescript
   // src/app/dashboard/page.tsx
   // - Responsive grid layout
   // - Placeholder cards for charts
   ```

### **Phase 2: Core Charts (Hours 2-4)**
1. **Power Flow Chart**
   ```typescript
   // src/components/charts/PowerFlowChart.tsx
   // - Stacked area chart with ECharts
   // - Real-time updates (60fps)
   // - Smooth animations
   ```

2. **Battery SOC Gauge**
   ```typescript
   // src/components/charts/BatteryGauge.tsx
   // - Radial gauge with gradient colors
   // - Animated state changes
   ```

3. **Forecast Chart**
   ```typescript
   // src/components/charts/ForecastChart.tsx
   // - Line chart with uncertainty bands
   // - Price prediction visualization
   ```

### **Phase 3: Advanced Features (Hours 4-6)**
1. **3D Globe View**
   ```typescript
   // src/components/globe/GlobeView.tsx
   // - Deck.gl integration
   // - Site pins with power flow arcs
   // - Interactive controls
   ```

2. **Auction Ledger Table**
   ```typescript
   // src/components/tables/AuctionLedger.tsx
   // - TanStack Table with virtual scrolling
   // - Real-time row additions
   // - Export functionality
   ```

### **Phase 4: LLM Integration (Hours 6-8)**
1. **Chat Interface**
   ```typescript
   // src/components/chat/LLMChat.tsx
   // - Streaming message interface
   // - Context-aware questions
   // - Response animations
   ```

---

## 🛠️ **TECHNICAL SPECIFICATIONS**

### **Design System**
```css
/* tailwind.config.js - Add these custom colors */
module.exports = {
  theme: {
    extend: {
      colors: {
        'energy-blue': '#0ea5e9',
        'forecast-emerald': '#10b981',
        'battery-amber': '#f59e0b',
        'cooling-rose': '#f43f5e',
        'grid-bg': '#0f172a',
        'grid-surface': '#1e293b'
      }
    }
  }
}
```

### **State Management Structure**
```typescript
// src/store/gridpilot-store.ts
interface GridPilotState {
  // Real-time system data
  systemState: {
    timestamp: string;
    prices_now: number;
    prices_pred: number[];
    allocations: Record<string, number>;
    soc: number;
    cooling_kw: number;
    expected_pnl: number;
  };
  
  // Auction history
  auctionHistory: AuctionEntry[];
  
  // LLM chat
  chatMessages: ChatMessage[];
  
  // UI state
  isConnected: boolean;
  selectedView: 'dashboard' | 'globe' | 'auction' | 'chat';
  
  // Actions
  updateSystemState: (data: any) => void;
  addAuctionEntry: (entry: AuctionEntry) => void;
  sendChatMessage: (message: string) => void;
}
```

### **WebSocket Integration**
```typescript
// src/hooks/useGridPilotData.ts
import useWebSocket from 'react-use-websocket';

export const useGridPilotData = () => {
  const { lastMessage, connectionStatus } = useWebSocket(
    process.env.NODE_ENV === 'development' 
      ? null // Use mock data
      : 'ws://localhost:8000/ws/live-data',
    {
      shouldReconnect: () => true,
      reconnectInterval: 3000,
    }
  );

  // Parse and update state
  useEffect(() => {
    if (lastMessage?.data) {
      const data = JSON.parse(lastMessage.data);
      // Update Zustand store
    }
  }, [lastMessage]);
};
```

---

## 📊 **SAMPLE COMPONENTS TO GET STARTED**

### **1. Power Flow Chart Component**
```typescript
// src/components/charts/PowerFlowChart.tsx
import ReactECharts from 'echarts-for-react';
import { useGridPilotStore } from '@/store/gridpilot-store';

export const PowerFlowChart = () => {
  const { systemState } = useGridPilotStore();
  
  const option = {
    backgroundColor: 'transparent',
    grid: { top: 40, right: 40, bottom: 40, left: 60 },
    xAxis: {
      type: 'time',
      axisLine: { lineStyle: { color: '#475569' } }
    },
    yAxis: {
      type: 'value',
      name: 'Power (kW)',
      axisLine: { lineStyle: { color: '#475569' } }
    },
    series: [
      {
        name: 'Inference',
        type: 'line',
        areaStyle: { color: '#0ea5e9' },
        data: [[systemState.timestamp, systemState.allocations.inference]]
      },
      {
        name: 'Training', 
        type: 'line',
        areaStyle: { color: '#10b981' },
        data: [[systemState.timestamp, systemState.allocations.training]]
      }
    ]
  };

  return (
    <div className="bg-grid-surface rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-4 text-white">Power Allocation</h3>
      <ReactECharts 
        option={option} 
        style={{ height: '300px' }}
        opts={{ renderer: 'canvas' }}
      />
    </div>
  );
};
```

### **2. Battery SOC Gauge**
```typescript
// src/components/charts/BatteryGauge.tsx
import ReactECharts from 'echarts-for-react';

export const BatteryGauge = ({ soc }: { soc: number }) => {
  const option = {
    series: [{
      type: 'gauge',
      startAngle: 180,
      endAngle: 0,
      center: ['50%', '75%'],
      radius: '90%',
      min: 0,
      max: 1,
      splitNumber: 8,
      axisLine: {
        lineStyle: {
          width: 6,
          color: [
            [0.25, '#f43f5e'],
            [0.5, '#f59e0b'], 
            [1, '#10b981']
          ]
        }
      },
      pointer: {
        icon: 'path://M12.02,2.91L12.02,2.91...',
        length: '12%',
        width: 30,
        offsetCenter: [0, '-60%']
      },
      gauge: {
        axisLine: { lineStyle: { width: 30 } },
        splitLine: { distance: -30, length: 30 }
      },
      data: [{ value: soc, name: 'SOC' }]
    }]
  };

  return (
    <div className="bg-grid-surface rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-2 text-white">Battery SOC</h3>
      <ReactECharts option={option} style={{ height: '200px' }} />
      <div className="text-center text-2xl font-bold text-white">
        {(soc * 100).toFixed(1)}%
      </div>
    </div>
  );
};
```

---

## 🔗 **BACKEND INTEGRATION ENDPOINTS**

### **When Backend is Ready, Replace Mock Data With:**

```typescript
// Real WebSocket connection
const WEBSOCKET_URL = 'ws://localhost:8000/ws/live-data';

// REST API endpoints
const API_BASE = 'http://localhost:8000/api';

// LLM explanation requests
const explainDecision = async (question: string) => {
  const response = await fetch(`${API_BASE}/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  return response.json();
};
```

---

## 🎨 **UI/UX REQUIREMENTS**

### **Visual Polish Checklist**
- [ ] Smooth 60fps animations on chart updates
- [ ] Consistent dark theme throughout
- [ ] Loading states for all async operations
- [ ] Error boundaries with graceful fallbacks
- [ ] Responsive design (desktop + tablet)
- [ ] Accessible color contrasts (WCAG AA)
- [ ] Micro-interactions on hover/click

### **Performance Requirements**
- [ ] First contentful paint < 1.5s
- [ ] WebSocket reconnection handling
- [ ] Chart rendering optimization
- [ ] Virtual scrolling for large tables
- [ ] Lazy loading for 3D components

---

## 📞 **COMMUNICATION PROTOCOL**

### **Daily Sync Points**
1. **Morning Standup**: Share progress, blockers
2. **Midday Check**: Demo current state
3. **Evening Review**: Integration testing

### **Slack/Discord Updates**
- Post screenshots of new components
- Share Figma/design iterations
- Flag any backend API changes needed

### **Code Collaboration**
- Work on separate branch: `frontend-development`
- Regular commits with descriptive messages
- Create PR when ready for backend integration

---

## 🚀 **SUCCESS METRICS**

### **Technical Goals**
- [ ] All 4 main views implemented
- [ ] Real-time data flowing smoothly
- [ ] LLM chat interface working
- [ ] 3D globe rendering correctly
- [ ] Auction table with live updates

### **Hackathon Demo Goals**
- [ ] Stunning visual presentation
- [ ] Smooth live demo without crashes
- [ ] "Wow factor" moments (3D globe, AI chat)
- [ ] Professional, production-ready feel

---

## 🆘 **TROUBLESHOOTING & SUPPORT**

### **Common Issues**
1. **WebSocket Connection Fails**: Use mock data initially
2. **Chart Performance**: Reduce update frequency to 30fps
3. **3D Rendering Issues**: Add WebGL detection fallback
4. **Build Errors**: Check Node.js version (18+)

### **Getting Help**
- **Backend Questions**: Tag @backend-team in Slack
- **Design Decisions**: Reference FRONTEND_ROADMAP.md
- **Technical Blockers**: Create GitHub issue with `frontend` label

---

**🎯 Remember: Your frontend will be the first impression judges get. Make it stunning, smooth, and professional. The backend team has the data pipeline ready - now make it shine!** 