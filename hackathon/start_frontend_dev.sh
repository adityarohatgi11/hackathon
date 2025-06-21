#!/bin/bash

# 🚀 GridPilot-GT Frontend Development Startup Script
# This script sets up the complete development environment for the frontend developer

set -e  # Exit on any error

echo "🚀 Starting GridPilot-GT Frontend Development Environment"
echo "========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    print_error "Please run this script from the hackathon root directory"
    exit 1
fi

# Step 1: Install Python dependencies
print_status "Installing Python dependencies..."
pip install fastapi uvicorn websockets python-multipart
print_success "Python dependencies installed"

# Step 2: Create UI directory structure
print_status "Setting up frontend directory structure..."
mkdir -p ui/gridpilot-dashboard
cd ui/gridpilot-dashboard

# Step 3: Initialize Next.js project (if not already exists)
if [ ! -f "package.json" ]; then
    print_status "Initializing Next.js project..."
    npx create-next-app@latest . --typescript --tailwind --app --src-dir --import-alias "@/*" --yes
    print_success "Next.js project initialized"
else
    print_warning "Next.js project already exists, skipping initialization"
fi

# Step 4: Install frontend dependencies
print_status "Installing frontend dependencies..."
npm install @tanstack/react-table echarts echarts-for-react @deck.gl/react @deck.gl/core mapbox-gl zustand framer-motion react-use-websocket ws @types/ws
npm install -D @types/mapbox-gl
print_success "Frontend dependencies installed"

# Step 5: Create mock data file
print_status "Creating mock data file..."
mkdir -p src/lib
cat > src/lib/mock-server.ts << 'EOF'
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
    }
  ]
};

export class MockWebSocket {
  onmessage: ((event: MessageEvent) => void) | null = null;
  
  constructor(url: string) {
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
EOF
print_success "Mock data file created"

# Step 6: Update Tailwind config with custom colors
print_status "Updating Tailwind configuration..."
cat > tailwind.config.ts << 'EOF'
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        'energy-blue': '#0ea5e9',
        'forecast-emerald': '#10b981',
        'battery-amber': '#f59e0b',
        'cooling-rose': '#f43f5e',
        'grid-bg': '#0f172a',
        'grid-surface': '#1e293b'
      },
    },
  },
  plugins: [],
};
export default config;
EOF
print_success "Tailwind configuration updated"

# Step 7: Create basic layout component
print_status "Creating basic layout component..."
mkdir -p src/components/layout
cat > src/components/layout/MainLayout.tsx << 'EOF'
'use client';

import React from 'react';

interface MainLayoutProps {
  children: React.ReactNode;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen bg-grid-bg text-white">
      <header className="bg-grid-surface border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-energy-blue">
            🔋 GridPilot-GT
          </h1>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-forecast-emerald rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-300">Live</span>
            </div>
          </div>
        </div>
      </header>
      
      <div className="flex">
        <nav className="w-64 bg-grid-surface min-h-screen p-4">
          <ul className="space-y-2">
            <li>
              <a href="#" className="block p-3 rounded-lg bg-energy-blue text-white">
                📊 Dashboard
              </a>
            </li>
            <li>
              <a href="#" className="block p-3 rounded-lg hover:bg-gray-700 text-gray-300">
                🌍 Globe View
              </a>
            </li>
            <li>
              <a href="#" className="block p-3 rounded-lg hover:bg-gray-700 text-gray-300">
                🏛️ Auction Ledger
              </a>
            </li>
            <li>
              <a href="#" className="block p-3 rounded-lg hover:bg-gray-700 text-gray-300">
                💬 AI Chat
              </a>
            </li>
          </ul>
        </nav>
        
        <main className="flex-1 p-6">
          {children}
        </main>
      </div>
    </div>
  );
};
EOF
print_success "Basic layout component created"

# Step 8: Update main page
print_status "Creating dashboard page..."
cat > src/app/page.tsx << 'EOF'
'use client';

import { MainLayout } from '@/components/layout/MainLayout';
import { mockData } from '@/lib/mock-server';

export default function Home() {
  const { systemState } = mockData;

  return (
    <MainLayout>
      <div className="space-y-6">
        <h2 className="text-3xl font-bold">System Dashboard</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Current Price Card */}
          <div className="bg-grid-surface rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-300 mb-2">Current Price</h3>
            <div className="text-3xl font-bold text-energy-blue">
              ${systemState.prices_now.toFixed(2)}
            </div>
            <div className="text-sm text-gray-400">per MWh</div>
          </div>
          
          {/* Battery SOC Card */}
          <div className="bg-grid-surface rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-300 mb-2">Battery SOC</h3>
            <div className="text-3xl font-bold text-battery-amber">
              {(systemState.soc * 100).toFixed(1)}%
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
              <div 
                className="bg-battery-amber h-2 rounded-full transition-all duration-300"
                style={{ width: `${systemState.soc * 100}%` }}
              ></div>
            </div>
          </div>
          
          {/* Expected P&L Card */}
          <div className="bg-grid-surface rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-300 mb-2">Expected P&L</h3>
            <div className="text-3xl font-bold text-forecast-emerald">
              ${systemState.expected_pnl.toFixed(2)}
            </div>
            <div className="text-sm text-gray-400">next hour</div>
          </div>
          
          {/* Cooling Load Card */}
          <div className="bg-grid-surface rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-300 mb-2">Cooling Load</h3>
            <div className="text-3xl font-bold text-cooling-rose">
              {systemState.cooling_kw.toFixed(1)} kW
            </div>
            <div className="text-sm text-gray-400">current</div>
          </div>
        </div>
        
        {/* Power Allocations */}
        <div className="bg-grid-surface rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">Power Allocations</h3>
          <div className="space-y-4">
            {Object.entries(systemState.allocations).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <span className="capitalize text-gray-300">{key}</span>
                <div className="flex items-center space-x-4">
                  <div className="w-32 bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-energy-blue h-2 rounded-full"
                      style={{ width: `${(value / 200) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-white font-mono">{value.toFixed(1)} kW</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="bg-yellow-900/20 border border-yellow-500 rounded-lg p-4">
          <h4 className="font-semibold text-yellow-300">🚧 Development Mode</h4>
          <p className="text-yellow-200 text-sm mt-1">
            Using mock data. Backend integration will provide real-time updates.
          </p>
        </div>
      </div>
    </MainLayout>
  );
}
EOF
print_success "Dashboard page created"

# Go back to project root
cd ../..

# Step 9: Create development startup script
print_status "Creating development startup script..."
cat > start_dev_servers.sh << 'EOF'
#!/bin/bash

# Start both backend and frontend development servers

echo "🚀 Starting GridPilot-GT Development Servers"
echo "============================================="

# Function to kill background processes on exit
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}
trap cleanup EXIT

# Start backend bridge server
echo "Starting FastAPI bridge server on port 8000..."
python bridge_server.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend development server
echo "Starting React frontend on port 3000..."
cd ui/gridpilot-dashboard
npm run dev &
FRONTEND_PID=$!
cd ../..

echo ""
echo "✅ Development servers started!"
echo "   🔗 Frontend: http://localhost:3000"
echo "   🔗 Backend API: http://localhost:8000"
echo "   🔗 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user to stop
wait
EOF

chmod +x start_dev_servers.sh
print_success "Development startup script created"

# Final instructions
echo ""
echo "🎉 Frontend Development Environment Setup Complete!"
echo "=================================================="
echo ""
echo "📋 What was created:"
echo "   ✅ Next.js TypeScript project in ui/gridpilot-dashboard/"
echo "   ✅ All required dependencies installed"
echo "   ✅ Mock data system for independent development"
echo "   ✅ Custom Tailwind theme with GridPilot colors"
echo "   ✅ Basic layout and dashboard components"
echo "   ✅ Development startup script"
echo ""
echo "🚀 To start development:"
echo "   ./start_dev_servers.sh"
echo ""
echo "📖 Next steps for your frontend developer:"
echo "   1. Run the startup script above"
echo "   2. Open http://localhost:3000 to see the dashboard"
echo "   3. Start building components using the mock data"
echo "   4. Reference DELEGATION_INSTRUCTIONS.md for detailed guidance"
echo "   5. Follow the component development priority in the guide"
echo ""
echo "🔗 Key files created:"
echo "   📄 ui/gridpilot-dashboard/src/lib/mock-server.ts (mock data)"
echo "   📄 ui/gridpilot-dashboard/src/components/layout/MainLayout.tsx (layout)"
echo "   📄 ui/gridpilot-dashboard/src/app/page.tsx (dashboard)"
echo "   📄 start_dev_servers.sh (startup script)"
echo ""
echo "💡 The frontend can now develop independently using mock data!"
echo "   When backend integration is ready, simply switch from mock to real WebSocket."
echo ""
print_success "Ready for frontend development! 🎯" 