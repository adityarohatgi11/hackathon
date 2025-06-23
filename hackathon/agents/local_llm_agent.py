from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import pandas as pd

# Try to import llama-cpp-python for local LLM
try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

# Try to import Anthropic for Claude API
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class LocalLLMAgent(BaseAgent):
    """Intelligent agent using Claude API for deep analysis and strategic insights."""

    subscribe_topics = ["strategy-action", "forecast", "feature-vector"]
    publish_topic = "llm-analysis"

    def __init__(self, model_path: str = None):
        super().__init__(name="LocalLLMAgent")
        self._model_path = model_path or self._find_model()
        self._llm = None
        self._claude_client = None
        self._use_claude = False
        self._use_local_llm = False
        
        # Initialize Claude API first (preferred)
        if HAS_ANTHROPIC:
            self._initialize_claude()
        
        # Fallback to local LLM
        if not self._use_claude and HAS_LLAMA and self._model_path:
            self._initialize_llm()
        
        if not self._use_claude and not self._use_local_llm:
            logger.warning("[%s] No AI model available, using rule-based analysis", self.name)

    def _find_model(self) -> str | None:
        """Find available GGML model file."""
        possible_paths = [
            "models/llama-2-7b-chat.q4_0.gguf",
            "models/mistral-7b-instruct.q4_0.gguf", 
            "models/codellama-7b.q4_0.gguf",
            "/usr/local/share/llama.cpp/models/",
            os.path.expanduser("~/.cache/huggingface/hub/")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    return path
                elif os.path.isdir(path):
                    # Look for .gguf files in directory
                    for file in os.listdir(path):
                        if file.endswith(('.gguf', '.bin')) and 'q4' in file.lower():
                            return os.path.join(path, file)
        
        logger.warning("No local LLM model found. Download a GGUF model to enable local LLM analysis.")
        return None

    def _initialize_llm(self):
        """Initialize local LLM."""
        try:
            self._llm = Llama(
                model_path=self._model_path,
                n_ctx=2048,  # Context window
                n_batch=512,  # Batch size
                verbose=False,
                n_gpu_layers=0,  # Use CPU only for compatibility
            )
            logger.info("[%s] Loaded local LLM from %s", self.name, self._model_path)
        except Exception as exc:
            logger.warning("[%s] Failed to load LLM: %s", self.name, exc)
            self._use_local_llm = False

    def _initialize_claude(self):
        """Initialize Anthropic Claude API."""
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                # Try to load from config
                try:
                    import toml
                    config = toml.load("config.toml")
                    api_key = config.get("ai", {}).get("anthropic_api_key")
                except:
                    pass
            
            if api_key:
                self._claude_client = anthropic.Anthropic(api_key=api_key)
                # Test the connection
                test_response = self._claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Test"}]
                )
                self._use_claude = True
                logger.info("[%s] Successfully initialized Claude API", self.name)
            else:
                logger.warning("[%s] No Anthropic API key found", self.name)
        except Exception as exc:
            logger.warning("[%s] Failed to initialize Claude: %s", self.name, exc)

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Analyze incoming data and generate insights."""
        try:
            message_type = self._identify_message_type(message)
            
            if self._use_claude:
                analysis = self._generate_claude_analysis(message, message_type)
            elif self._use_local_llm:
                analysis = self._generate_llm_analysis(message, message_type)
            else:
                analysis = self._generate_rule_based_analysis(message, message_type)
            
            return {
                "timestamp": pd.Timestamp.now().isoformat(),
                "analysis": analysis,
                "message_type": message_type,
                "source": self.name,
            }
            
        except Exception as exc:
            logger.exception("[%s] Error in analysis: %s", self.name, exc)
            return None

    def _identify_message_type(self, message: Dict[str, Any]) -> str:
        """Identify the type of incoming message."""
        if "action" in message:
            return "strategy"
        elif "forecast" in message:
            return "forecast"
        elif "features" in message:
            return "data"
        else:
            return "unknown"

    def _generate_llm_analysis(self, message: Dict[str, Any], message_type: str) -> Dict[str, Any]:
        """Generate analysis using local LLM."""
        try:
            prompt = self._build_prompt(message, message_type)
            
            response = self._llm(
                prompt,
                max_tokens=300,
                temperature=0.7,
                top_p=0.9,
                stop=["</analysis>", "\n\n\n"],
            )
            
            analysis_text = response['choices'][0]['text'].strip()
            
            return {
                "summary": analysis_text,
                "recommendations": self._extract_recommendations(analysis_text),
                "risk_assessment": self._extract_risk_level(analysis_text),
                "method": "local_llm"
            }
            
        except Exception as exc:
            logger.warning("[%s] LLM analysis failed: %s. Using fallback.", self.name, exc)
            return self._generate_rule_based_analysis(message, message_type)

    def _build_prompt(self, message: Dict[str, Any], message_type: str) -> str:
        """Build prompt for LLM analysis."""
        if message_type == "strategy":
            action = message.get("action", {})
            energy_alloc = action.get("energy_allocation", 0)
            hash_alloc = action.get("hash_allocation", 0)
            battery_rate = action.get("battery_charge_rate", 0)
            
            prompt = f"""<analysis>
Analyze this energy trading strategy decision:

Energy Allocation: {energy_alloc:.2%}
Hash Allocation: {hash_alloc:.2%}
Battery Charge Rate: {battery_rate:.2f}

Provide a brief analysis of:
1. The allocation balance and risk
2. Battery management strategy
3. Potential improvements
4. Overall risk level (Low/Medium/High)

Analysis:</analysis>"""

        elif message_type == "forecast":
            forecast_data = message.get("forecast", [])
            if forecast_data:
                first_pred = forecast_data[0] if forecast_data else {}
                price = first_pred.get("predicted_price", 0)
                
                prompt = f"""<analysis>
Analyze this energy price forecast:

Next hour predicted price: ${price:.2f}/MWh
Forecast method: {first_pred.get("method", "unknown")}

Provide brief insights on:
1. Price trend direction
2. Market volatility indicators  
3. Trading opportunities
4. Risk factors

Analysis:</analysis>"""
            else:
                prompt = "<analysis>No forecast data available for analysis.</analysis>"

        else:  # data type
            prices = message.get("prices", [])
            if prices:
                latest = prices[-1] if prices else {}
                energy_price = latest.get("energy_price", 0)
                
                prompt = f"""<analysis>
Analyze current market data:

Current Energy Price: ${energy_price:.2f}/MWh
Data Points: {len(prices)} records

Provide brief analysis of:
1. Current market conditions
2. Price stability
3. Trading signals
4. Market sentiment

Analysis:</analysis>"""
            else:
                prompt = "<analysis>No market data available for analysis.</analysis>"

        return prompt

    def _extract_recommendations(self, analysis_text: str) -> list[str]:
        """Extract actionable recommendations from analysis."""
        recommendations = []
        
        # Simple keyword-based extraction
        if "increase" in analysis_text.lower():
            recommendations.append("Consider increasing allocation")
        if "decrease" in analysis_text.lower():
            recommendations.append("Consider reducing exposure")
        if "battery" in analysis_text.lower() and "charge" in analysis_text.lower():
            recommendations.append("Optimize battery charging strategy")
        if "risk" in analysis_text.lower() and "high" in analysis_text.lower():
            recommendations.append("Implement risk mitigation measures")
        
        return recommendations if recommendations else ["Monitor market conditions"]

    def _extract_risk_level(self, analysis_text: str) -> str:
        """Extract risk level from analysis."""
        text_lower = analysis_text.lower()
        if "high" in text_lower and "risk" in text_lower:
            return "High"
        elif "low" in text_lower and "risk" in text_lower:
            return "Low"
        else:
            return "Medium"

    def _generate_rule_based_analysis(self, message: Dict[str, Any], message_type: str) -> Dict[str, Any]:
        """Generate analysis using rule-based logic."""
        try:
            if message_type == "strategy":
                return self._analyze_strategy_rules(message)
            elif message_type == "forecast":
                return self._analyze_forecast_rules(message)
            elif message_type == "data":
                return self._analyze_data_rules(message)
            else:
                return {
                    "summary": "Unknown message type received",
                    "recommendations": ["Review message format"],
                    "risk_assessment": "Medium",
                    "method": "rule_based"
                }
                
        except Exception as exc:
            logger.exception("[%s] Rule-based analysis failed: %s", self.name, exc)
            return {
                "summary": "Analysis failed due to error",
                "recommendations": ["Check system logs"],
                "risk_assessment": "High",
                "method": "error_fallback"
            }

    def _analyze_strategy_rules(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based strategy analysis."""
        action = message.get("action", {})
        energy_alloc = action.get("energy_allocation", 0.5)
        hash_alloc = action.get("hash_allocation", 0.5)
        battery_rate = action.get("battery_charge_rate", 0.0)
        
        # Analyze allocation balance
        imbalance = abs(energy_alloc - hash_alloc)
        if imbalance > 0.3:
            risk = "High"
            summary = f"Highly concentrated allocation detected (Energy: {energy_alloc:.1%}, Hash: {hash_alloc:.1%})"
            recommendations = ["Consider more balanced portfolio allocation"]
        elif imbalance > 0.1:
            risk = "Medium" 
            summary = f"Moderate allocation imbalance (Energy: {energy_alloc:.1%}, Hash: {hash_alloc:.1%})"
            recommendations = ["Monitor performance and consider rebalancing"]
        else:
            risk = "Low"
            summary = f"Well-balanced allocation strategy (Energy: {energy_alloc:.1%}, Hash: {hash_alloc:.1%})"
            recommendations = ["Maintain current allocation strategy"]
        
        # Battery analysis
        if abs(battery_rate) > 0.8:
            risk = "High"
            recommendations.append("Aggressive battery usage detected - monitor degradation")
        elif abs(battery_rate) > 0.4:
            recommendations.append("Moderate battery usage - within normal parameters")
        
        return {
            "summary": summary,
            "recommendations": recommendations,
            "risk_assessment": risk,
            "method": "rule_based"
        }

    def _analyze_forecast_rules(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based forecast analysis."""
        forecast_data = message.get("forecast", [])
        
        if not forecast_data:
            return {
                "summary": "No forecast data available",
                "recommendations": ["Check forecasting system"],
                "risk_assessment": "High",
                "method": "rule_based"
            }
        
        # Analyze price trend
        prices = [f.get("predicted_price", 0) for f in forecast_data[:6]]  # Next 6 hours
        if len(prices) > 1:
            trend = "increasing" if prices[1] > prices[0] else "decreasing"
            volatility = np.std(prices) if len(prices) > 2 else 0
            
            if volatility > 1.0:
                risk = "High"
                summary = f"High volatility forecast detected (σ={volatility:.2f}), prices {trend}"
                recommendations = ["Consider conservative strategy", "Increase monitoring frequency"]
            elif volatility > 0.5:
                risk = "Medium"
                summary = f"Moderate volatility forecast (σ={volatility:.2f}), prices {trend}" 
                recommendations = ["Normal operations with increased awareness"]
            else:
                risk = "Low"
                summary = f"Stable price forecast (σ={volatility:.2f}), prices {trend}"
                recommendations = ["Favorable conditions for standard operations"]
        else:
            risk = "Medium"
            summary = "Limited forecast data available"
            recommendations = ["Improve forecast data collection"]
        
        return {
            "summary": summary,
            "recommendations": recommendations,
            "risk_assessment": risk,
            "method": "rule_based"
        }

    def _analyze_data_rules(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based data analysis."""
        prices = message.get("prices", [])
        
        if not prices:
            return {
                "summary": "No price data received",
                "recommendations": ["Check data pipeline"],
                "risk_assessment": "High",
                "method": "rule_based"
            }
        
        # Analyze recent price movement
        if len(prices) >= 3:
            recent_prices = [p.get("energy_price", 0) for p in prices[-3:]]
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
            
            if abs(price_change) > 0.1:  # >10% change
                risk = "High"
                direction = "spike" if price_change > 0 else "drop"
                summary = f"Significant price {direction} detected ({price_change:.1%} change)"
                recommendations = ["Review position sizes", "Consider risk management measures"]
            elif abs(price_change) > 0.05:  # >5% change
                risk = "Medium"
                summary = f"Moderate price movement detected ({price_change:.1%} change)"
                recommendations = ["Monitor closely for continued trend"]
            else:
                risk = "Low"
                summary = f"Stable price conditions ({price_change:.1%} change)"
                recommendations = ["Normal operations recommended"]
        else:
            risk = "Medium"
            summary = f"Limited price history ({len(prices)} data points)"
            recommendations = ["Allow more data collection for better analysis"]
        
        return {
            "summary": summary,
            "recommendations": recommendations,
            "risk_assessment": risk,
            "method": "rule_based"
        }

    def _generate_claude_analysis(self, message: Dict[str, Any], message_type: str) -> Dict[str, Any]:
        """Generate deep strategic analysis using Claude API."""
        try:
            prompt = self._build_intelligent_prompt(message, message_type)
            
            response = self._claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis_text = response.content[0].text.strip()
            
            # Parse structured response
            return self._parse_claude_response(analysis_text)
            
        except Exception as exc:
            logger.warning("[%s] Claude analysis failed: %s. Using fallback.", self.name, exc)
            return self._generate_rule_based_analysis(message, message_type)

    def _build_intelligent_prompt(self, message: Dict[str, Any], message_type: str) -> str:
        """Build sophisticated prompt for Claude analysis."""
        if message_type == "strategy":
            action = message.get("action", {})
            energy_alloc = action.get("energy_allocation", 0)
            hash_alloc = action.get("hash_allocation", 0)
            battery_rate = action.get("battery_charge_rate", 0)
            method = action.get("method", "unknown")
            
            prompt = f"""You are an expert energy trading strategist analyzing a resource allocation decision for a GridPilot energy management system.

CURRENT ALLOCATION:
- Energy Trading: {energy_alloc:.1%}
- Hash Mining: {hash_alloc:.1%}  
- Battery Action: {battery_rate:.2f} (positive=charge, negative=discharge)
- Decision Method: {method}

Please provide a comprehensive analysis in this JSON format:
{{
    "summary": "Brief assessment of the allocation strategy",
    "market_interpretation": "What this allocation suggests about current market conditions",
    "risk_assessment": "Low/Medium/High with justification",
    "strategic_recommendations": ["specific actionable recommendations"],
    "optimization_opportunities": ["potential improvements to the strategy"],
    "market_timing": "Assessment of timing and market positioning",
    "confidence_score": 0.0-1.0
}}

Focus on strategic depth, market dynamics, and actionable insights that could improve performance."""

        elif message_type == "forecast":
            forecast_data = message.get("forecast", [])
            if forecast_data:
                predictions = forecast_data[:5]  # First 5 predictions
                method = forecast_data[0].get("method", "unknown") if forecast_data else "unknown"
                
                prompt = f"""You are an expert energy market analyst reviewing price forecasts for strategic decision-making.

FORECAST DATA:
{json.dumps(predictions, indent=2)}
Forecast Method: {method}

Provide strategic analysis in this JSON format:
{{
    "summary": "Key insights from the forecast trends",
    "price_trajectory": "Expected price movement and patterns",
    "volatility_assessment": "Market stability and risk factors",
    "trading_signals": ["specific buy/sell/hold recommendations"],
    "arbitrage_opportunities": ["cross-market opportunities identified"],
    "risk_factors": ["potential threats to forecast accuracy"],
    "strategic_timing": "Optimal timing for major decisions",
    "confidence_score": 0.0-1.0
}}

Focus on actionable trading insights and strategic implications."""

        else:  # data type
            prices = message.get("prices", [])
            features = message.get("features", [])
            inventory = message.get("inventory", {})
            
            prompt = f"""You are an expert quantitative analyst examining real-time market data for energy trading insights.

MARKET DATA SUMMARY:
- Price Records: {len(prices)} data points
- Feature Engineering: {len(features)} engineered features
- Current Utilization: {inventory.get('utilization_rate', 'unknown')}%
- Battery SOC: {inventory.get('battery_soc', 'unknown')}

RECENT PRICES: {json.dumps(prices[-3:], indent=2) if prices else "No data"}

Provide deep market analysis in this JSON format:
{{
    "summary": "Current market state and key patterns",
    "price_momentum": "Short and medium-term price trends",
    "market_regime": "Bull/Bear/Sideways with characteristics",
    "volatility_analysis": "Current volatility state and implications", 
    "feature_insights": "Key patterns from engineered features",
    "system_optimization": "Recommendations for current system state",
    "market_opportunities": ["immediate opportunities identified"],
    "risk_alerts": ["current market risks to monitor"],
    "confidence_score": 0.0-1.0
}}

Focus on pattern recognition, regime identification, and optimal positioning."""

        return prompt

    def _parse_claude_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Claude's JSON response with fallback."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                # Ensure required fields
                return {
                    "summary": parsed.get("summary", "Analysis completed"),
                    "recommendations": parsed.get("strategic_recommendations", parsed.get("trading_signals", ["Continue monitoring market conditions"])),
                    "risk_assessment": parsed.get("risk_assessment", "Medium").split()[0],  # Extract just Low/Medium/High
                    "method": "claude_ai",
                    "confidence": parsed.get("confidence_score", 0.7),
                    "detailed_analysis": parsed  # Include full analysis
                }
        except:
            pass
        
        # Fallback parsing
        return {
            "summary": response_text[:200] + "..." if len(response_text) > 200 else response_text,
            "recommendations": ["Monitor market conditions", "Review allocation strategy"],
            "risk_assessment": "Medium",
            "method": "claude_ai_fallback",
            "confidence": 0.5,
            "raw_response": response_text
        }


if __name__ == "__main__":
    LocalLLMAgent().start() 