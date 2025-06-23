from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd

# Try to import ChromaDB for vector storage
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class VectorStoreAgent(BaseAgent):
    """Agent for storing and retrieving embeddings using ChromaDB."""

    subscribe_topics = ["strategy-action", "forecast", "llm-analysis", "feature-vector"]
    publish_topic = "knowledge-retrieval"

    def __init__(self, persist_directory: str = "data/vectorstore"):
        super().__init__(name="VectorStoreAgent")
        self._persist_directory = persist_directory
        self._use_vectorstore = HAS_CHROMA
        self._client = None
        self._collection = None
        
        if self._use_vectorstore:
            self._initialize_vectorstore()
        else:
            logger.warning("[%s] ChromaDB not available, using simple in-memory storage", self.name)
            self._memory_store = []

    def _initialize_vectorstore(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure directory exists
            os.makedirs(self._persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self._client = chromadb.PersistentClient(
                path=self._persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name="gridpilot_knowledge",
                metadata={"description": "GridPilot-GT knowledge base"}
            )
            
            logger.info("[%s] ChromaDB initialized at %s", self.name, self._persist_directory)
            
        except Exception as exc:
            logger.warning("[%s] Failed to initialize ChromaDB: %s. Using fallback.", self.name, exc)
            self._use_vectorstore = False
            self._memory_store = []

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Store incoming messages and provide knowledge retrieval."""
        try:
            # Store the message
            self._store_message(message)
            
            # Check if this is a query for knowledge retrieval
            if message.get("query_type") == "knowledge_search":
                return self._search_knowledge(message.get("query", ""))
            
            # For other messages, potentially provide relevant context
            return self._get_relevant_context(message)
            
        except Exception as exc:
            logger.exception("[%s] Error processing message: %s", self.name, exc)
            return None

    def _store_message(self, message: Dict[str, Any]) -> None:
        """Store message in vector database."""
        try:
            # Create document text from message
            doc_text = self._message_to_text(message)
            doc_id = f"{message.get('source', 'unknown')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
            if self._use_vectorstore:
                # Store in ChromaDB
                self._collection.add(
                    documents=[doc_text],
                    ids=[doc_id],
                    metadatas=[{
                        "timestamp": message.get("timestamp", pd.Timestamp.now().isoformat()),
                        "source": message.get("source", "unknown"),
                        "message_type": self._identify_message_type(message)
                    }]
                )
            else:
                # Store in memory
                self._memory_store.append({
                    "id": doc_id,
                    "text": doc_text,
                    "metadata": {
                        "timestamp": message.get("timestamp", pd.Timestamp.now().isoformat()),
                        "source": message.get("source", "unknown"),
                        "message_type": self._identify_message_type(message)
                    }
                })
                
                # Keep only last 1000 items in memory
                if len(self._memory_store) > 1000:
                    self._memory_store = self._memory_store[-1000:]
            
            logger.debug("[%s] Stored message: %s", self.name, doc_id)
            
        except Exception as exc:
            logger.warning("[%s] Failed to store message: %s", self.name, exc)

    def _message_to_text(self, message: Dict[str, Any]) -> str:
        """Convert message to searchable text."""
        text_parts = []
        
        # Add source and timestamp
        text_parts.append(f"Source: {message.get('source', 'unknown')}")
        text_parts.append(f"Timestamp: {message.get('timestamp', 'unknown')}")
        
        # Add content based on message type
        if "action" in message:
            action = message["action"]
            text_parts.append(f"Strategy Action:")
            text_parts.append(f"Energy allocation: {action.get('energy_allocation', 0):.2%}")
            text_parts.append(f"Hash allocation: {action.get('hash_allocation', 0):.2%}")
            text_parts.append(f"Battery rate: {action.get('battery_charge_rate', 0):.2f}")
            text_parts.append(f"Method: {action.get('method', 'unknown')}")
            
        elif "forecast" in message:
            forecast = message["forecast"]
            if isinstance(forecast, list) and forecast:
                first_pred = forecast[0]
                text_parts.append(f"Price Forecast:")
                text_parts.append(f"Predicted price: ${first_pred.get('predicted_price', 0):.2f}")
                text_parts.append(f"Method: {first_pred.get('method', 'unknown')}")
                
        elif "analysis" in message:
            analysis = message["analysis"]
            text_parts.append(f"Analysis:")
            text_parts.append(f"Summary: {analysis.get('summary', '')}")
            text_parts.append(f"Risk: {analysis.get('risk_assessment', 'unknown')}")
            if analysis.get('recommendations'):
                text_parts.append(f"Recommendations: {'; '.join(analysis['recommendations'])}")
                
        elif "features" in message:
            text_parts.append(f"Market Data:")
            prices = message.get("prices", [])
            if prices:
                latest = prices[-1]
                text_parts.append(f"Energy price: ${latest.get('energy_price', 0):.2f}")
                text_parts.append(f"Hash price: ${latest.get('hash_price', 0):.2f}")
            inventory = message.get("inventory", {})
            if inventory:
                text_parts.append(f"Inventory utilization: {inventory.get('utilization_rate', 0):.1f}%")
        
        return " | ".join(text_parts)

    def _identify_message_type(self, message: Dict[str, Any]) -> str:
        """Identify message type for categorization."""
        if "action" in message:
            return "strategy"
        elif "forecast" in message:
            return "forecast"
        elif "analysis" in message:
            return "analysis"
        elif "features" in message:
            return "data"
        else:
            return "unknown"

    def _search_knowledge(self, query: str) -> Dict[str, Any]:
        """Search for relevant knowledge based on query."""
        try:
            if self._use_vectorstore:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=5,
                    include=["documents", "metadatas", "distances"]
                )
                
                relevant_docs = []
                if results["documents"]:
                    for i, doc in enumerate(results["documents"][0]):
                        relevant_docs.append({
                            "content": doc,
                            "metadata": results["metadatas"][0][i],
                            "similarity": 1.0 - results["distances"][0][i]  # Convert distance to similarity
                        })
            else:
                # Simple text search in memory
                relevant_docs = []
                query_lower = query.lower()
                for item in self._memory_store[-100:]:  # Search last 100 items
                    if query_lower in item["text"].lower():
                        relevant_docs.append({
                            "content": item["text"],
                            "metadata": item["metadata"],
                            "similarity": 0.8  # Mock similarity
                        })
                
                # Sort by timestamp (newest first)
                relevant_docs.sort(key=lambda x: x["metadata"]["timestamp"], reverse=True)
                relevant_docs = relevant_docs[:5]
            
            return {
                "timestamp": pd.Timestamp.now().isoformat(),
                "query": query,
                "results": relevant_docs,
                "source": self.name,
            }
            
        except Exception as exc:
            logger.exception("[%s] Knowledge search failed: %s", self.name, exc)
            return {
                "timestamp": pd.Timestamp.now().isoformat(),
                "query": query,
                "results": [],
                "error": str(exc),
                "source": self.name,
            }

    def _get_relevant_context(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Get relevant historical context for current message."""
        try:
            message_type = self._identify_message_type(message)
            
            # Search for similar historical events
            if message_type == "strategy":
                query = "strategy allocation energy hash battery"
            elif message_type == "forecast":
                query = "forecast price prediction"
            elif message_type == "data":
                query = "market data prices"
            else:
                return None
            
            context = self._search_knowledge(query)
            
            # Only return if we found relevant results
            if context.get("results"):
                return {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "context_type": "historical_similarity",
                    "context": context,
                    "source": self.name,
                }
            
            return None
            
        except Exception as exc:
            logger.warning("[%s] Failed to get context: %s", self.name, exc)
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            if self._use_vectorstore:
                count = self._collection.count()
                return {
                    "storage_type": "chromadb",
                    "total_documents": count,
                    "persist_directory": self._persist_directory,
                }
            else:
                return {
                    "storage_type": "memory",
                    "total_documents": len(self._memory_store),
                    "persist_directory": None,
                }
        except Exception as exc:
            logger.warning("[%s] Failed to get stats: %s", self.name, exc)
            return {"error": str(exc)}


if __name__ == "__main__":
    VectorStoreAgent().start() 