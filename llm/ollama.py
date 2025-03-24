import requests
import json
from typing import List, Dict, Any

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

class OllamaLLM:
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
    
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate a response using Ollama with RAG context"""
        
        # Format context for the prompt
        formatted_context = self._format_context(context)
        
        # Construct prompt with context
        prompt = f"""
You are a helpful assistant specialized in bug reports analysis.
Here are some similar bug reports from the past:

{formatted_context}

Based on these similar reports, please analyze the following new bug report:
{query}

Please provide:
1. Identification of similar patterns or issues
2. Possible solutions based on past resolutions
3. Any additional context that might be helpful
"""
        
        # Call Ollama API
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                })
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                return f"Error: Unable to get response from Ollama (Status code: {response.status_code})"
                
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def _format_context(self, context_results: List[Dict[str, Any]]) -> str:
        """Format the retrieved context for the prompt"""
        if not context_results:
            return "No similar bug reports found."
        
        formatted_text = ""
        
        for i, item in enumerate(context_results.get("documents", [[]])[0]):
            metadata = context_results.get("metadatas", [[]])[0][i]
            formatted_text += f"--- Bug Report {i+1} ---\n"
            formatted_text += f"{item}\n\n"
        
        return formatted_text 
    
    def generate_response_advanced(self, query: str, context_results: Dict[str, Any]) -> str:
        """Generate response with advanced context handling"""
        prompt = f"""You are analyzing bug reports from a software development team. Your task is to find patterns, similarities, and potential solutions based on historical data.

        NEW BUG REPORT:
        {query}

        SIMILAR HISTORICAL REPORTS:
        """
            
        # Structure the retrieved context more effectively
        for i, item in enumerate(context_results.get("documents", [[]])[0]):
            metadata = context_results.get("metadatas", [[]])[0][i]
            similarity = 1 - context_results.get("distances", [[]])[0][i]
            
            prompt += f"\n--- Report {i+1} (Similarity: {similarity:.2f}) ---\n"
            prompt += f"Date: {metadata.get('date', 'Unknown')}\n"
            prompt += f"Reporter: {metadata.get('user', 'Unknown')}\n"
            prompt += f"Description: {item}\n"
            
            prompt += """
        Based on these similar historical issues, please provide:

        1. PATTERN ANALYSIS: Common elements, behaviors, or conditions across these issues
        2. ROOT CAUSE ASSESSMENT: Likely underlying causes based on the patterns
        3. RECOMMENDED SOLUTIONS: Approaches that worked in similar cases or new recommendations
        4. PRIORITY ASSESSMENT: How urgent this issue appears based on historical context
        5. RELATED COMPONENTS: What parts of the system are likely affected

        Format your response clearly with these sections.
        """

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Lower temperature for more factual responses
                        "top_p": 0.9
                    }
                })
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                return f"Error: Unable to get response (Status code: {response.status_code})"
                
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"