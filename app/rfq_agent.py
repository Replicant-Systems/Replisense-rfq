from autogen import ConversableAgent
from dotenv import load_dotenv
import os
import json

load_dotenv()

LLM_CONFIG = {
    "model": "llama3-70b-8192",
    "api_key": os.getenv("GROQ_API_KEY"),
    "base_url": "https://api.groq.com/openai/v1",
    "api_type": "openai",
    "temperature": 0.1,
    "max_tokens": 800,
    "cache_seed": 42,
}

class RFQFieldGenerator:
    def __init__(self):
        if not LLM_CONFIG.get("api_key"):
            raise ValueError("GROQ_API_KEY not set in environment.")
        
        self.agent = ConversableAgent(
            name="field_generator",
            system_message="You are an expert at extracting structured RFQ data from raw text. Return valid JSON only.",
            llm_config=LLM_CONFIG,
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        
    def generate(self, raw_text: str, source_file: str = "email-body") -> dict:
        prompt = f"""Extract the following fields from the RFQ text below:
- title
- client_name
- client_email
- client_contact
- client_phone
- rfq_to
- delivery_location
- delivery_deadline
- response_due_date
- description
- line_items (list with part_number, description, quantity, target_price)
- requested_documents
- confidence_score (0.0 to 1.0)
- missing_fields (list)
- requires_review (true/false)

Only return a valid JSON dictionary.

Text:
\"\"\"
{raw_text}
\"\"\"
"""
        try:
            print("📤 Sending prompt to LLM...")
            reply = self.agent.generate_reply([{"role": "user", "content": prompt}])
            print("📥 LLM response:", reply)

            # Parse JSON if reply is a string
            if isinstance(reply, str):
                try:
                    reply = json.loads(reply)
                except json.JSONDecodeError as e:
                    print("❌ Failed to parse JSON from LLM string:", str(e))
                    return {"success": False, "error": "LLM returned invalid JSON string."}

            if not isinstance(reply, dict):
                print("❌ LLM did not return a dictionary.")
                return {"success": False, "error": "LLM did not return dict."}

            # Add metadata
            reply["source_file"] = source_file
            reply["success"] = True
            reply["message"] = f"RFQ processed from {source_file}"
            
            print("📄 Parsed fields:", reply)
            return reply

        except Exception as e:
            print("❌ Exception during generation", str(e))
            return {"success": False, "error": str(e)}