import os
import json
import time
from datetime import datetime
from openai import OpenAI, RateLimitError, APIError

class LLMClient:
    def __init__(self, model="gpt-4o-mini", max_retries=3, retry_delay=2.0):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Load API key from OPENAI_API_KEY environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def query(self, prompt: str, system_prompt: str, max_tokens=600, temperature=0.3, block_id=-1) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=max_tokens
                )
                
                raw_text = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                response_tokens = response.usage.completion_tokens if response.usage else 0
                
                # Log the call
                self._log_call(block_id, prompt_tokens, response_tokens, raw_text)
                
                return raw_text
                
            except (RateLimitError, APIError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    print(f"LLM API Error after {self.max_retries} retries: {e}")
                    return None
            except Exception as e:
                print(f"Unexpected LLM Error: {e}")
                return None
        return None

    def _log_call(self, block_id, prompt_tokens, response_tokens, raw_response):
        log_dir = "MADRL/results/phase2/llm_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "block_id": block_id,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "raw_response": raw_response
        }
        
        file_path = os.path.join(log_dir, f"block_{block_id}.json")
        try:
            with open(file_path, "w") as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to log LLM call: {e}")
