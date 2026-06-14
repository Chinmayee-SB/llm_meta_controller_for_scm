import json
import re

def parse_llm_response(response_text: str, prev_alphas: dict) -> tuple:
    """
    Parses the LLM response.
    Returns: (alphas_dict, reasoning, self_correction, compiled_rule, confidence)
    """
    if response_text is None:
        return (prev_alphas, "PARSE_FAILURE", "", "", 0.0)

    # Strip markdown code fences if present (```json ... ```)
    clean_text = response_text.strip()
    if clean_text.startswith("```"):
        match = re.search(r"```(?:json)?(.*?)```", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1).strip()
            
    try:
        data = json.loads(clean_text)
        
        # Extract fields
        reasoning = data.get("reasoning", "")
        self_correction = data.get("self_correction", "")
        compiled_rule = data.get("compiled_rule", "")
        confidence = data.get("confidence", 0.0)
        
        # Check for missing required field: compiled_rule
        if "compiled_rule" not in data:
            raise ValueError("Missing compiled_rule")
            
        # Extract alphas (supporting flat or nested structure based on user prompts)
        alphas_dict = {}
        if "alphas" in data and isinstance(data["alphas"], dict):
            # Nested structure
            alphas_data = data["alphas"]
            alphas_dict[0] = alphas_data.get("stage_1_retailer", prev_alphas.get(0, 0.5))
            alphas_dict[1] = alphas_data.get("stage_2_warehouse", prev_alphas.get(1, 0.5))
            alphas_dict[2] = alphas_data.get("stage_3_distributor", prev_alphas.get(2, 0.5))
            alphas_dict[3] = alphas_data.get("stage_4_manufacturer", prev_alphas.get(3, 0.5))
        else:
            # Flat structure
            alphas_dict[0] = data.get("alpha_stage_1_retailer", prev_alphas.get(0, 0.5))
            alphas_dict[1] = data.get("alpha_stage_2_warehouse", prev_alphas.get(1, 0.5))
            alphas_dict[2] = data.get("alpha_stage_3_distributor", prev_alphas.get(2, 0.5))
            alphas_dict[3] = data.get("alpha_stage_4_manufacturer", prev_alphas.get(3, 0.5))
            
        # Validate and clip: all alpha values must be floats in [0.0, 1.0]
        for k in alphas_dict:
            try:
                val = float(alphas_dict[k])
                alphas_dict[k] = max(0.0, min(1.0, val))
            except (ValueError, TypeError):
                # If non-float is given, fail the parse
                raise ValueError(f"Invalid alpha value for key {k}")

        return (alphas_dict, reasoning, self_correction, compiled_rule, confidence)
        
    except Exception as e:
        # On any failure: return (prev_alphas, "PARSE_FAILURE", "", "", 0.0)
        return (prev_alphas, "PARSE_FAILURE", "", "", 0.0)
