import difflib

def extract_rulebook(llm_history: list) -> list:
    """
    Extracts compiled rules from blocks where KPI improved vs previous block.
    Deduplicates near-identical rules using sequence matching.
    Returns max 10 rules.
    """
    rules = []
    
    for record in llm_history:
        rule = record.get("compiled_rule", "").strip()
        if not rule or rule.lower() in ["none", ""]:
            continue
            
        actual_cost = record.get("actual_total_cost", float('inf'))
        prev_cost = record.get("prev_cost", float('-inf')) # Note: for improvement, actual < prev
        
        actual_sl = record.get("actual_service_level", 0.0)
        prev_sl = record.get("prev_sl", 1.0) # Note: for improvement, actual > prev
        
        # Check if KPI improved vs previous block
        improved = False
        # Cost decrease or service level increase
        if actual_cost < prev_cost or actual_sl > prev_sl:
            improved = True
            
        if improved:
            # Deduplicate near-identical rules
            duplicate = False
            for existing in rules:
                ratio = difflib.SequenceMatcher(None, rule.lower(), existing.lower()).ratio()
                if ratio > 0.8:  # simple string similarity threshold
                    duplicate = True
                    break
                    
            if not duplicate:
                rules.append(rule)
                
    return rules[:10]

def format_rulebook_for_prompt(rules: list) -> str:
    """
    Formats rules as a numbered list for injection into the LLM prompt.
    """
    if not rules:
        return ""
        
    formatted = "COMPILED RULES FROM WARM-UP:\n"
    for i, rule in enumerate(rules, 1):
        formatted += f"{i}. {rule}\n"
    return formatted
