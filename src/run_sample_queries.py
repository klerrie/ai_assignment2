"""
Generate sample query outputs.
"""

import json
import os
from query import query_rag

def main():
    """Generate sample queries and save to outputs/sample_queries.json."""
    questions = [
        "How do employees request time off?",
        "What documents are required during the onboarding process?",
        "How does the payroll processing work?"
    ]
    
    results = []
    for q in questions:
        print(f"Processing: {q}")
        results.append(query_rag(q))
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sample_queries.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(results)} queries to outputs/sample_queries.json")

if __name__ == "__main__":
    main()

