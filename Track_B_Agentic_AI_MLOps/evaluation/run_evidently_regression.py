from __future__ import annotations
import json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    key=os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key: raise SystemExit("No LLM API key found. Set OPENAI_API_KEY (or adapt judge provider) and rerun.")
    # Import only at runtime so the rest of the project remains inspectable without Evidently installed.
    try:
        import evidently
    except ImportError as e: raise SystemExit("Evidently is not installed. Run: uv sync") from e
    # The golden/current response collection is intentionally separated from the judge so no judge scores are fabricated.
    golden=json.loads((ROOT/"evaluation"/"golden_set.json").read_text())
    current_path=ROOT/"artifacts"/"live_current_responses.json"
    if not current_path.exists():
        raise SystemExit("First run the live agent evaluation to create artifacts/live_current_responses.json.")
    current=json.loads(current_path.read_text())
    # Evidently's LLM API changes across minor releases. Keep this adapter small and fail loudly if the installed API differs.
    try:
        from evidently import Report
        from evidently.metrics import TextEvals
        from evidently.descriptors import LLMJudge, BinaryClassificationPromptTemplate
    except Exception as e:
        raise SystemExit("Installed Evidently LLM API differs from this pinned adapter. See README and update the three imports only.") from e
    rows=[]
    by_id={x["id"]:x for x in current}
    for g in golden:
        c=by_id[g["id"]]
        rows.append({"id":g["id"],"query":g["query"],"reference":g["golden_answer"],"response":c["response"],"context":c.get("context","")})
    import pandas as pd
    df=pd.DataFrame(rows)
    correctness=LLMJudge(prompt=BinaryClassificationPromptTemplate(criteria="The response preserves the important facts in the reference answer and does not contradict it.",target_category="correct",non_target_category="incorrect",include_reasoning=True),provider="openai",model="gpt-4.1-mini")
    grounded=LLMJudge(prompt=BinaryClassificationPromptTemplate(criteria="The response is fully supported by the supplied retrieval context and does not add unsupported factual claims.",target_category="grounded",non_target_category="ungrounded",include_reasoning=True),provider="openai",model="gpt-4.1-mini")
    report=Report([TextEvals(column_name="response",descriptors=[correctness,grounded])])
    snap=report.run(current_data=df)
    out=ROOT/"reports"/"evidently_llm_regression_report.html"; snap.save_html(str(out))
    # Export raw snapshot so the pass percentage can be inspected/logged rather than invented.
    (ROOT/"artifacts"/"evidently_llm_snapshot.json").write_text(json.dumps(snap.dict(),indent=2,default=str))
    print(out)
if __name__=="__main__":main()
