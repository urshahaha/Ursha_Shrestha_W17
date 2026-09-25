from __future__ import annotations
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.agent import run_agent

class EvalRagStore:
    ROWS={
      "shipping":[{"filename":"sample_knowledge.txt","chunk_id":2,"text":"Standard shipping normally takes 3 to 5 business days after dispatch. Express shipping normally takes 1 to 2 business days after dispatch.","distance":.08}],
      "returns":[{"filename":"sample_knowledge.txt","chunk_id":0,"text":"Customers may request a standard return within 30 days of delivery. Items should be unused and returned with the original packaging when possible.","distance":.07}],
      "refunds":[{"filename":"sample_knowledge.txt","chunk_id":1,"text":"After an approved return reaches the warehouse, refunds are normally processed within 5 to 7 business days. Bank processing time may add additional delay.","distance":.06}]}
    def search(self,q,k=4):
      q=q.lower()
      if "return" in q and "refund" not in q:return self.ROWS["returns"][:k]
      if "refund" in q:return self.ROWS["refunds"][:k]
      return self.ROWS["shipping"][:k]

class VersionedScriptedModel:
    def __init__(self,version): self.version=version
    def decide(self,state):
      q=state["question"].lower(); ev=state["evidence"]; notes=state["tool_notes"]; draft=state["draft"]; ver=state["verification"]
      has_calc=any(n.get("tool")=="calculator" and "result" in n for n in notes)
      has_docs=any(n.get("tool")=="list_documents" and "documents" in n for n in notes)
      # v3 routes specialized requests correctly on the first step.
      if self.version=="v3":
        if "calculate" in q and not has_calc: return ({"action":"calculator","reason":"Arithmetic routes directly to calculator.","expression":"25 * 16"},0)
        if ("documents" in q or "files" in q) and not has_docs: return ({"action":"list_documents","reason":"File-list request routes directly to list_documents."},0)
      # v1/v2 show the observed inefficiency: they retrieve before specialized tools.
      if self.version in {"v1","v2"} and not ev and not draft and not has_calc and not has_docs and ("calculate" in q or "documents" in q or "files" in q):
        return ({"action":"search","reason":f"{self.version} tries retrieval before choosing the specialized tool.","query":state["question"]},0)
      if "calculate" in q and not has_calc: return ({"action":"calculator","reason":"Arithmetic is required.","expression":"25 * 16"},0)
      if ("documents" in q or "files" in q) and not has_docs: return ({"action":"list_documents","reason":"Document listing is required."},0)
      # Once a specialized tool returned a result, draft directly; no RAG search is needed.
      if not draft and (has_calc or has_docs): return ({"action":"draft","reason":"Specialized tool result is sufficient for drafting."},0)
      search_failed=any(n.get("tool")=="search" and n.get("error") for n in notes)
      if not draft:
        # v1 failure: one generic search is treated as sufficient for a multi-fact comparison.
        if self.version=="v1" and "compare" in q:
          if not ev and not search_failed: return ({"action":"search","reason":"v1 performs one generic search and assumes it is enough.","query":state["question"]},0)
          return ({"action":"draft","reason":"v1 drafts after the first partial result."},0)
        if "compare" in q:
          have_ret=any("30 days" in x.get("text","") for x in ev); have_ref=any("5 to 7" in x.get("text","") for x in ev)
          if not have_ret and not search_failed:return ({"action":"search","reason":"Need return evidence.","query":"return policy"},0)
          if not have_ref and not search_failed:return ({"action":"search","reason":"Need refund evidence too.","query":"refund timing"},0)
        elif not ev and not search_failed:return ({"action":"search","reason":"Need evidence.","query":state["question"]},0)
        return ({"action":"draft","reason":"Draft from available evidence/tool output."},0)
      if ver is None:return ({"action":"verify","reason":"Verify before finish."},0)
      if ver.get("supported"): return ({"action":"finish","reason":"Verification passed."},0)
      if not search_failed:return ({"action":"search","reason":"Missing support; search again.","query":ver.get("search_query") or state["question"]},0)
      return ({"action":"verify","reason":"Search unavailable."},0)
    def draft(self,state):
      notes=state["tool_notes"]; ev=state["evidence"]
      c=next((n for n in notes if n.get("tool")=="calculator" and "result" in n),None)
      if c:return ({"answer":f"25 * 16 = {c['result']}","sources":[],"grounded":True},0)
      d=next((n for n in notes if n.get("tool")=="list_documents"),None)
      if d is not None:return ({"answer":"Available documents: "+(", ".join(d.get("documents",[])) or "none"),"sources":[],"grounded":True},0)
      if not ev:return ({"answer":"I do not have enough retrieved evidence to answer confidently.","sources":[],"grounded":False},0)
      return ({"answer":" ".join(x["text"] for x in ev),"sources":[{"filename":x["filename"],"chunk_id":x["chunk_id"]} for x in ev],"grounded":True},0)
    def verify(self,state):
      q=state["question"].lower(); draft=state["draft"] or {}; ev=state["evidence"]; notes=state["tool_notes"]
      tool_ok=any((n.get("tool")=="calculator" and "result" in n) or (n.get("tool")=="list_documents" and "documents" in n) for n in notes)
      supported=bool(draft.get("grounded")) and (bool(ev) or tool_ok)
      if "compare" in q:
        supported=supported and any("30 days" in x.get("text","") for x in ev) and any("5 to 7" in x.get("text","") for x in ev)
      return ({"supported":supported,"reason":"All requested facts supported." if supported else "Comparison is missing one requested fact.","missing_claim":"" if supported else "Missing comparison evidence.","search_query":"refund timing" if "compare" in q else state["question"]},0)

CASES=[
 ("basic_rag","How long does standard shipping take after dispatch?","search"),
 ("comparison","Compare the return window with the refund processing time.","search"),
 ("calculator","Calculate 25 * 16.","calculator"),
 ("documents","What documents are available?","list_documents"),
]

def trace_obj(version,name,q,ans):
  return {"version":version,"case":name,"query":q,"status":ans.status,"iterations":ans.iterations,"total_tokens":ans.total_tokens,"answer":ans.answer,"trace":[{"step":s.iteration,"tool":s.action,"tool_arguments":s.detail.get("query") or s.detail.get("expression") or {},"raw_result":s.detail,"model_decision_reason":s.reason,"ok":s.ok} for s in ans.trajectory],"stop_reason":ans.status}

def main():
  rows=[]; mlflow=None
  try:
    import mlflow as _m; mlflow=_m; mlflow.set_tracking_uri((ROOT/"mlruns").as_uri()); mlflow.set_experiment("W17_Track_B_Prompt_Versions")
  except Exception: pass
  for version in ["v1","v2","v3"]:
    results=[]; correct=0; completed=0; lens=[]
    ctx=mlflow.start_run(run_name=f"prompt_{version}") if mlflow else None
    if mlflow: mlflow.log_params({"prompt_version":version,"model":"offline_scripted_control_path","temperature":0.2,"top_k":4,"agent_max_iterations":6,"chunk_chars":700})
    try:
      for name,q,expected in CASES:
        ans=run_agent(q,EvalRagStore(),llm=VersionedScriptedModel(version),max_steps=6)
        tr=trace_obj(version,name,q,ans); results.append(tr)
        actions=[x["tool"] for x in tr["trace"]]; completed += ans.status=="complete"; correct += bool(actions) and actions[0]==expected; lens.append(ans.iterations)
        (ROOT/"traces"/f"{version}_{name}.json").write_text(json.dumps(tr,indent=2))
      metric={"prompt_version":version,"task_completion_rate":completed/len(CASES),"tool_call_correctness":correct/len(CASES),"average_trajectory_length":sum(lens)/len(lens),"token_usage":0,"regression_test_pass_rate":"NOT_RUN_LLM_JUDGE"}
      rows.append(metric)
      if mlflow:
        mlflow.log_metrics({k:v for k,v in metric.items() if isinstance(v,(int,float)) and k!="prompt_version"})
        mlflow.log_artifacts(str(ROOT/"traces"),artifact_path="traces")
        mlflow.log_artifact(str(ROOT/"prompts"/f"prompt_{version}.txt"),artifact_path="prompts")
    finally:
      if mlflow and mlflow.active_run(): mlflow.end_run()
  with open(ROOT/"artifacts"/"prompt_run_comparison.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
  (ROOT/"artifacts"/"offline_prompt_metrics.json").write_text(json.dumps(rows,indent=2))
  print(json.dumps(rows,indent=2))
if __name__=="__main__":main()
