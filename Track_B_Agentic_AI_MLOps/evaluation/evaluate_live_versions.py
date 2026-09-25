from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from app.agent import run_agent
from app.agent_llm import AgentLLM

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

CASES=json.loads((ROOT/"evaluation"/"golden_set.json").read_text())

def main():
    try:
        import mlflow
    except ImportError as e: raise SystemExit("Run uv sync first; MLflow is required for the live comparison.") from e
    mlflow.set_tracking_uri((ROOT/"mlruns").as_uri()); mlflow.set_experiment("W17_Track_B_Live_Prompt_Versions")
    all_runs=[]
    for v in ["v1","v2","v3"]:
        llm=AgentLLM(prompt_version=v); rows=[]; completed=0; tool_ok=0; lengths=[]; tokens=0
        with mlflow.start_run(run_name=f"live_prompt_{v}"):
            mlflow.log_params({"prompt_version":v,"model":llm.model,"temperature":0.2,"top_k":4,"agent_max_iterations":6,"chunk_chars":700})
            for c in CASES:
                ans=run_agent(c["query"],EvalRagStore(),llm=llm,max_steps=6)
                actions=[s.action for s in ans.trajectory]; completed += ans.status=="complete"; tool_ok += bool(actions) and actions[0]==c["expected_tool"]; lengths.append(ans.iterations); tokens += ans.total_tokens
                context="\n".join(str(s.detail) for s in ans.trajectory if s.action=="search")
                rows.append({"id":c["id"],"query":c["query"],"response":ans.answer,"context":context,"status":ans.status,"iterations":ans.iterations,"tokens":ans.total_tokens,"trace":[s.model_dump() for s in ans.trajectory]})
            m={"task_completion_rate":completed/len(CASES),"tool_call_correctness":tool_ok/len(CASES),"average_trajectory_length":sum(lengths)/len(lengths),"token_usage":tokens}
            mlflow.log_metrics(m); out=ROOT/"artifacts"/f"live_{v}_responses.json"; out.write_text(json.dumps(rows,indent=2)); mlflow.log_artifact(str(out),artifact_path="responses")
            all_runs.append({"prompt_version":v,**m})
    (ROOT/"artifacts"/"live_current_responses.json").write_text((ROOT/"artifacts"/"live_v3_responses.json").read_text())
    (ROOT/"artifacts"/"live_prompt_metrics.json").write_text(json.dumps(all_runs,indent=2)); print(json.dumps(all_runs,indent=2))
if __name__=="__main__":main()
