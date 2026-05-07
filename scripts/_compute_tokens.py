"""One-shot helper: compute DocETL input-token estimates from char counts."""
import json

papers = json.load(open("data/processed/rev/papers.json", encoding="utf-8"))
n = len(papers)

RTR_STATIC = 3200   # chars: prompt boilerplate around candidate_passages
FDR_STATIC = 3800   # chars: prompt boilerplate around full_text

total_candidate = sum(len(p.get("candidate_passages") or "") for p in papers)
total_fulltext  = sum(len(p.get("full_text") or "") for p in papers)
meta_chars = sum(
    len((p.get("paper_id") or "") + (p.get("paper_doi") or "") +
        (p.get("pmcid") or "") + (p.get("pmid") or "") +
        (p.get("title") or ""))
    for p in papers
)

rtr_chars = total_candidate + meta_chars + RTR_STATIC * n
fdr_chars = total_fulltext  + meta_chars + FDR_STATIC * n
rtr_in  = rtr_chars / 4
fdr_in  = fdr_chars / 4
rtr_out = rtr_in  * 0.10
fdr_out = fdr_in  * 0.10

PRICE_IN  = 0.15 / 1e6   # USD per token  (gpt-4o-mini)
PRICE_OUT = 0.60 / 1e6

rtr_cost = rtr_in * PRICE_IN + rtr_out * PRICE_OUT
fdr_cost = fdr_in * PRICE_IN + fdr_out * PRICE_OUT

print(f"n papers                  : {n}")
print()
print(f"candidate_passages chars  : {total_candidate:>15,}")
print(f"full_text chars           : {total_fulltext:>15,}")
print(f"metadata chars (all)      : {meta_chars:>15,}")
print()
print(f"RTR total input chars     : {rtr_chars:>15,}  =>  {rtr_in/1e6:.3f}M tokens")
print(f"FDR total input chars     : {fdr_chars:>15,}  =>  {fdr_in/1e6:.3f}M tokens")
print()
print(f"RTR output tokens (10%)   : {rtr_out/1e3:>12,.0f}K  =>  {rtr_out/1e6:.3f}M")
print(f"FDR output tokens (10%)   : {fdr_out/1e3:>12,.0f}K  =>  {fdr_out/1e6:.3f}M")
print()
print(f"RTR cost (in+out, USD)    : {rtr_cost:.2f}")
print(f"FDR cost (in+out, USD)    : {fdr_cost:.2f}")
