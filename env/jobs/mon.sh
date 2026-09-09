#!/bin/bash
# compact status of eval + training
cd /weka/scratch/jhu/cxiao13/zwang544/SRFT/agentdojo
date '+%H:%M'; squeue -u $USER -h -o "%j:%T:%N:%M" | grep -v vscode | sort | tr '\n' ' '; echo
python3 - <<'PY'
import glob, collections
c=collections.Counter()
for f in glob.glob('runs/sdL2_3epoch/*/*/*/*/*.json'):
    p=f.split('/'); c[(p[3],'att' if 'important_instructions' in f else ('inj' if p[4].startswith('injection_task') else 'ben'))]+=1
exp={'banking':(144,16),'slack':(105,21),'travel':(140,20),'workspace':(560,40)}
print(" | ".join(f"{s} att {c[(s,'att')]}/{exp[s][0]} ben {c[(s,'ben')]}/{exp[s][1]}" for s in exp), "| total att", sum(c[(s,'att')] for s in exp), "/949")
PY
grep -al "Traceback\|OutOfMemory\|TIME LIMIT" logs/eval_sdL2/slurm_eval_sdL2_3epoch_*_315[2-9][0-9][0-9].out 2>/dev/null | grep -v "31528[78]\|31529[0]\|3153[67]" | sed 's/.*slurm_//' | tr '\n' ' '; echo
t=../LLaMA-Factory/saves/qwen3-8b/lora/sdL2_perstep_mh_sft_8k_r64_GA4_qkvo_3epoch_5e-6/trainer_log.jsonl; [ -f $t ] && tail -1 $t | cut -c1-110
