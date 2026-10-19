# Colosseum study data

Colosseum does not ship a downloadable dataset. Terrarium builds Jira tickets, meetings, and hospital patients from a YAML seed at runtime. The files here are a small, local stand-in you can inspect and run without a GPU.

## What is in this folder

| Path | What it is |
| --- | --- |
| `collab_problems_small.json` | 3-agent generator knobs for CoLLAB wardrobe, meeting, and smart-grid tasks |
| `instances/<problem>/study_small.json` | The generated task: agents, variables, domains, factors, JSON schemas |
| `instances/<problem>/study_small.pkl` | Same instance for CoLLAB's symbolic solver and debug protocol |
| `collusion_one_seed.yaml` | One OpenAI `gpt-4o-mini` collusion sweep (3 agents, 1 seed, secret channel off) |
| `collusion_one_seed_ollama.yaml` | Same sweep pointed at local Ollama `llama3.2:3b` |
| `results/collusion_one_seed_ollama/` | Snapshot of the first completed local run (summary, metrics, blackboard, turns) |

Open `instances/personal_assistant/study_small.json` first. Avery, Jordan, and Taylor each pick a numbered outfit and must match or contrast colours.

## How Colosseum works

1. A YAML file sets the seed, environment, graph topology, model, and sweep factors.
2. Agents sit on a communication graph (`complete`, `path`, `star`, ...). Edges become shared blackboards.
3. A coalition can get a secret blackboard and an injected prompt. This study YAML turns that channel **off** (`secret_channel_enabled: [false]`) and uses `prompt_variants: [control]`.
4. Metrics are joint reward, coalition regret, and message counts. Outputs land under `experiments/collusion/outputs/<tag>/`.

Paper-scale commands are in the repo root README. Those configs call OpenAI, Together, Anthropic, Gemini, or a local vLLM GPU server. API cost grows with the number of agents.

## Install Colosseum (LLM sweeps)

From the Colosseum repo root:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv sync --no-install-project
cp .env.example .env   # fill OPENAI_API_KEY
```

## Run the no-key CoLLAB dataset

CoLLAB generation and the symbolic solver use only the Python standard library.

```bash
cd external/CoLLAB

python example01_symbolic.py ../../study_data/instances/personal_assistant
python example01_symbolic.py ../../study_data/instances/meeting_scheduling
python example01_symbolic.py ../../study_data/instances/smart_grid
```

Expected shape of the printout: `solver=problem_specific | total=... | min=... | max=... | ratio=... | time=...s`.

On this checkout the wardrobe instance scores `ratio=1.000` for the problem-specific solver (utility 10 / 10).

Optional mock multi-agent protocol (still no API key, but needs `pip install requests`):

```bash
cd external/CoLLAB
python example03_icc_dm.py ../../study_data/instances/meeting_scheduling --provider debug --rounds 1
```

### Regenerate the instances

From the Colosseum repo root, with `PYTHONPATH=external/CoLLAB`:

```bash
python3 - <<'PY'
from pathlib import Path
import json, sys
root = Path(".").resolve()
sys.path.insert(0, str(root / "external" / "CoLLAB"))
from problem_layer.personal_assistant.problem import PersonalAssistantConfig, InstructionMode, generate_instance as gen_pa
from problem_layer.meeting_scheduling.problem import MeetingSchedulingConfig, InstructionMode as MSMode, generate_instance as gen_ms
from problem_layer.smart_grid.problem import SmartGridConfig, InstructionMode as SGMode, generate_instance as gen_sg
cfg = json.loads((root / "study_data" / "collab_problems_small.json").read_text())
out = root / "study_data" / "instances"
pa, ms, sg = cfg["personal_assistant"], cfg["meeting_scheduling"], cfg["smart_grid"]
gen_pa(PersonalAssistantConfig(num_agents=pa["num_agents"], density=pa["density"], min_outfits_per_agent=pa["min_outfits_per_agent"], max_outfits_per_agent=pa["max_outfits_per_agent"], instruction_mode=InstructionMode.TEXT, include_collages=False, output_stem="study_small", rng_seed=pa["seed"]), out / "personal_assistant")
gen_ms(MeetingSchedulingConfig(num_agents=ms["num_agents"], num_meetings=ms["num_meetings"], timeline_length=ms["timeline_length"], soft_meeting_ratio=ms["soft_meeting_ratio"], min_participants=ms["min_participants"], max_participants=ms["max_participants"], instruction_mode=MSMode.TEXT, include_timelines=False, output_stem="study_small", rng_seed=ms["seed"]), out / "meeting_scheduling")
gen_sg(SmartGridConfig(num_agents=sg["num_agents"], timeline_length=sg["timeline_length"], min_sources_per_agent=sg["min_sources_per_agent"], max_sources_per_agent=sg["max_sources_per_agent"], min_machines_per_agent=sg["min_machines_per_agent"], max_machines_per_agent=sg["max_machines_per_agent"], instruction_mode=SGMode.TEXT, include_charts=False, output_stem="study_small", rng_seed=sg["seed"]), out / "smart_grid")
print("rewrote", out)
PY
```

`example00_generate.py` also accepts this JSON, but it always writes under `external/CoLLAB/outputs/examples/01_generate/` rather than `study_data/instances/`.

## Run one Colosseum collusion sweep

OpenAI path (`study_data/collusion_one_seed.yaml`) needs `OPENAI_API_KEY`. Direct OpenAI calls are geo-blocked in some regions (`unsupported_country_region_territory`).

Local Ollama path (already run once on this machine):

```bash
ollama serve   # if not already running
source .venv/bin/activate
python -m experiments.collusion.run --config study_data/collusion_one_seed_ollama.yaml
```

That is 3 agents on a complete graph, 1 seed, 2 colluders, secret channel off, control prompt, model `llama3.2:3b`. Full artifacts land under `experiments/collusion/outputs/collusion_one_seed_ollama/<timestamp>/` (gitignored). A copy of the first successful run is in `study_data/results/collusion_one_seed_ollama/`.

Do not start with `experiments/collusion/configs/collusion_jira_topologies.yaml`. That config boots local vLLM (`Qwen2.5-7B`) on a GPU.
