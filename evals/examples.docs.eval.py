"""Example documentation-lift eval suite.

Runs inside the eval container through ``python -m pi.evals.cli``; each case
asks the agent for a strict JSON answer that the structured-output judge
scores. Replace these cases with real documentation-lift scenarios.
"""

from pi.evals.framework import describe_eval, structured_output_judge
from pi.evals.harness import create_pi_documentation_eval_harness

harness = create_pi_documentation_eval_harness()
judge = structured_output_judge(expected={"ok": True}, match="strict", allow_extras=False)


@describe_eval("Examples", harness=harness, judges=[judge], judge_threshold=None)
async def answers_with_json(run):
    """Reply with the requested JSON and nothing else."""
    await run('Reply with exactly this JSON and nothing else: {"ok": true}')
