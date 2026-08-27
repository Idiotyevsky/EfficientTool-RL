import asyncio
import json

from verl.tools.schemas import OpenAIFunctionToolSchema

from efficienttool_rl.data import HotpotExample, Passage
from efficienttool_rl.protocol import SEARCH_TOOL_SCHEMA, SYSTEM_PROMPT
from efficienttool_rl.training import to_verl_record
from efficienttool_rl.verl.search_tool import HotpotSearchTool


def schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(SEARCH_TOOL_SCHEMA)


def example() -> HotpotExample:
    return HotpotExample(
        example_id="q1",
        question="Where?",
        answer="London",
        passages=(Passage("A", "Ada was born in London."), Passage("B", "Berlin is in Germany.")),
        supporting_titles=("A",),
        split="train",
    )


def test_verl_record_keeps_answer_out_of_tool_kwargs():
    record = to_verl_record(example(), index=3)
    assert record["agent_name"] == "tool_agent"
    assert record["prompt"][0]["content"] == SYSTEM_PROMPT
    assert record["reward_model"]["ground_truth"] == "London"
    kwargs = record["extra_info"]["tools_kwargs"]["search"]["create_kwargs"]
    assert "answer" not in kwargs
    assert kwargs["passages"][0]["text"] == "Ada was born in London."


def test_hotpot_search_tool_executes_isolated_bm25():
    async def run():
        tool = HotpotSearchTool({"top_k": 2}, schema())
        instance_id, _ = await tool.create(
            create_kwargs={"passages": [{"title": p.title, "text": p.text} for p in example().passages]}
        )
        response, reward, metrics = await tool.execute(instance_id, {"query": "born London"})
        await tool.release(instance_id)
        return json.loads(response.text), reward, metrics

    payload, reward, metrics = asyncio.run(run())
    assert payload["ok"] is True
    assert payload["results"][0]["title"] == "A"
    assert reward == 0.0
    assert metrics["search_calls"] == 1
