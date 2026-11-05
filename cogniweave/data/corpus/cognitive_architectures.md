# Cognitive architectures for language agents

Cognitive architectures such as Soar and ACT-R model intelligence as a cycle of perception, memory retrieval, decision making and learning, operating over distinct memory systems. These ideas from cognitive science give language agents a principled structure instead of an ad hoc prompt chain.

The CoALA framework describes language agents in terms of working memory, long-term memories, an action space and a decision procedure. Long-term memory is divided into episodic memory for experiences, semantic memory for facts and procedural memory for skills and strategies.

Working memory is the agent's limited attention span. Because the context window is finite, an agent must decide which retrieved items, lessons and observations deserve a place in the prompt, and evict the least salient ones when the budget is exceeded.

The ReAct pattern interleaves reasoning steps with tool actions and observations. Grounding each step in an observation from a tool reduces hallucination, but every extra step adds latency and cost, so agents should stop as soon as they have enough evidence.

Reflexion lets an agent learn from feedback without updating model weights. After a failure, the agent writes a short verbal lesson into episodic memory and retrieves it on the next attempt, which can improve performance across repeated tasks.

Metacognition means an agent monitors its own confidence. When confidence is low, a well-designed system should escalate, gather more evidence or ask for review rather than passing an uncertain answer downstream.

Procedural learning can be framed as a bandit problem. An agent that tries several strategies and tracks the reward each one earns can gradually prefer the strategy that works best for its role, while still exploring alternatives.
