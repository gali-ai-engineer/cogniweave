# Memory for agents

Retrieval for agent memory should balance relevance, recency and importance. The Generative Agents work scored each memory as a weighted sum of these three signals, so that a vivid lesson from last week can outrank a trivial observation from a minute ago.

Episodic memory stores what happened: the task, the action taken and the outcome. Recording outcomes allows later runs to reuse what worked and avoid repeating mistakes, which is the basis of learning across sessions.

Semantic memory holds knowledge that is not tied to a particular episode, such as documents, facts and definitions. Chunking documents into passages and indexing them with embeddings lets agents retrieve evidence and cite its source.

Shared memory between agents makes collaboration possible but introduces risk. Without provenance and confidence on each entry, one agent's mistaken note can silently contaminate the work of every other agent.

Memory grows without bound unless it is managed. Practical systems summarise or expire old episodes, cap the size of stored items and keep the most important lessons, otherwise retrieval quality and cost both degrade over time.

Persisting memory to disk turns a stateless model into an agent that improves with use. A lesson learned from a critic in one run can be recalled in the next run, so the same mistake does not need to be corrected twice.
