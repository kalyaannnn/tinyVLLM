1.	BlockTable cleanup:
	•	Add a release(alloc: BlockAllocator) method to BlockTable to free all physical blocks via alloc.free(self.blocks) and then clear the table.
	•	Call bt.release(alloc) when a request finishes, ensuring reclaimed blocks can be reused.
	2.	Dynamic request admission & scheduling:
	•	Modify the engine to accept new requests while decoding others. Maintain separate queues for “waiting to prefill” and “ready to decode.”
	•	Each tick should prefill a limited number of waiting requests (based on available blocks or time budget), then decode one token for all requests that have completed prefill.
	•	Only rebuild block tables when block tables actually grow.
	3.	Engine metrics & logging:
	•	Track and log the number of free and used blocks, current batch size, maximum sequence length, prefill latency (ms), and decode throughput (tokens/s).
	•	Print or record these metrics at regular intervals to evaluate performance.
	4.	KV memory reuse & eviction (mandatory):
	•	Ensure all finished requests free their blocks (release method above).
	•	Implement a basic eviction or preemption policy when the allocator runs out of blocks—for example, pausing the longest-running low‑priority request and recomputing its KV later.
	•	This prevents memory leaks and enables serving long or many requests.
	5.	Prefill performance improvements:
	•	Replace the Python‑loop reference prefill with a fused Triton kernel for causal attention so prompt processing scales efficiently.
	•	Validate the fused kernel against the reference prefill on small inputs before using it in the engine.
	6.	Model generalization:
	•	Add Rotary Position Embedding (RoPE) and Grouped Query Attention (GQA) support to handle Llama/Mistral/Gemma‑style models.
	•	Verify attention outputs and logits on small RoPE models before scaling up to 7B-class models.
	7.	Speculative decoding (mandatory):
	•	Integrate a draft model to propose multiple tokens, then use the paged decode kernel to verify and commit as many tokens as possible, rolling back the rest.
	•	This improves throughput on long generations and should be part of the final runtime.
	8.	End‑to‑end demo & benchmarks:
	•	Build a script that accepts multiple prompts, runs them through your engine (prefill + decode + sampling), and returns completions.
	•	Benchmark total throughput (tokens/s), memory usage, and latency across different model sizes (e.g., GPT‑2, Llama‑7B) and batch sizes.
	•	Use these metrics to iterate on scheduling, block reuse, and kernel performance.
