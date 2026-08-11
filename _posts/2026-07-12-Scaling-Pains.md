---
title: "Scaling Pains"
date: 2026-07-12 10:00:00 +0000
categories: [scaling]
---

![Scaling Pains]({{ '/assets/images/scaling-pains.png' | relative_url }})

I'd say I understand how LLMs work. I've trained models, built machine learning systems and read many of the foundational papers. But over the last few years, models have grown from something that could fit on a single GPU to systems trained across thousands of accelerators.

At some point, I realized there was a gap in my understanding. I could explain the Transformer architecture, but what I couldn't explain was how any of this actually scales. Answers to questions like - how do you train a model that doesn't fit on a single machine? What happens when thousands of accelerators need to communicate every step? I sort of had an idea, but I wanted to really dive into it.

Google's *How to Scale Your Model* is one of the few resources I've found that focuses on these questions directly. So here is my journey through the book and my thoughts overall.

## Part 1: Rooflines

The first section of the scaling book introduces rooflines, which is a useful mental model. Before reading this chapter, I mostly thought about model performance in terms of raw compute, but this chapter really made me appreciate how often the bottleneck is actually memory bandwidth.

The roofline model gives a simple way to reason about this tradeoff. Some operations are compute-bound, where performance is limited by the accelerator's peak FLOPs. Others are memory-bound, where the limiting factor is how quickly data can move between the memory and compute units.

I think part 1 took me some time to get used to this sort of thinking model, but once it clicked, the answers to the questions came easier.

## Part 2: TPUs

This section was definitely drier than the other chapters. The main thing I took away was the difference in philosophy between GPUs and TPUs. GPUs were originally built for graphics and later evolved into highly flexible parallel processors, while TPUs are much more specialized. TPUs heavily optimize matrix multiplication using systolic arrays, where data flows through a coordinated pipeline rather than treating operations independently. The questions in this section introduced interesting concepts.

## Part 3: Sharding

This was by far the hardest chapter in the book, and the one I spent the most time on. It's incredibly dense, introducing both the mechanics of sharding tensors across devices and the communication patterns required to make distributed computation work. The exercises were significantly more challenging than the previous chapters, but they also forced me to slow down and build intuition instead of skimming through the material.

One thing I appreciated was how the chapter systematically derives the four ways to perform distributed matrix multiplication depending on how the input matrices are sharded. It also introduces the four core communication primitives: AllGather, ReduceScatter, AllReduce, and AllToAll. Before this chapter, I knew what these operations did at a high level from seeing them in papers. What I didn't understand was the mathematical reasoning behind them: why each primitive is needed, what information is being communicated, and how they naturally arise from different sharding strategies. I found myself constantly coming back to concepts introduced in this chapter.

### Takeaways

<B> The four communication primitives introduced in this chapter: </B>
    
* **AllGather:** Every device collects the shards from all other devices to reconstruct the full tensor. Intuitively, it removes a sharding dimension.

*   **ReduceScatter:** Performs a reduction across devices while simultaneously scattering the reduced result so each device keeps only its assigned shard. Intuitively, it does the opposite of an AllGather and so requires the same amount of time
        
*   **AllReduce:** Values are reduced (e.g., summed) across all devices, and the final result is broadcast back to every device. This is essentially a ReduceScatter followed by and AllGather.
        
*   **AllToAll:** Devices exchange different shards with one another so that each device ends up owning a different partition of the data. Intuitively, this means moving a sharding axis from one dimension to another. 

Distributed matrix multiplication boils down to four sharding cases.

1. No sharding along the contracting dimension

   Each device has all the data it needs, so the multiplication can be performed locally with **no communication**.

   $$
   \mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y]
   \rightarrow
   \mathbf{C}[I_X, K_Y]
   $$

2. One input is sharded along the contracting dimension

   The computation is

   $$
   \mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K_Y]
   \rightarrow
   \mathbf{C}[I, K_Y]
   $$

   Before this can happen, we first perform an AllGather:

   $$
   \mathbf{A}[I, J]
   =
   \mathrm{AllGather}_X(\mathbf{A}[I, J_X])
   $$

3. Both inputs are sharded along the contracting dimension

   Each device computes a partial result locally, and the partial outputs are summed with an **AllReduce** (or **ReduceScatter** if the output should remain sharded).

   $$
   \mathbf{A}[I, J_X] \cdot \mathbf{B}[J_X, K] \rightarrow \mathbf{C}_{\mathrm{partial}}[I, K]
   $$

   Combine partial sums:

   $$
   \mathbf{C}[I, K]
   =
   \mathrm{AllReduce}(\mathbf{C}_{\mathrm{partial}})
   $$

   or, if the output remains sharded:

   $$
   \mathbf{C}[I_X, K]
   =
   \mathrm{ReduceScatter}(\mathbf{C}_{\mathrm{partial}})
   $$

4. Both inputs are sharded along the non-contracting dimension

   One of the matrices must be redistributed before the multiplication can be performed efficiently.

   $$
   \mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_X] \rightarrow \mathbf{C}[I_X, K_X]
   $$

   AllGather one of the dimensions and then perform the matrix multiply.

## Part 4: Seeing the Transformer Come Together

After spending so much time in Part 3 dealing with the mechanics of distributed computation, sharding, and communication, Part 4 felt like a breath of fresh air. Instead of reasoning about abstract matrix multiplications and collective operations in isolation, we finally get to see how those ideas map onto a real transformer.

## Part 5: Parallelism Strategies Behind Modern LLM Training

Part 5 was probably the first section where several concepts that I had only heard mentioned before finally clicked.

Terms like data parallelism, tensor parallelism, pipeline parallelism, and FSDP are everywhere when reading about large-scale training. They often appear as buzzwords in papers and engineering blogs, but without understanding the underlying tensor layouts, it is difficult to understand why one method is chosen over another.

Here are the 4 parallelism schemes discussed in this section. Each scheme can be thought of as uniquely defined by a sharding for `In`, `W_in`, `W_out`, and `Out`.

1. **Data Parallelism (DP)**

   Activations are sharded along the batch dimension, while parameters and optimizer state are replicated on each device.

   $$
   \text{In}[B_X, D] \cdot_D W_{\text{in}}[D, F] \cdot_F W_{\text{out}}[F, D] \rightarrow \text{Out}[B_X, D]
   $$

2. **Fully-Sharded Data Parallelism (FSDP / ZeRO-3)**

   Activations are sharded along the batch dimension, like pure data parallelism. Parameters are sharded along the same mesh axis. Optimizer state is also sharded along the batch dimension. This reduces duplicated memory.

   $$
   \text{In}[B_X, D] \cdot_D W_{\text{in}}[D_X, F] \cdot_F W_{\text{out}}[F, D_X] \rightarrow \text{Out}[B_X, D]
   $$

3. **Tensor Parallelism (Megatron Sharding)**

   Activations are sharded along $D$ ($d_{\text{model}}$), and parameters are sharded along $F$ ($d_{ff}$).

   $$
   \text{In}[B, D_Y] \cdot_D W_{\text{in}}[D, F_Y] \cdot_F W_{\text{out}}[F_Y, D] \rightarrow \text{Out}[B, D_Y]
   $$

4. **Mixed Parallelism**

    Modern frontier models usually combine all of these techniques.


    Syntax:

    $$
    \text{In}[B_X, D_Y] \cdot_D W_{\text{in}}[D_X, F_Y] \cdot_F W_{\text{out}}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]
    $$
   

5. **Pipeline Parallelism**

   Weights are sharded along the layer dimension, and activations are microbatched and rolled along the layer dimension. Communication between pipeline stages is minimal — just moving activations over a single hop.

   $$
   \text{In}[L_Z, B, D][i] \cdot_D W_{\text{in}}[L_Z, D, F][i] \cdot_F W_{\text{out}}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D][i]
   $$

### Takeaways: The Different Ways to Shard a Transformer

| Strategy | What is Sharded? | Main Communication |
|---|---|---|
| Data Parallelism | Batch dimension | All-reduce gradients |
| FSDP | Parameters, gradients, optimizer states | All-gather parameters |
| Tensor Parallelism | Matrix dimensions | All-reduce/all-gather activations |
| Pipeline Parallelism | Transformer layers | Send activations between stages |
| Mixed Parallelism | Combination of all above | Combination of collectives |

## Part 6: Training LLaMA 3 on TPUs

Everything from Parts 1-5 - rooflines, communication primitives, the four parallelism schemes - gets pointed at a single, concrete target: training LLaMA 3-70B on a TPU v5p pod. I think this is the section that made everything else click, because instead of reasoning about generic matrices `A[I_X, J]` I was plugging in real numbers for a model I already knew existed and had used.

It was also the first time the book made me curious about a model's architecture for its own sake rather than as a vehicle for a sharding exercise. Once you've counted up LLaMA's parameters by hand and watched how lopsided the distribution is toward the MLP block, you start wondering how that compares to other model families - what Mixtral's MoE layers do to that ratio, how DeepSeek's MLA changes the attention parameter count, whether Gemini or Grok make different tradeoffs. That curiosity is really the best sign the earlier chapters actually landed.

### Takeaways

**Where LLaMA 3-70B parameters actually live:**

| Component | Parameters | Share |
|---|---|---|
| MLP weights | 56.3B | ~80% |
| Attention | 12B | ~17% |
| Vocabulary embeddings | 2.1B | ~3% |


Seeing how the parallelism strategies compare to a model the size of LLaMA was awesome. 

## Part 7: Inference

I hadn't fully appreciated how much there was to inference until this chapter. Training gets all the attention (pun intended), but serving turns out to have its own entirely separate set of bottlenecks, and prefill and generation behave nothing alike.

Prefill is usually compute-bound, since prompts tend to be longer than ~240 tokens, which is enough to push the matmuls into the compute-bound regime of the roofline. Generation is the opposite story: attention is always memory-bandwidth-bound, since it has to read the entire KV cache for every single token generated, while the MLP can go either way depending on batch size - compute-bound if the batch is large enough to amortize weight loading, memory-bound if it isn't.

### Takeaways

The general formula for theoretical step time during generation:

$$
\begin{align} \tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}} \end{align}
$$

Attention is a fixed bandwidth tax that scales with batch size and KV cache size - there's no way around reading the whole cache every step. The MLP term is a max of a compute term and a memory term, so it's compute-bound once batch size is large enough that $\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}$ exceeds $\frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}$, and memory-bound otherwise.

This also finally explained why so much recent model design - MQA, GQA, MLA, sliding window attention, and so on - is really all pointed at the same target: shrinking the KV cache. Since attention is always bandwidth-bound on the cache, cutting its size is one of the few direct levers on generation latency.

## Part 8: Serving LLaMA on TPUs

Short section, but a nice one - same flavor as Part 6, taking Part 7's formulas and grounding them in LLaMA. Working through KV cache size, batch size, and latency/throughput tradeoffs for a real model made the theory feel concrete again after the abstraction of Part 7.

## Part 9: Profiling JAX Code

This one was a bigger learning curve than the theory itself, since I was coming into it as a JAX beginner. Reading about rooflines and sharding is one thing; actually profiling code and matching what I saw to those concepts was another.

The tooling ended up being half the battle. TensorBoard profiling is natively supported in Colab, but Colab had dropped v5e-8 support, so I had to move to Kaggle instead - where TensorBoard isn't native. That meant every single profile was a manual round trip: capture a trace, download it, upload it to Perfetto to actually view it. Tedious, but it forced me to get comfortable poking around traces by hand instead of leaning on a nicely integrated viewer, which was probably good for me in the end.

## Part 10: Programming JAX

This was the hardest part of the whole book for me. JAX is a genuinely different beast, and getting comfortable thinking functionally instead of imperatively took real time. Some of the questions I just sat with for a while before anything clicked - but once the answers came, they came sweet.

By the end, I was actually comfortable with JAX, which wasn't true going in. The standout exercise was building a fully sharded MoE layer - it forced me to pull together sharding, communication primitives, and parallelism strategies from every earlier chapter and actually implement them, not just reason about them on paper. This section took the longest by a wide margin, but it's also the one I learned the most from.

Part of getting comfortable was understanding JAX's three sharding modes. In **auto** mode, you just tell the compiler your input/output shardings and let XLA/Shardy figure out everything in between. In **explicit** mode, JAX itself (not the compiler) propagates sharding through each op and yells at you when it's ambiguous, so you get more visibility and safety than auto. In **manual** mode (`shard_map`), you get a fully device-local view of the program and write every piece of communication yourself.

## Part 12: GPUs

Admittedly I went through this one lightly, but it was interesting seeing GPUs and TPUs compared directly after spending the whole book thinking in TPU terms. The big difference is topology: GPUs connect in a hierarchical, fat-tree fashion - 8 GPUs per node over NVLink, then nodes stitched together over InfiniBand switches - while TPUs sit on a uniform 2D/3D torus where every chip just talks to its neighbors. That tree structure is also why GPU collectives are messier in practice, degrading well below peak bandwidth unless message sizes are large, whereas TPUs stay closer to their theoretical numbers by design.

## Conclusion

Start to end, this took me about two months. Some sections were genuinely hard, and I'd tell anyone going through this not to get demotivated if a chapter kicks your ass. That's kind of the point. I'm honestly very glad I went through the whole thing.

As for what's next, I'm not really sure - there are so many directions this could lead. That's a good problem to have.