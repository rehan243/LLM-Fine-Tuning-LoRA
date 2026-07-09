# Engineering Log

Running notes on design decisions and lessons learned.


### 2026-07-06

Fine-tuning with LoRA can significantly reduce the number of trainable parameters, which helps in speeding up training and lowering resource costs. However, I found that the quality of the model's performance can sometimes plateau if the rank of the adapters is too low; I inadvertently set it to 4 initially and had to increase it to 16 to see a meaningful improvement in downstream tasks. This tradeoff between efficiency and performance is something to keep in mind during experimentation.

### 2026-07-07

**LoRA/QLoRA Fine-Tuning and Adapter Management**

**Observation:** After fine-tuning a large language model using LoRA and QLoRA, I noticed that the adapter size significantly impacted the model's inference speed. For instance, using 64-bit LoRA adapters reduced the inference time by 15% compared to 16-bit, but at the cost of increased memory usage. This tradeoff highlights the importance of balancing model size and performance for real-time applications.

### 2026-07-09

During QLoRA fine-tuning, I noticed that using smaller rank values (e.g., 4 or 8) significantly reduces VRAM usage but can lead to unstable training if the learning rate isn't adjusted downward accordingly. It's crucial to tune the optimizer parameters carefully because too high a learning rate with low-rank adapters tends to cause divergence.
