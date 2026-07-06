# Engineering Log

Running notes on design decisions and lessons learned.


### 2026-07-06

Fine-tuning with LoRA can significantly reduce the number of trainable parameters, which helps in speeding up training and lowering resource costs. However, I found that the quality of the model's performance can sometimes plateau if the rank of the adapters is too low; I inadvertently set it to 4 initially and had to increase it to 16 to see a meaningful improvement in downstream tasks. This tradeoff between efficiency and performance is something to keep in mind during experimentation.
