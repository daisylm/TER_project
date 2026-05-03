Linux and Hardware optimization for Real Time Workloads.

Our project is not about building an IA system, or optimize it's functionalities but to study it's behavior when it runs on a computer as a Program, our<u> Stress-testing tool </u>to push the system to its limits.
Its a modern and smart direction, because today, IA workloads are among the most demanding types of computation that run on systems.

Researchers study topics such as CPU versus GPU performance, memory usage during inference, and scheduling strategies for handling multiple LLM requests. Systems like vLLM have been developed specifically to improve memory efficiency and throughput when serving LLMs. Other works analyze how inference workloads behave on different hardware architectures and how resource utilization changes depending on the model and configuration.

In our case, we are using **LLM inference as a tool** to <u>study the system behavior</u> : the intersection of AI and oprating Systems.

**Step 1:**

I built a python script that generates a set of prompts, I determined that the content of the prompt impacts strongly the system resources. SO, we need diagnostic prompt classes that activate different parts of the system isolation.
| prompt     | Characteristics of the prompt                                       |
| ------------ | ----------------------------------------------------------------------- |
| Baseline   | simple prompts ( ask the LLM to do the summerization or explannation) |
| Ram_stress | very large context, many sections, long-range references              |
| CPU_stress | smaller context, multiple scenarios, comparison task, multi-tasks     |

Next step: Introduce a new prompt category : **Mixed stress** prompt that combines all three categories

Initially, we use a **prompt_generator.py** to generate the 3 categories of prompts. 
We generated 4 prompts from each category ( 12 in Total ) : First, I thought of using some tools to generate automatically the prompts, but then I thought we won't have control over these tools, since the content of the prompts is an essential element to reveal the bottleneck, then I switched to this plan which use a script that generates a controlled set of prompt workloads.

The prompts  are stored in a Json file.

The **main_benchmark.py** calls the Json file with prompts, initiate the ollama server instant, start perf record/stat send the prompts to the LLM ( tinyllama), then generates flame graphs of each test, as I programmed it to run the test 4 times for each prompt to ensure test reliability and to allow me to compare between one graph and another.
The benchmark, produces a Json file with each prompt and it's output : RAM usage, execution time, prompt ID, test id.



