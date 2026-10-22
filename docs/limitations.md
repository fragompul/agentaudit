# Limitations

* **Small local models.** The real evaluation uses 1B-7B quantised models on one machine; absolute scores and failure modes do not transfer to frontier agents. Generality of the conclusions rests on the simulation study and on the re-analysis of public results.
* **Simulated environments** are cleaner than real software tasks, and the checkers verify only state that the generator anticipates.
* **Tasks are i.i.d. draws** from a population defined by our generator; inference concerns that population, not "agent ability" in general.
* **Linear decomposition** of binary outcomes; components are not latent-scale quantities.
* **Public re-analysis** assumes each submission was evaluated on all 500 SWE-bench Verified instances (the files do not state it) and ignores scaffold differences.
* **Confidence sequences** are conservative: they trade power for the right to stop at any time.
