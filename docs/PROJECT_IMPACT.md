# Project Impact & Development Context

## Why This Project Exists

Most federated learning research is conducted under network assumptions that do not reflect 
the infrastructure realities of Sub-Saharan Africa (SSA). Benchmark results from well-connected 
research environments in Europe or North America cannot be directly trusted for deployment 
decisions in Zambia, Tanzania, or Kenya — where connectivity is intermittent, bandwidth is 
scarce, and client dropout is the norm rather than the exception.

This project was built to answer a concrete question:

> **How do privacy-preserving federated learning algorithms actually behave when the network 
> looks like rural Zambia — not a university data centre?**

---

## The Problem This Addresses

In SSA, sensitive data exists in abundance: community health records, agricultural yield 
reports, financial transaction histories from mobile money platforms. Federated learning 
offers a path to training useful AI models on this data without centralising it — preserving 
privacy while enabling collective intelligence.

But before deployment, researchers and practitioners need to know:
- Will the global model still converge when 60% of clients drop out every round?
- Does FedProx's proximal term actually help under severe connectivity constraints?
- How much bandwidth does FL consume under realistic SSA conditions?
- At what point does network impairment begin to meaningfully degrade model quality?

No existing open-source simulator answers these questions with SSA-calibrated parameters.

---

## What This Simulator Contributes

1. **SSA-Calibrated Network Profiles** — Four network profiles derived from GSMA 2023 
   mobile connectivity data for Sub-Saharan Africa, covering urban, rural, and severely 
   disrupted connectivity scenarios specific to the region.

2. **Reproducible Comparative Experiments** — 30 multi-seed runs across 6 experiment 
   configurations, with statistical analysis (Wilcoxon signed-rank tests, Cohen's d effect 
   sizes) enabling credible algorithm comparison.

3. **An Honest Finding** — Under MNIST, both FedAvg and FedProx demonstrate surprising 
   resilience to network impairment. Under CIFAR-10, degradation becomes visible. The 
   difference between algorithms is modest but directionally consistent — FedProx shows 
   a medium effect size (Cohen's d = 0.50) under severe disruption that warrants further 
   investigation on harder tasks.

4. **Open Infrastructure for Future Research** — The simulator is fully configurable via 
   YAML, supports custom network profiles, and can be extended to new algorithms, datasets, 
   and aggregation strategies without modifying core code.

---

## Real-World Relevance

The findings from this simulator have direct implications for:

- **Rural health data initiatives** — Community health workers in rural Zambia operating 
  on 2G/3G connections can participate in federated training without their data leaving 
  local devices, but only if the FL system tolerates their dropout rates. This simulator 
  quantifies that tolerance.

- **Agricultural AI** — Smallholder farmer data (crop yields, soil conditions, weather 
  patterns) is geographically distributed and privacy-sensitive. FL is a natural fit, 
  but deployment requires understanding how rural connectivity affects convergence.

- **Mobile money fraud detection** — Financial institutions operating across SSA need 
  fraud detection models trained on transaction data that cannot be centralised due to 
  regulatory and privacy constraints. FL under SSA network conditions is the practical 
  path forward.

---

## Broader Academic Contribution

This project contributes to a growing body of work on **context-aware machine learning 
systems** — the recognition that AI systems must be evaluated in the conditions where 
they will actually be deployed, not idealised laboratory settings.

The simulation framework, calibrated profiles, and experimental methodology developed 
here can serve as a foundation for:
- Future empirical FL studies targeting African deployment contexts
- Curriculum development in African computer science programmes
- Pre-deployment evaluation tools for NGOs and research institutions working in SSA

---

## Personal Motivation

As a computer science student at the University of Zambia, I am acutely aware of the gap 
between the AI systems being built globally and the infrastructure realities of the 
continent I live on. This project is my attempt to build something that is not just 
technically competent, but contextually relevant — research that asks questions about 
Africa, with Africa in mind.

The long-term vision is to extend this work toward real-world pilot deployments in 
collaboration with Zambian health and agricultural institutions, validating simulation 
findings against actual network behaviour in the field.

---

*Frank Meyo, University of Zambia — CSC 4004 Final Year Project, 2026*
