# When Retraining Changes the Explanation: Evaluating the Stability of Post-Hoc GNN Explainers on Molecular Graphs

## Abstract

Graph Neural Networks (GNNs) are increasingly deployed for molecular toxicity and bioactivity prediction, where post-hoc explainability methods are used to extract chemical toxicophores. While existing research evaluates explainers on feature attribution accuracy, the stability of these explanations under model retraining remains largely unmeasured. In this work, I evaluate three explainer families (GNNExplainer, Gradient x Input, and GNN-LRP) across 10 independent training runs of a Graph Convolutional Network on the MUTAG benchmark. I find that GNNExplainer exhibits severe instability across random seeds, achieving a mean cross-seed Jaccard similarity of only 0.157 on top-3 attributed subgraphs. In a notable failure case on Molecule 0, two models with identical architecture and correct mutagenicity predictions produce completely disjoint explanations (Jaccard = 0.000), with one run highlighting the toxicophoric nitro group (-NO2) and the other highlighting inert aromatic carbons. In contrast, Gradient x Input achieves a mean Jaccard score of 0.481, providing more than 3x higher reproducibility across retraining runs. These results demonstrate that high predictive accuracy does not imply explanation stability, and post-hoc graph explainers must be benchmarked for reproducibility before deployment in safety-critical chemistry workflows.

---

## 1. Introduction

Graph Neural Networks have become a standard approach for predicting molecular properties directly from chemical structures. When a model predicts that a candidate compound is toxic or mutagenic, domain experts need to inspect the rationale behind the decision. Post-hoc explainability methods aim to meet this requirement by identifying the compact subgraph or set of atoms most responsible for the graph-level prediction.

In drug discovery and toxicology, an unreliable explanation carries real costs. If an explainer points to an inert hydrocarbon chain rather than the active toxicophore, chemists may misdirect lead optimization or discard safe scaffolds. More critically, an explainer is untrustworthy if simply retraining the underlying model with a different random seed yields a completely different set of explanatory atoms.

Prior benchmarks typically evaluate explainers using fidelity or synthetic ground truth on static checkpoints. Whether standard graph explainers generate consistent explanations across stochastic retraining runs remains understudied. In this paper, I train identical 3-layer GCN architectures across 10 random seeds on the MUTAG dataset and evaluate the cross-seed stability of GNNExplainer, Gradient x Input, and GNN-LRP.

---

## 2. Related Work

Post-hoc GNN explainability began with optimization-based methods, primarily GNNExplainer (Ying et al., 2019), which learns continuous masks over graph structure and node attributes by maximizing mutual information with the model output. In parallel, first-order gradient methods adapted from computer vision, such as Saliency Maps and Gradient x Input (Simonyan et al., 2013; Shrikumar et al., 2017), compute input feature importance via backpropagated gradients. Layer-wise Relevance Propagation (Bach et al., 2015) was extended to graph message-passing operations (GNN-LRP) to decompose graph-level predictions back to individual input nodes without explicit mask optimization. While recent literature in vision and tabular domains has questioned the robustness and stability of feature attributions under data and weight perturbations (Agarwal et al., 2022), systematic stability assessments across retraining runs in molecular graph classification remain sparse.

---

## 3. Methodology

### 3.1 Dataset and Model Architecture

I evaluate on the MUTAG benchmark (Debnath et al., 1991), consisting of 188 nitroaromatic and heteroaromatic compounds evaluated for mutagenicity in *Salmonella typhimurium*. Each molecule is represented as an undirected graph with 7-dimensional one-hot node features indicating atom type.

The predictive model is a 3-layer Graph Convolutional Network (Kipf and Welling, 2017):
- 3 sequential GCNConv layers with hidden dimension 64 and ReLU activations.
- Global mean pooling to aggregate node representations into a graph embedding.
- A 2-layer classification head with Dropout (p = 0.5) and ReLU, producing logits over 2 classes (non-mutagenic vs mutagenic).

I train the model across 10 distinct random seeds (0, 1, 7, 13, 21, 42, 99, 123, 256, 512) on an 80/20 train/test split. Test accuracy across the 10 seeds ranges from 71.05% to 89.47% (mean: 79.47%), confirming that all trained checkpoints achieve valid convergence.

### 3.2 Explainability Methods

I benchmark three distinct algorithmic paradigms:

- **GNNExplainer (Optimization-based):** For a trained model f and target molecule G, GNNExplainer optimizes a continuous edge mask M and node feature mask F to maximize the mutual information between f(G) and f(G * M), constrained by sparsity and entropy regularizers.
- **Gradient x Input (First-order Attribution):** Computes importance for each node v as the dot product between the input node feature vector x_v and the gradient of the target class logit with respect to x_v: S(v) = |x_v * (df_c / dx_v)|.
- **GNN-LRP (Relevance Propagation):** Decomposes the final prediction score recursively backwards through linear and graph convolutional layers following conservation rules, attributing a scalar relevance score R(v) to each atom.

### 3.3 Stability Evaluation Protocol

To quantify explanation stability under retraining, I select fixed test probe molecules and extract the top-k most important atoms (k = 3) identified by each method on each trained seed. For any two seeds s_i and s_j, I compute the Jaccard similarity coefficient between their top-k node sets:

$$J(S_i, S_j) = \frac{|S_i \cap S_j|}{|S_i \cup S_j|}$$

I compute J(s_i, s_j) for all 45 unique pairs among the 10 seeds. The mean Jaccard score across all pairs and probe molecules serves as the primary stability metric, where 1.0 represents perfect reproducibility and 0.0 indicates completely disjoint explanations.

---

## 4. Results

### 4.1 Quantitative Stability Benchmark

Table 1 and Figure 1 summarize the cross-seed stability results across all 10 model initializations.

**Table 1: Cross-seed explanation stability across 10 retraining runs (top-3 node Jaccard similarity)**

| Explainer | Family | Mean Jaccard | Std Dev | Stability Rank |
|:---|:---|:---:|:---:|:---|
| Gradient x Input | First-order gradient | 0.481 | 0.290 | 1 (Highest) |
| GNN-LRP | Relevance propagation | 0.315 | 0.339 | 2 |
| GNNExplainer | Subgraph mask optimization | 0.157 | 0.159 | 3 (Lowest) |

<p align="center">
  <img src="figures/jaccard_stability_chart.png" alt="Figure 1: Cross-seed stability comparison" width="80%">
</p>

*Figure 1: Mean pairwise Jaccard similarity of top-3 explanatory nodes across 10 random initialization seeds. Gradient x Input achieves more than 3x higher cross-seed stability than GNNExplainer.*

Gradient x Input achieves the highest cross-seed stability with a mean Jaccard score of 0.481, outperforming GNNExplainer by more than a factor of three. GNN-LRP occupies the middle tier with a mean Jaccard score of 0.315. GNNExplainer exhibits the lowest stability at 0.157, indicating that explanations from two retraining runs share fewer than one atom in three on average.

### 4.2 Qualitative Failure Case: Complete Explanation Divergence

Figure 2 illustrates a critical failure mode observed on Molecule 0 (true class: mutagenic). Both Seed 42 and Seed 0 classify the molecule correctly as mutagenic, but their GNNExplainer attributions diverge completely.

<p align="center">
  <img src="figures/failure_mol0_seed42_vs_0.png" alt="Figure 2: GNNExplainer explanation divergence on Molecule 0" width="95%">
</p>

*Figure 2: GNNExplainer produces contradictory explanations for Molecule 0 (mutagenic) across two retraining runs. Seed 42 attributes prediction to the -NO2 group (N and O nodes), while Seed 0 attributes it to three aromatic carbon atoms. Both models predict correctly, yielding Jaccard overlap = 0.000.*

**Reading Figure 2:**
- **Seed 42 (Left):** Highlights Nitrogen (Node 4) and Oxygen (Node 14) along with connecting Carbon (Node 16). The explainer correctly detects the nitro group (-NO2), the verified biochemical toxicophore responsible for mutagenicity in nitroarenes.
- **Seed 0 (Right):** Highlights three aromatic ring carbons (Nodes 3, 17, and 18) located on the opposite side of the ring system, leaving the nitro group completely unhighlighted.
- **Divergence:** The shared top-3 atom set is empty, yielding a Jaccard overlap of exactly 0.000.

Both models achieve high confidence on the correct class, but one produces a chemically sound explanation while the other produces an artifact.

### 4.3 Ground Truth Success Cases

On other molecules, such as Molecule 3 (Figure 3), all three explainability methods consistently highlight the nitro group across multiple seeds.

<p align="center">
  <img src="figures/mol3_comparison.png" alt="Figure 3: Explanation benchmark on Molecule 3" width="95%">
</p>

*Figure 3: Attribution maps on Molecule 3. When optimization converges cleanly, all three methods correctly identify the nitro group (-NO2) as the primary mutagenic driver.*

This indicates that GNNExplainer is not fundamentally incapable of finding true toxicophores, but rather that its convergence to the true toxicophore is highly contingent on the initial network weights.

---

## 5. Discussion

### 5.1 Mechanistic Drivers of Explainer Instability

The dramatic variance among explainer families stems from their computational formulations:

1. **Optimization Landscapes in GNNExplainer:** GNNExplainer solves a non-convex optimization problem per molecule. Because each training seed converges to a distinct point in weight space, the loss surface of mutual information differs across runs. The optimizer frequently settles into distinct local minima, attributing importance to whichever peripheral nodes happen to align with the specific weight initialization.
2. **Determinism of Gradient x Input:** Gradient x Input requires no auxiliary optimization. It computes the directional derivative through the frozen computation graph in a single backward pass. While the weights differ across seeds, the primary gradient flow remains concentrated along the input features that drive the final linear classifier, yielding substantially more stable attributions.
3. **Layer Accumulation in GNN-LRP:** GNN-LRP propagates relevance backward layer by layer through all five parametric transformations (three GCNConv and two linear layers). Because small variations in weight matrices compound multiplicatively at each layer, relevance propagation exhibits greater cross-seed variance than single-step gradients, while remaining more consistent than mask optimization.

### 5.2 Accuracy Does Not Guarantee Explanation Reliability

The failure on Molecule 0 highlights a critical principle for interpretable machine learning: predictive accuracy is not a proxy for explanation validity. Two models can learn identical input-output mappings on the test set while relying on different internal feature combinations. If an explainer is unstable across retraining, domain scientists cannot determine whether a highlighted substructure represents genuine chemical causation or model-seed idiosyncrasy.

---

## 6. Conclusion and Practical Recommendations

In this work, I demonstrated that post-hoc GNN explanations on molecular graphs can be highly unstable across standard model retraining runs. GNNExplainer occasionally discovers valid chemical toxicophores (e.g., nitro groups in Molecule 3), but can shift focus entirely to irrelevant carbon atoms under a simple seed change (Jaccard = 0.000 in Molecule 0). Gradient x Input yields over 3x higher cross-seed stability (0.481 vs 0.157), offering a more reliable baseline for safety-critical deployment.

For computational chemists and ML practitioners deploying GNNs in drug discovery:
1. Never rely on a single-seed explanation from an optimization-based explainer.
2. Use multi-seed explanation consensus or deterministic gradient baselines before selecting chemical substructures for wet-lab validation.
3. Future research should evaluate explanation stability on larger Graph Foundation Models (GFMs) under parameter-efficient fine-tuning and prompt tuning.

---

## References

1. Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). GNNExplainer: Generating explanations for graph neural networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
2. Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional networks: Visualising image classification models and saliency maps. *arXiv preprint arXiv:1312.6034*.
3. Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K. R., & Samek, W. (2015). On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. *PLoS ONE*, 10(7), e0130140.
4. Debnath, A. K., Lopez de Compadre, R. L., Debnath, G., Shusterman, A. J., & Hansch, C. (1991). Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds. *Journal of Medicinal Chemistry*, 34(2), 786-797.
5. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR)*.
6. Agarwal, C., Krishna, S., Saxena, E., Pawelczyk, M., Johnson, N., Puri, I., Zitnik, M., & Lakkaraju, H. (2022). OpenXAI: Towards a transparent evaluation of model explanations. *Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*.