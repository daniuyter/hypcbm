import torch
import torch.nn as nn

class PosthocLinearCBM(nn.Module):
    def __init__(self, concept_bank, backbone_name, idx_to_class=None, n_classes=5):
        """
        NOTE: ADAPTED FROM POST-HOC-CBM, Yuksekgonul et al., 2023.

        PosthocCBM Linear Layer. 
        Takes an embedding as the input, outputs class-level predictions using only concept margins.
        Args:
            concept_bank (ConceptBank)
            backbone_name (str): Name of the backbone, e.g. clip:RN50.
            idx_to_class (dict, optional): A mapping from the output indices to the class names. Defaults to None.
            n_classes (int, optional): Number of classes in the classification problem. Defaults to 5.
        """
        super(PosthocLinearCBM, self).__init__()
        # Get the concept information from the bank
        self.backbone_name = backbone_name
        self.cavs = concept_bank.vectors
        self.intercepts = concept_bank.intercepts
        self.norms = concept_bank.norms
        self.names = concept_bank.concept_names.copy()
        self.n_concepts = self.cavs.shape[0]

        self.n_classes = n_classes
        # Will be used to plot classifier weights nicely
        self.idx_to_class = idx_to_class if idx_to_class else {i: i for i in range(self.n_classes)}

        # A single linear layer will be used as the classifier
        self.classifier = nn.Linear(self.n_concepts, self.n_classes)

    def compute_dist(self, emb):
        # Computing the geometric margin to the decision boundary specified by CAV.
        margins = (torch.matmul(self.cavs, emb.T) +
           self.intercepts) / (self.norms)
        return margins.T

    def forward(self, emb, return_dist=False):
        x = self.compute_dist(emb)
        out = self.classifier(x)
        if return_dist:
            return out, x
        return out
    
    def forward_projs(self, projs):
        return self.classifier(projs)
    
    def trainable_params(self):
        return self.classifier.parameters()
    
    def classifier_weights(self):
        return self.classifier.weight
    
    def set_weights(self, weights, bias):
        self.classifier.weight.data = torch.tensor(weights).to(self.classifier.weight.device)
        self.classifier.bias.data = torch.tensor(bias).to(self.classifier.weight.device)
        return 1

    def analyze_classifier(self, k=5, print_lows=False):
        if hasattr(self.classifier, "weight") and self.classifier.weight is not None:
            weights = self.classifier.weight.clone().detach()
        elif hasattr(self.classifier, "z") and hasattr(self.classifier.z, "tensor"):
            weights = self.classifier.z.tensor.clone().detach().T
        else:
            raise AttributeError("Unsupported classifier type for weight analysis.")
        output = []

        if len(self.idx_to_class) == 2:
            weights = [weights.squeeze(), weights.squeeze()]
        sparsity_eps = 1e-2
        if isinstance(weights, torch.Tensor):
            global_sparsity = float((weights.abs() <= sparsity_eps).float().mean().item())
        else:
            # binary case uses list of tensors
            stacked = torch.stack([w if isinstance(w, torch.Tensor) else torch.tensor(w) for w in weights])
            global_sparsity = float((stacked.abs() <= sparsity_eps).float().mean().item())
        summary_lines = [f"Classifier sparsity (|w| <= {sparsity_eps:.0e}): {global_sparsity:.4f}"]

        for idx, cls in self.idx_to_class.items():
            cls_weights = weights[idx]
            valid_k = min(k, cls_weights.numel())
            topk_vals, topk_indices = torch.topk(cls_weights, k=valid_k)
            topk_indices = topk_indices.detach().cpu().numpy()
            topk_concepts = [self.names[j] for j in topk_indices]
            nonzero = int((cls_weights.abs() > sparsity_eps).sum().item())
            analysis_str = [f"Class : {cls}", f"\tNon-zero weights: {nonzero}/{cls_weights.numel()}"]
            for j, c in enumerate(topk_concepts):
                analysis_str.append(f"\t {j+1} - {c}: {topk_vals[j]:.3f}")
            analysis_str = "\n".join(analysis_str)
            output.append(analysis_str)

            if print_lows:
                topk_vals, topk_indices = torch.topk(-cls_weights, k=valid_k)
                topk_indices = topk_indices.detach().cpu().numpy()
                topk_concepts = [self.names[j] for j in topk_indices]
                analysis_str = [f"Class : {cls}"]
                for j, c in enumerate(topk_concepts):
                    analysis_str.append(f"\t {j+1} - {c}: {-topk_vals[j]:.3f}")
                analysis_str = "\n".join(analysis_str)
                output.append(analysis_str)

        analysis = "\n".join(summary_lines + output)
        return analysis

