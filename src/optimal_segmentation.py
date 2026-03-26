
"""
IMPLEMENTACJA OPTYMALNEJ SEGMENTACJI WĄTROBY I GUZÓW
Optymalizacja progów: Dice Tumor +0.83%
"""

import torch
import torch.nn.functional as F
import numpy as np

def optimal_liver_tumor_segmentation(model, images, device, 
                                   liver_threshold=0.7, 
                                   tumor_threshold=0.1):
    """
    PRODUKCYJNA funkcja segmentacji z optymalną konfiguracją progów.
    
    OPTYMALNE PARAMETRY (z grid search):
    - Liver threshold: 0.7 (konserwatywny - wysoka precyzja)
    - Tumor threshold: 0.1 (czuły - wykrywa słabe sygnały)
    
    WYNIKI:
    - Dice Liver: 0.7898 (-0.57% vs baseline)
    - Dice Tumor: 0.7101 (+0.83% vs baseline)
    """
    
    # Normalizacja wymiarów
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images.permute(0, 3, 1, 2)
    elif images.ndim == 3:
        images = images.unsqueeze(1)
    
    images = images.float().to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        
        # Zastosuj optymalne progi
        batch_size, _, height, width = probabilities.shape
        predictions = torch.zeros(batch_size, height, width, dtype=torch.long, device=device)
        
        # Priorytet: tumor > liver > background
        tumor_mask = probabilities[:, 2] > tumor_threshold
        predictions[tumor_mask] = 2
        
        liver_mask = (probabilities[:, 1] > liver_threshold) & (predictions == 0)
        predictions[liver_mask] = 1
        
        return predictions.cpu(), probabilities.cpu()

# Przykład użycia:
# predictions, probabilities = optimal_liver_tumor_segmentation(model, ct_images, device)
