# Licences et sources

## 1. Données

| Ressource | Usage | Remarque |
|---|---|---|
| DIV2K | Recherche académique | Consulter les conditions d'utilisation sur la page officielle du jeu de données |
| Set5, Set14 | Évaluation | Vérifier les conditions de la source de téléchargement utilisée |

## 2. Bibliothèques et modèles pré-entraînés

| Ressource | Usage dans le projet |
|---|---|
| PyTorch et torchvision | Entraînement et chargement des modèles |
| VGG19 (poids ImageNet de torchvision) | Perte perceptuelle du GAN |
| LPIPS | Métrique de similarité perceptuelle |
| scikit-image | PSNR et SSIM |

Les poids VGG19 proviennent d'un entraînement sur ImageNet : leur usage est limité à un cadre de recherche et d'enseignement.

## 3. Références bibliographiques

- Lim et al., « Enhanced Deep Residual Networks for Single Image Super-Resolution » (EDSR), 2017.
- Ledig et al., « Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network » (SRGAN), 2017.
- Wang et al., « ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks », 2018.
- Zhang et al., « The Unreasonable Effectiveness of Deep Features as a Perceptual Metric » (LPIPS), 2018.
- Agustsson et Timofte, « NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study », 2017.

## 4. Licence du code

Code du projet : projet académique, EMSI Rabat. Aucune licence de redistribution n'est accordée à ce stade.