# Super-résolution à faible coût

Projet d'IA générative visuelle : agrandir une image basse résolution (×4) en limitant les détails inventés, avec un budget GPU de 12 Go de VRAM maximum (Google Colab).

Auteur : Alae AZZAOUI, EMSI Rabat (5e année, Intelligence Artificielle et Data Science).

## 1. Objectifs

- Créer des paires basse et haute résolution à partir d'images haute résolution.
- Comparer quatre approches : agrandissement bicubique, réseau convolutif simple, GAN et diffusion légère.
- Mesurer la qualité (PSNR, SSIM, LPIPS), la vitesse et la mémoire.
- Constituer un catalogue de cas où le modèle invente des structures.

## 2. Structure du dépôt

| Dossier | Contenu |
|---|---|
| `notebooks/` | Notebooks numérotés : données, baseline, modèle, évaluation, démo |
| `src/` | Code réutilisable : dataset, modèles, pertes, métriques, évaluation |
| `configs/` | Paramètres des expériences |
| `data/` | Fiche des données et script de téléchargement (aucune donnée lourde) |
| `results/` | Tableaux de métriques, figures et exemples générés |
| `demo/` | Application de démonstration |
| `report/` | Rapport final |
| `docs/` | Licences, fiche du modèle, réflexion éthique |

## 3. Données

- Entraînement : 300 images de DIV2K (haute résolution), choisies aléatoirement (graine 42).
- Validation : 50 images de DIV2K (validation).
- Test : Set5 et Set14 (évaluation légère, sans mélange avec l'entraînement).
- Patches haute résolution de 128 × 128 extraits à la volée, versions basse résolution obtenues par réduction bicubique ×4 (32 × 32).

Détails et licences : voir `docs/fiche_donnees.md`.

## 4. Installation et exécution

L'entraînement s'effectue sur Google Colab (GPU T4). Le code est écrit dans ce dépôt et exécuté depuis Colab.

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/alaeazzaoui/super-resolution-low-cost.git
%cd super-resolution-low-cost
!pip install -q -r requirements.txt
```

Les données et les points de sauvegarde des modèles sont conservés sur Google Drive (`MyDrive/sr_project/`) et ne sont jamais versionnés dans Git.

## 5. Méthodes

| Méthode | Description |
|---|---|
| Bicubique | Interpolation bicubique, référence de comparaison |
| EDSR léger | Réseau résiduel (8 blocs, 64 canaux), `PixelShuffle`, perte L1 |
| GAN | EDSR affiné avec un discriminateur, une perte perceptuelle VGG19 et une perte adversariale |
| Diffusion légère | À venir |

## 6. Résultats

Évaluation sur 50 patches centraux de 128 × 128 (validation), facteur ×4.

| Méthode | PSNR (dB) | SSIM | LPIPS | Temps (ms) | VRAM (Mo) |
|---|---|---|---|---|---|
| Bicubique | 25,881 | 0,7122 | 0,456 | 1,70 | n/a |
| EDSR léger | 26,407 | 0,7428 | 0,384 | 2,88 | 39,8 |
| GAN | En cours | En cours | En cours | En cours | En cours |
| Diffusion légère | À venir | À venir | À venir | À venir | À venir |

Un PSNR et un SSIM élevés indiquent une meilleure fidélité à l'image attendue, un LPIPS bas indique une meilleure proximité visuelle perçue.

## 7. Avancement

- [x] Structure du dépôt et environnement Colab
- [x] Préparation des données et dataset de patches
- [x] Baseline bicubique et métriques
- [x] Réseau EDSR léger
- [ ] GAN
- [ ] Évaluation sur Set5 et Set14
- [ ] Diffusion légère
- [ ] Catalogue des détails inventés
- [ ] Démonstration interactive
- [ ] Rapport et soutenance