# Fiche du modèle

## 1. Vue d'ensemble

| Élément | Valeur |
|---|---|
| Tâche | Super-résolution d'image unique, facteur ×4 |
| Architecture | EDSR léger : 8 blocs résiduels, 64 canaux, agrandissement par `PixelShuffle` (2 étapes ×2) |
| Entrée | Image RGB basse résolution (valeurs entre 0 et 1) |
| Sortie | Image RGB haute résolution, 4 fois plus grande en largeur et en hauteur |
| Cadre | Projet académique, EMSI Rabat |

## 2. Entraînement

| Paramètre | EDSR léger | GAN (affinage) |
|---|---|---|
| Données | 300 images DIV2K, patches 128 × 128 | Identiques |
| Taille de lot | 16 | 16 |
| Optimiseur | Adam, taux d'apprentissage 2e-4 | Adam, taux d'apprentissage 1e-4 (G et D) |
| Perte | L1 | 0,01 × L1 + perte perceptuelle VGG19 + 0,005 × perte adversariale |
| Durée | 10 époques | 10 époques |
| Matériel | GPU T4 (Google Colab) | GPU T4 (Google Colab) |

## 3. Performances (validation, 50 patches de 128 × 128)

| Méthode | PSNR (dB) | SSIM | LPIPS | Temps (ms) | VRAM (Mo) |
|---|---|---|---|---|---|
| Bicubique | 25,881 | 0,7122 | 0,456 | 1,70 | n/a |
| EDSR léger | 26,407 | 0,7428 | 0,384 | 2,88 | 39,8 |
| GAN | À compléter | À compléter | À compléter | À compléter | À compléter |

## 4. Usages prévus

- Démonstration pédagogique de la super-résolution.
- Comparaison de méthodes (bicubique, réseau convolutif, GAN, diffusion).

## 5. Usages déconseillés

- Analyse d'images où l'exactitude des détails est critique : imagerie médicale, preuve judiciaire, identification de personnes, lecture de plaques ou de textes.
- Tout usage présentant l'image agrandie comme une reproduction fidèle de la scène réelle.

## 6. Limites connues

- Entraînement court et sur un petit jeu de données (300 images) : gain modeste par rapport au bicubique.
- Évaluation sur des patches centraux, pas sur des images entières.
- Un seul entraînement par modèle : pas d'estimation de la variabilité des résultats.
- Les modèles entraînés avec une perte adversariale peuvent produire des textures plausibles mais inexactes (voir `docs/ethique.md`).
- Mesure de la mémoire GPU incluant probablement le réseau LPIPS chargé sur le GPU.

## 7. Fichiers

Les points de sauvegarde (`edsr_light.pt`, `gan_g.pt`, `gan_d.pt`) sont conservés sur Google Drive et ne sont pas versionnés dans Git.