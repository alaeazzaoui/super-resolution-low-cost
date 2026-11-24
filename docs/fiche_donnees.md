# Fiche des données

## 1. Jeu principal : DIV2K

| Élément | Valeur |
|---|---|
| Source | https://data.vision.ee.ethz.ch/cvl/DIV2K/ |
| Contenu | Images haute résolution (2K), format PNG |
| Références | Agustsson et Timofte, « NTIRE 2017 Challenge on Single Image Super-Resolution », 2017 |
| Usage dans le projet | Recherche académique, sans diffusion des images |

## 2. Sous-ensemble utilisé

| Partition | Nombre d'images | Origine |
|---|---|---|
| Entraînement | 300 | DIV2K_train_HR, tirage aléatoire (graine 42) |
| Validation | 50 | DIV2K_valid_HR, tirage aléatoire (graine 42) |
| Test | Set5 et Set14 | À renseigner lors de la préparation |

## 3. Préparation

- Patches haute résolution de 128 × 128 extraits à la volée (16 patches par image à l'entraînement, un patch central à la validation).
- Version basse résolution obtenue par réduction bicubique ×4 (32 × 32).
- Augmentation à l'entraînement : retournement horizontal aléatoire.
- Aucune image n'est chargée entièrement en mémoire.

## 4. Stockage

Les images sont conservées sur Google Drive (`MyDrive/sr_project/data/`) et ne sont pas versionnées dans Git.

## 5. Limites

- Taille réduite du jeu d'entraînement (300 images) : le modèle généralise moins bien qu'avec le jeu complet.
- Évaluation sur des patches centraux de 128 × 128, non représentatifs de toute l'image.
- Images naturelles (paysages, objets, scènes) : les résultats ne sont pas transposables à d'autres domaines (images médicales, documents, etc.).